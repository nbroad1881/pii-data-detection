import os
import json

from datetime import datetime
from functools import partial
from dataclasses import dataclass, field, asdict

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    set_seed,
)
from tokenizers import AddedToken
from datasets import Dataset, concatenate_datasets
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from piidd.training.utils import (
    SwapNameCallback,
    create_peft_model,
    filter_no_pii,
    compute_metrics,
    make_gt_dataframe,
)
from piidd.models.custom_deberta import (
    add_layer_drop,
    MultiSampleDebertaV2ForTokenClassification,
)
from piidd.processing.pre import strided_tokenize, add_token_map

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv("../../../.env")


@dataclass
class Config(TrainingArguments):

    model_path: str = field(
        default="roberta-base",
    )
    max_length: int = field(default=384)
    stride: int = field(default=64)
    num_proc: int = field(default=8)
    lr: float = field(default=1e-5)

    train_on_all_data: bool = field(default=False)
    remove_bi: bool = field(default=False)
    filter_no_pii_percent_allow: float = field(default=0.2)
    num_extra_samples: int = field(default=2000)

    add_newline_token: bool = field(default=True)

    use_lora: bool = field(default=False)
    model_dtype: str = field(
        default="fp16", metadata={"help": "fp32, fp16, int8, int4"}
    )
    lora_r: int = field(default=16)
    lora_modules: str = field(default="all-linear")
    lora_dropout: float = field(default=0.1)
    use_dora: bool = field(default=False)

    random_seed: bool = field(default=True)

    layer_drop_prob: float = field(default=0.0)

    use_multisample_dropout: bool = field(default=False)

    fold: int = field(default=0)

    main_dataset_path: str = field(default="train.json")
    extra_dataset_path: str = field(default=None)
    remove_classes: str = field(default="I-EMAIL,I-USERNAME")
    swap_names: bool = field(default=False)

    debugg: bool = field(default=False)
    


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:

    os.environ["WANDB_RUN_GROUP"] = cfg.wandb_run_group

    cfg = Config(
        **{
            k: v
            for k, v in OmegaConf.to_container(cfg, resolve=True).items()
            if k in asdict(Config("."))
        }
    )

    cfg.output_dir = HydraConfig.get().runtime.output_dir

    if cfg.random_seed:
        cfg.seed = datetime.now().microsecond
        set_seed(cfg.seed)

    if "," in cfg.lora_modules:
        cfg.lora_modules = cfg.lora_modules.split(",")

    cfg.run_name = None

    data = json.load(open(cfg.main_dataset_path))

    base_labels = {
        "EMAIL",
        "ID_NUM",
        "NAME_STUDENT",
        "PHONE_NUM",
        "STREET_ADDRESS",
        "URL_PERSONAL",
        "USERNAME",
    }
    all_labels = []

    if not cfg.remove_bi:
        for ll in base_labels:
            all_labels.append(f"B-{ll}")
            all_labels.append(f"I-{ll}")
    else:
        for ll in base_labels:
            all_labels.append(ll)
    all_labels.append("O")

    if cfg.remove_classes is not None:
        remove_classes = cfg.remove_classes.split(",")
        for rc in remove_classes:
            all_labels = [x for x in all_labels if rc not in x]

    all_labels = sorted(all_labels)
    label2id = {ll: i for i, ll in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}

    ds1 = Dataset.from_dict(
        {
            "full_text": [x["full_text"] for x in data],
            "document": [x["document"] for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data],
        }
    )

    ds1_train = ds1.filter(
        lambda x: int(x["document"]) % 4 != cfg.fold, num_proc=cfg.num_proc
    )
    ds1_test = ds1.filter(
        lambda x: int(x["document"]) % 4 == cfg.fold, num_proc=cfg.num_proc
    )

    if cfg.extra_dataset_path is not None:
        if cfg.extra_dataset_path.endswith(".json"):
            data2 = json.load(open(cfg.extra_dataset_path))
            ds2 = Dataset.from_dict(
                {
                    "full_text": ["" for x in data2],
                    "document": [x["document"] for x in data2],
                    "tokens": [x["tokens"] for x in data2],
                    "trailing_whitespace": [x["trailing_whitespace"] for x in data2],
                    "provided_labels": [x["labels"] for x in data2],
                }
            )
        elif cfg.extra_dataset_path.endswith(".pq"):
            ds2 = Dataset.from_parquet(cfg.extra_dataset_path)

        ds2 = ds2.shuffle().select(range(min(cfg.num_extra_samples, len(ds2))))

    if "essay" in ds2.column_names:
        ds2 = ds2.rename_column("essay", "full_text")
    if "labels" in ds2.column_names:
        ds2 = ds2.rename_column("labels", "provided_labels")

    raw_ds = concatenate_datasets([ds1_train.remove_columns(["document"]), ds2])

    if cfg.train_on_all_data:
        raw_ds = concatenate_datasets([ds1_test.remove_columns(["document"]), raw_ds])

    if cfg.debugg:
        raw_ds = raw_ds.select(range(500))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    if cfg.add_newline_token:
        # lots of newlines in the text
        # adding this should be helpful
        tokenizer.add_tokens(AddedToken("\n", normalized=False))

    raw_ds = raw_ds.filter(
        filter_no_pii,
        num_proc=cfg.num_proc,
        fn_kwargs={"percent_allow": cfg.filter_no_pii_percent_allow},
        keep_in_memory=True,
    )

    keep_cols = ["input_ids", "attention_mask", "labels"]
    remove_cols = [x for x in raw_ds.column_names if x not in keep_cols]

    ds = raw_ds.map(
        strided_tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": cfg.max_length,
            "stride": cfg.stride,
        },
        num_proc=cfg.num_proc,
        batched=True,
        batch_size=1,
        remove_columns=remove_cols,
        keep_in_memory=True,
    )

    ds1_test = ds1_test.add_column("idx", list(range(len(ds1_test))))
    val_ds = ds1_test.map(add_token_map, num_proc=cfg.num_proc, keep_in_memory=True)

    gt_df = make_gt_dataframe(val_ds)

    tokenized_val_ds = ds1_test.map(
        strided_tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": cfg.max_length,
            "stride": cfg.stride,
        },
        num_proc=cfg.num_proc,
        batched=True,
        batch_size=1,
        remove_columns=[x for x in ds1_test.column_names if x not in keep_cols],
        keep_in_memory=True,
    )

    if "deberta-v" in cfg.model_path:
        add_layer_drop(cfg.layer_drop_prob)

    if cfg.use_lora:
        model = create_peft_model(
            cfg.model_path,
            num_labels=len(all_labels),
            id2label=id2label,
            label2id=label2id,
            model_dtype=cfg.model_dtype,
            gradient_checkpointing=cfg.gradient_checkpointing,
            lora_r=cfg.lora_r,
            lora_modules=cfg.lora_modules,
            lora_alpha=cfg.lora_r * 2,
            lora_dropout=cfg.lora_dropout,
            use_multisample_dropout=cfg.use_multisample_dropout,
        )
    else:
        if cfg.use_multisample_dropout:
            model = MultiSampleDebertaV2ForTokenClassification.from_pretrained(
                cfg.model_path,
                num_labels=len(all_labels),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

        else:
            model = AutoModelForTokenClassification.from_pretrained(
                cfg.model_path,
                num_labels=len(all_labels),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    model.config.update(
        {
            "training_max_length": cfg.max_length,
            "training_stride": cfg.stride,
        }
    )

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    trainer = Trainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        eval_dataset=tokenized_val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(
            compute_metrics,
            id2label=id2label,
            tokenized_eval_ds=tokenized_val_ds,
            eval_data=val_ds,
            gt_df=gt_df,
            output_dir=cfg.output_dir,
        ),
    )

    if cfg.swap_names:
        swcb = SwapNameCallback(
            raw_ds,
            partial(
                strided_tokenize,
                tokenizer=tokenizer,
                label2id=label2id,
                max_length=cfg.max_length,
                stride=cfg.stride,
            ),
            trainer=trainer,
        )

        trainer.add_callback(swcb)

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()

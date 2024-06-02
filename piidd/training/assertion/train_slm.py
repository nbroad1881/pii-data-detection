from dataclasses import dataclass, field, asdict
import os
import json
from datetime import datetime
from pathlib import Path
from functools import partial

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from tokenizers import AddedToken
from datasets import concatenate_datasets
import wandb
from dotenv import load_dotenv
import pandas as pd
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy.special import softmax

from piidd.models.span import DebertaV2ForSpanClassification, DebertaV1ForSpanClassification
from piidd.processing.pre import create_dataset_for_span_classification, tokenize_for_span_classification



os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv("../../../.env")


START_TOKEN = "<START>"
END_TOKEN = "<END>"

@dataclass
class Config(TrainingArguments):

    model_path: str = field(
        default="roberta-base",
    )
    num_token_before: int = field(default=100)
    num_token_after: int = field(default=100)
    num_proc: int = field(default=8)

    train_on_all_data: bool = field(default=False)
    remove_bi: bool = field(default=False)
    add_newline_token: bool = field(default=True)

    random_seed: bool = field(default=True)

    layer_drop_prob: float = field(default=0.0)
    use_multisample_dropout: bool = field(default=False)

    fold: int = field(default=0)

    main_df_path: str = field(default="oof_ds.pq")
    main_dataset_path: str = field(default="train.json")
    extra_df_path: str = field(default="extra_df.pq")
    extra_dataset_path: str = field(default="extra.json")

    num_extra_samples: int = field(default=10000)

    debugg: bool = field(default=False)


def compute_metrics(eval_preds, output_dir, val_df, id2label):

    preds, label_ids = eval_preds

    preds = softmax(preds, axis=-1)

    for i in range(preds.shape[1]):

        val_df[f"{id2label[i]}"] = preds[:, i]

    preds = preds.argmax(-1)

    val_df.to_parquet(
            Path(output_dir)
            / (datetime.now().strftime("%Y%m%d%H%M%S") + "_preds.parquet"),
        )

    return {"accuracy": (preds == label_ids).mean()}


@hydra.main(config_path="conf")
def main(cfg: DictConfig):


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

    cfg.run_name = None


    data = json.load(open(cfg.main_dataset_path))
    df = pd.read_parquet(cfg.main_df_path)

    ds1 = create_dataset_for_span_classification(
        data, df, START_TOKEN, END_TOKEN, num_before=cfg.num_token_before, num_after=cfg.num_token_after
    )

    extra_data = json.load(open(cfg.extra_dataset_path))
    extra_df = pd.read_parquet(cfg.extra_df_path)

    if cfg.num_extra_samples > 0:
        extra_df = extra_df.sample(cfg.num_extra_samples, replace=True)

    ds2 = create_dataset_for_span_classification(
        extra_data, extra_df, START_TOKEN, END_TOKEN, num_before=cfg.num_token_before, num_after=cfg.num_token_after
    )

    ds = concatenate_datasets([ds1, ds2]).train_test_split(test_size=0.1, seed=cfg.seed)

    val_df = ds["test"].to_pandas()

    all_labels = sorted(df["label"].unique())
    label2id = {ll: i for i, ll in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}


    if cfg.debugg:
        ds["train"] = ds["train"].select(range(500))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    if cfg.add_newline_token:
        # lots of newlines in the text
        # adding this should be helpful
        tokenizer.add_tokens(AddedToken("\n", normalized=False))

    
    tokenizer.add_tokens([START_TOKEN, END_TOKEN])

    start_token_id, end_token_id = tokenizer.convert_tokens_to_ids([START_TOKEN, END_TOKEN])

    if "deberta-v" in cfg.model_path:
        model_class = DebertaV2ForSpanClassification
    elif "deberta" in cfg.model_path:
        model_class = DebertaV1ForSpanClassification    
    else:
        raise ValueError("Model not supported")

    
    model = model_class.from_pretrained(
        cfg.model_path,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    model.config.update(
        {
            "num_tokens_before": cfg.num_token_before,
            "num_tokens_after": cfg.num_token_after,
            "start_token_id": start_token_id,
            "end_token_id": end_token_id,
        }
    )

    tokenized = ds.map(
        tokenize_for_span_classification,
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id},
        batched=True,
        num_proc=cfg.num_proc,
        remove_columns=ds["train"].column_names,
    )

    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=cfg,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, output_dir=cfg.output_dir, val_df=val_df, id2label=id2label),
    )


    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
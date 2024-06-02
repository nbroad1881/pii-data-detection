import os
from functools import partial
from datetime import datetime
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    set_seed,
)
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
from scipy.special import softmax
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

dotenv_path = Path(__file__).parent.parent.parent / ".env"
if dotenv_path.exists():
    print("Loaded .env file!")
    load_dotenv(str(dotenv_path))


def tokenize(examples, prompt_template, tokenizer, max_length, label2id):

    texts = [
        prompt_template.format(text=t, name=n)
        for t, n in zip(examples["text"], examples["name"])
    ]

    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    tokenized["label"] = [label2id[ll] for ll in examples["label"]]

    return tokenized


def compute_metrics(eval_preds, id2label, eval_ds, output_dir):
    preds, labels = eval_preds

    scores = {}

    unq_labels = set(labels.tolist())

    for uq in unq_labels:
        mask = labels == uq
        scores[f"accuracy_{id2label[uq]}"] = (
            (preds.argmax(-1)[mask] == labels[mask]).mean().item()
        )

    to_save = []

    label_str = [id2label[i] for i in range(preds.shape[-1])]

    for p, ll, example in zip(preds, labels, eval_ds):
        to_save.append(
            (
                *[pp for pp in softmax(p, -1)],
                id2label[ll],
                example["text"],
                example["document"],
                example["name"],
            )
        )

    df = pd.DataFrame(
        to_save, columns=label_str + ["label", "text", "document", "name"]
    )

    save_dir = Path(output_dir) / wandb.run.id

    save_dir.mkdir(exist_ok=True, parents=True)

    df.to_csv(
        save_dir / (datetime.now().strftime("%Y%m%d%H%M%S") + "_preds.csv"),
        index=False,
    )

    return {"accuracy": (preds.argmax(-1) == labels).mean().item(), **scores}


@dataclass
class Config(TrainingArguments):

    model_path: str = field(
        default="microsoft/phi-2",
    )
    max_length: int = field(default=384)
    num_proc: int = field(default=8)

    remove_bi: bool = field(default=True)

    train_on_all_data: bool = field(default=False)

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
    prompt_template: str = field(default="{text}\n\nIs {name} a person's name?")

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

    df = pd.read_parquet(cfg.main_dataset_path)

    if cfg.remove_bi:
        df["label"] = [ll.split("-")[-1] for ll in df["label"]]

    all_labels = sorted(list(set(df["label"])))
    label2id = {ll: i for i, ll in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}

    ds = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=cfg.seed)

    model_kwargs = {}
    if cfg.model_dtype in {"int8", "int4"}:

        if cfg.model_dtype == "int8":
            raise NotImplementedError("int8 not supported yet")
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    if cfg.model_dtype == "fp32":
        model_kwargs["torch_dtype"] = torch.float32
    elif cfg.model_dtype == "fp16":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path,
        num_labels=len(all_labels),
        label2id=label2id,
        id2label=id2label,
        **model_kwargs,
        attn_implementation="sdpa",
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )

    config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=2 * cfg.lora_r,
        lora_dropout=cfg.lora_dropout,
        task_type="SEQ_CLS",
        target_modules=cfg.lora_modules,
        bias="none",
    )

    model = get_peft_model(model, config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model.config.pad_token_id = tokenizer.eos_token_id

    collator = DataCollatorWithPadding(tokenizer)

    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs={
            "prompt_template": cfg.prompt_template,
            "tokenizer": tokenizer,
            "max_length": cfg.max_length,
            "label2id": label2id,
        },
        num_proc=cfg.num_proc,
    )

    trainer = Trainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=partial(
            compute_metrics,
            id2label=id2label,
            eval_ds=ds["test"],
            output_dir=cfg.output_dir,
        ),
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()

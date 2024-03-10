import os
import random
from functools import partial
from datetime import datetime
from dataclasses import dataclass, field
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
    HfArgumentParser,
)
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
from scipy.special import softmax
import wandb

dotenv_path = Path(__file__).parent.parent.parent / ".env"
if dotenv_path.exists():
    print("Loaded .env file!")
    load_dotenv(str(dotenv_path))


def tokenize(examples, tokenizer, max_length, label2id):
    tokenized = tokenizer(
        examples["context"],
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

    correct = preds.argmax(-1) == labels

    to_save = []

    label_str = [id2label[i] for i in range(preds.shape[-1])]

    for c, p, ll, example in zip(correct, preds, labels, eval_ds):
        if not c:

            to_save.append(
                (*[pp for pp in softmax(p, -1)], id2label[ll], example["context"], example["document"])
            )

    df = pd.DataFrame(to_save, columns=label_str + ["label", "context", "document"])

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
        default="roberta-base",
    )
    dataset_path: str = field(
        default=os.environ["PROJECT_HOME_DIR"] + "/data/assertion_ds_v1.parquet"
    )
    max_length: int = field(default=384)
    num_proc: int = field(default=8)

    train_on_all_data: bool = field(default=False)

    filter_no_pii_percent_allow: float = field(default=0.2)

    add_newline_token: bool = field(default=True)

    use_lora: bool = field(default=False)
    model_dtype: str = field(
        default="fp16", metadata={"help": "fp32, fp16, int8, int4"}
    )
    lora_r: int = field(default=16)
    lora_modules: str = field(default="all-linear")

    random_seed: bool = field(default=True)


def main():
    parser = HfArgumentParser((Config,))

    args = parser.parse_args_into_dataclasses()[0]

    if args.random_seed:
        args.seed = datetime.now().microsecond
        set_seed(args.seed)

    if "," in args.lora_modules:
        args.lora_modules = args.lora_modules.split(",")

    args.run_name = None

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
    for ll in base_labels:
        all_labels.append(f"B-{ll}")
        all_labels.append(f"I-{ll}")
    all_labels.append("O")

    all_labels = sorted(all_labels)
    label2id = {ll: i for i, ll in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}

    model_kwargs = {}
    if args.model_dtype in {"int8", "int4"}:

        if args.model_dtype == "int8":
            raise NotImplementedError("int8 not supported yet")
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    if args.model_dtype == "fp32":
        model_kwargs["torch_dtype"] = torch.float32
    elif args.model_dtype == "fp16":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=len(all_labels),
        label2id=label2id,
        id2label=id2label,
        **model_kwargs,
    )

    if args.model_dtype in {"int8", "int4"}:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.1,
        task_type="SEQ_CLS",
        target_modules=args.lora_modules,
        bias="none",
    )

    model = get_peft_model(model, config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model.config.pad_token_id = tokenizer.eos_token_id

    collator = DataCollatorWithPadding(tokenizer)

    ds = Dataset.from_parquet(args.dataset_path).train_test_split(test_size=0.2)

    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length,
            "label2id": label2id,
        },
        num_proc=args.num_proc,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=partial(compute_metrics, id2label=id2label, eval_ds=ds["test"], output_dir=args.output_dir),
    )

    trainer.train()


if __name__ == "__main__":
    main()

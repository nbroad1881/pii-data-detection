import os
import datetime
from dataclasses import dataclass, field

from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset, Features, ClassLabel, Value, concatenate_datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "pii-dd"


def tokenize(example, tokenizer):

    return tokenizer(
        example["text"],
        padding=False,
        truncation=False,
    )


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    pred_classes = preds.argmax(-1)

    p, r, f1, _ = precision_recall_fscore_support(
        labels, pred_classes, average="binary"
    )
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": (pred_classes == labels).mean(),
    }


@dataclass
class Config(TrainingArguments):

    model_path: str = field(
        default="roberta-base",
    )
    dataset_path: str = field(default=None)

    num_proc: int = field(default=8)

    train_on_all_data: bool = field(default=False)

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

    config = parser.parse_args_into_dataclasses()[0]

    if config.random_seed:
        config.seed = datetime.datetime.now().microsecond
        set_seed(config.seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_path, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    ds = Dataset.from_parquet(config.dataset_path)
    ds = ds.cast(
        Features(
            {"label": ClassLabel(num_classes=2, names=[0, 1]), "text": Value("string")}
        )
    )

    ds = ds.train_test_split(test_size=0.1, stratify_by_column="label")

    if config.train_on_all_data:
        ds["train"] = concatenate_datasets([ds["train"], ds["test"]])

    ds = ds.map(
        tokenize,
        batched=True,
        num_proc=config.num_proc,
        fn_kwargs={"tokenizer": tokenizer},
    )

    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=16),
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()

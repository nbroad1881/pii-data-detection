import json
from pathlib import Path
from dataclasses import dataclass, field

from datasets import Dataset, concatenate_datasets
from transformers import TrainingArguments, HfArgumentParser
from span_marker import SpanMarkerModel, Trainer, SpanMarkerModelCardData

from piidd.training.utils import (
    filter_dataset_for_labels,
    filter_no_pii,
    chunk_for_spanmarker,
)


@dataclass
class Config(TrainingArguments):

    model_path: str = field(
        default="roberta-base",
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
    dataset_name: str = field(default="pii-dd+mixtral")


def main() -> None:

    parser = HfArgumentParser((Config,))

    args = parser.parse_args_into_dataclasses()[0]

    args.run_name = None

    data = json.load(open("/drive2/kaggle/pii-dd/data/train.json"))

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

    ds1 = Dataset.from_dict(
        {
            "full_text": [x["full_text"] for x in data],
            "document": [x["document"] for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data],
        }
    ).train_test_split(test_size=0.25)

    ds2 = Dataset.from_parquet("/drive2/kaggle/pii-dd/data/ds-v2.pq")

    ds2 = ds2.rename_column("essay", "full_text")
    ds2 = ds2.rename_column("labels", "provided_labels")

    raw_ds = concatenate_datasets([ds1["train"].remove_columns(["document"]), ds2])

    raw_ds = raw_ds.map(
        filter_dataset_for_labels,
        num_proc=args.num_proc,
        fn_kwargs={"labels2keep": [x for x in all_labels if "NAME_STUDENT" in x]},
    )
    ds1["test"] = ds1["test"].map(
        filter_dataset_for_labels,
        num_proc=args.num_proc,
        fn_kwargs={"labels2keep": [x for x in all_labels if "NAME_STUDENT" in x]},
    )

    raw_ds = raw_ds.filter(
        filter_no_pii,
        num_proc=args.num_proc,
        fn_kwargs={"percent_allow": args.filter_no_pii_percent_allow},
        keep_in_memory=True,
    )

    ds1["test"] = ds1["test"].filter(
        filter_no_pii,
        num_proc=args.num_proc,
        keep_in_memory=True,
    )

    raw_ds = chunk_for_spanmarker(raw_ds, args.max_length, args.num_proc)
    ds1["test"] = chunk_for_spanmarker(ds1["test"], args.max_length, args.num_proc)

    raw_ds = raw_ds.rename_column("provided_labels", "ner_tags")
    ds1["test"] = ds1["test"].rename_column("provided_labels", "ner_tags")
    labels = [x for x in all_labels if "NAME_STUDENT" in x] + ["O"]
    # ['O', 'art-broadcastprogram', 'art-film', 'art-music', 'art-other', ...

    # Initialize a SpanMarker model using a pretrained BERT-style encoder
    model_id = args.output_dir
    model = SpanMarkerModel.from_pretrained(
        args.model_path,
        labels=labels,
        # SpanMarker hyperparameters:
        model_max_length=256,
        marker_max_length=128,
        entity_max_length=8,
        is_split_into_words=True,
        # Model card arguments
        model_card_data=SpanMarkerModelCardData(
            model_id=model_id,
            encoder_id=args.model_path,
            dataset_name=args.dataset_name,
            license="cc-by-sa-4.0",
            language="en",
        ),
    )

    print(raw_ds[0])
    print(ds1["test"][0])

    # Initialize the trainer using our model, training args & dataset, and train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=raw_ds,
        eval_dataset=ds1["test"],
    )
    trainer.train()


if __name__ == "__main__":
    main()

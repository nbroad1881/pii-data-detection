import json
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import numpy as np


def tokenize(example, tokenizer, max_length, stride):
    text = []

    tokens = example["tokens"][0]
    trailing_whitespace = example["trailing_whitespace"][0]

    for t, ws in zip(tokens, trailing_whitespace):
        text.append(t)
        if ws:
            text.append(" ")

    tokenized = tokenizer(
        "".join(text),
        return_offsets_mapping=True,
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
    )

    # tokenized is now a list of lists depending on how long the input is, the max length, and the stride

    text = "".join(text)

    idxs = {}
    if "idx" in example:
        # only applies to validation data
        idxs = {"idx": [example["idx"][0]] * len(tokenized.input_ids)}

    return {**tokenized, **idxs}


def add_token_map(example):
    """
    token_map is a list of indices that map the tokenized input to the original input.
    token_map is the same length as the text, and the i-th value is the index of the token that the i-th character belongs to.

    """
    token_map = []

    idx = 0

    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):

        token_map.extend([idx] * len(t))
        if ws:
            token_map.append(-1)

        idx += 1

    return {"token_map": token_map}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="/kaggle/input/pii-detection-removal-from-educational-data/test.json",
    )
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_proc", type=int, default=2)
    parser.add_argument("--dataset_output_path", type=str, default="ds.pq")
    parser.add_argument("--tokenized_output_path", type=str, default="tds.pq")
    parser.add_argument("--preds_output_path", type=str, default="preds.npy")

    args = parser.parse_args()

    data = json.load(open(args.data_path))

    ds = Dataset.from_dict(
        {
            "full_text": [x["full_text"] for x in data],
            "document": [x["document"] for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        }
    )

    ds = ds.add_column("idx", list(range(len(ds))))
    ds = ds.map(add_token_map, num_proc=args.num_proc)

    ds.to_parquet(args.dataset_output_path)

    keep_cols = ["input_ids", "attention_mask"]
    remove_cols = [x for x in ds.column_names if x not in keep_cols]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    ds = ds.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length,
            "stride": args.stride,
        },
        num_proc=args.num_proc,
        batched=True,
        batch_size=1,
        remove_columns=remove_cols,
    )

    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    targs = TrainingArguments(
        ".",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        dataloader_num_workers=1,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(ds).predictions

    ds.to_parquet(args.tokenized_output_path)

    np.save(args.preds_output_path, predictions)


if __name__ == "__main__":
    main()

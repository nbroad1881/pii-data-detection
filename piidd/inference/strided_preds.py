import json
import argparse
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import numpy as np

from piidd.processing.pre import add_token_map, strided_tokenize

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
    parser.add_argument("--msd", type=int, default=0)
    parser.add_argument("--add_labels", type=int, default=0)

    args = parser.parse_args()

    targs = TrainingArguments(
        ".",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        dataloader_num_workers=1,
    )

    data = json.load(open(args.data_path))

    ds = Dataset.from_dict(
        {
            "full_text": [x["full_text"] for x in data],
            "document": [x["document"] for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        }
    )

    if args.add_labels:
        ds = ds.add_column("provided_labels", [x["labels"] for x in data])

    ds = ds.add_column("idx", list(range(len(ds))))

    with targs.main_process_first(desc="Adding token map"):
        ds = ds.map(add_token_map, num_proc=args.num_proc)

    Path(args.dataset_output_path).parent.mkdir(exist_ok=True, parents=True)
    ds.to_parquet(args.dataset_output_path)

    keep_cols = ["input_ids", "attention_mask"]
    remove_cols = [x for x in ds.column_names if x not in keep_cols]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)


    if args.msd:
        print("using msd model")

        from piidd.models.layerdrop_deberta import MultiSampleDebertaV2ForTokenClassification
        model_class = MultiSampleDebertaV2ForTokenClassification
    else:
        model_class = AutoModelForTokenClassification

    model = model_class.from_pretrained(args.model_path)

    with targs.main_process_first(desc="Tokenizing"):
        ds = ds.map(
            strided_tokenize,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": args.max_length,
                "stride": args.stride,
                "label2id": model.config.label2id,
            },
            num_proc=args.num_proc,
            batched=True,
            batch_size=1,
            remove_columns=remove_cols,
        )

    trainer = Trainer(
        model=model,
        args=targs,
    )

    predictions = trainer.predict(ds).predictions

    ds.to_parquet(args.tokenized_output_path)

    np.save(args.preds_output_path, predictions)


if __name__ == "__main__":
    main()

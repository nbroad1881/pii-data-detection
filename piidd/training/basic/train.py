import os
import json
import argparse
import random
from itertools import chain
from functools import partial

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from tokenizers import AddedToken
import evaluate
from datasets import Dataset, concatenate_datasets
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "pii-dd"

# https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/468844
def filter_no_pii(example, percent_allow=0.2):
    # Return True if there is PII
    # Or 20% of the time if there isn't

    has_pii = set("O") != set(example["provided_labels"])

    return has_pii or (random.random() < percent_allow)


def tokenize(example, tokenizer, label2id, max_length):
    text = []
    labels = []

    for t, l, ws in zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        labels.extend([l] * len(t))
        if ws:
            text.append(" ")
            labels.append("O")

    tokenized = tokenizer(
        "".join(text), return_offsets_mapping=True, max_length=max_length, truncation=True,
    )

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        while start_idx >= len(labels):
            start_idx -= 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {**tokenized, "labels": token_labels, "length": length}


def compute_metrics(p, metric, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value
    return final_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--lr", type=float)

    args = parser.parse_args()

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
    for l in base_labels:
        all_labels.append(f"B-{l}")
        all_labels.append(f"I-{l}")
    all_labels.append("O")

    all_labels = sorted(all_labels)
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}

    ds1 = Dataset.from_dict(
        {
            "full_text": [x["full_text"] for x in data],
            # "document": [x["document"] for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data],
        }
    ).train_test_split(test_size=0.2)

    

    ds2 = Dataset.from_parquet("/drive2/kaggle/pii-dd/data/ds-v2.pq")

    ds2 = ds2.rename_column("essay", "full_text")
    ds2 = ds2.rename_column("labels", "provided_labels")

    ds = concatenate_datasets([ds1["train"], ds2])
    # ds = ds1["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # lots of newlines in the text
    # adding this should be helpful
    tokenizer.add_tokens(AddedToken("\n", normalized=False))

    ds = ds.filter(
        filter_no_pii,
        num_proc=args.num_proc,
    )

    ds = ds.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": args.max_length,
        },
        num_proc=args.num_proc,
    )

    val_ds = ds1["test"].filter(
        filter_no_pii,
        num_proc=args.num_proc,
    ).map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": args.max_length,
        },
        num_proc=args.num_proc,
    )

    metric = evaluate.load("seqeval")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    args = TrainingArguments(
        "utc-dv3l-xd-v2",
        fp16=True,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        report_to="wandb",
        evaluation_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        logging_steps=15,
        metric_for_best_model="overall_recall",
        greater_is_better=True,
        gradient_checkpointing=True,
        num_train_epochs=3,
        dataloader_num_workers=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metric=metric, all_labels=all_labels),
    )

    trainer.train()


if __name__ == "__main__":
    main()

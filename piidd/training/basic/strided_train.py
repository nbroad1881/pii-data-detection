import os
import json

from datetime import datetime
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    set_seed,
)
from tokenizers import AddedToken
from datasets import Dataset, concatenate_datasets
import numpy as np
import pandas as pd

from piidd.training.utils import SwapNameCallback, create_peft_model, filter_no_pii
from piidd.models.layerdrop_deberta import add_layer_drop, MultiSampleDebertaV2ForTokenClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize(example, tokenizer, label2id, max_length, stride):
    text = []
    char_labels = []

    tokens = example["tokens"][0]
    provided_labels = example["provided_labels"][0]
    trailing_whitespace = example["trailing_whitespace"][0]

    for t, label, ws in zip(tokens, provided_labels, trailing_whitespace):
        text.append(t)
        char_labels.extend([label] * len(t))
        if ws:
            text.append(" ")
            char_labels.append("O")

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

    char_labels = np.array(char_labels)

    text = "".join(text)
    token_labels = (
        np.ones((len(tokenized.input_ids), max_length), dtype=np.int32) * label2id["O"]
    )

    for i in range(len(tokenized.input_ids)):

        for j, (start_idx, end_idx) in enumerate(tokenized.offset_mapping[i]):
            # CLS token
            if start_idx == 0 and end_idx == 0:
                continue

            # case when token starts with whitespace
            while (
                start_idx < end_idx
                and start_idx < len(text)
                and text[start_idx].isspace()
            ):
                start_idx += 1

            # the whole token might be whitespace
            if start_idx >= end_idx:
                continue

            start_idx = min(start_idx, len(char_labels) - 1)

            token_labels[i, j] = label2id[char_labels[start_idx]]

    idxs = {}
    if "idx" in example:
        # only applies to validation data
        idxs = {"idx": [example["idx"][0]] * len(tokenized.input_ids)}

    return {**tokenized, "labels": token_labels, **idxs}


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


def make_gt_dataframe(eval_data):

    all_token_ids = []
    all_labels = []
    all_token_texts = []
    all_doc_idxs = []
    for tokens, labels, doc_idx in zip(
        eval_data["tokens"], eval_data["provided_labels"], eval_data["document"]
    ):
        for i, label in enumerate(labels):
            if label != "O":
                all_token_ids.append(i)
                all_labels.append(label)
                all_token_texts.append(tokens[i])
                all_doc_idxs.append(doc_idx)

    df = pd.DataFrame(
        {
            "token": all_token_ids,
            "label": all_labels,
            "document": all_doc_idxs,
            "token_text": all_token_texts,
        }
    )

    df["row_id"] = range(len(df))

    return df


def make_pred_df(all_preds, doc_idxs, all_tokens):
    """
    all_preds is a list of lists of tuples

    """

    all_token_ids = []
    all_labels = []
    all_doc_idxs = []
    token_texts = []

    for preds, doc_idx, tokens in zip(all_preds, doc_idxs, all_tokens):
        for token_id, label in preds:
            if tokens[token_id].isspace():
                continue

            all_token_ids.append(token_id)
            all_labels.append(label)
            all_doc_idxs.append(doc_idx)
            token_texts.append(tokens[token_id])

    df = pd.DataFrame(
        {
            "token": all_token_ids,
            "label": all_labels,
            "document": all_doc_idxs,
            "token_text": token_texts,
        }
    )

    df["row_id"] = range(len(df))

    return df


def get_token_preds(char_preds, token_map, id2label, threshold=0.9):
    # pair of (token_id, pooled_scores)
    token_preds = []

    # go through token map
    # find char indices for a token
    # get preds for that span
    current = token_map[0]
    start_idx = 0
    i = 1
    while i < len(token_map):
        # both -1 --> continue
        if current == -1 and token_map[i] == -1:
            i += 1
            continue
        # current is -1, token_map[i] is not --> track start index, set current to new value
        if current == -1 and token_map[i] != -1:
            start_idx = i
            current = token_map[i]
            i += 1

        # current is not -1, token_map[i] is -1 --> keep moving forward, don't change start_idx or current
        elif current != -1 and token_map[i] == current:
            i += 1

        # current is not -1, token_map[i] is -1
        elif current != token_map[i]:
            end_idx = i

            pooled_scores = char_preds[start_idx:end_idx].max(0)
            # pooled_scores.shape == (len(all_labels),)

            # can do thresholding here

            top_ids = pooled_scores.argsort()[::-1]  # reverse to go from high to low

            if id2label[top_ids[0]] == "O":
                if pooled_scores[top_ids[0]] < threshold:
                    token_preds.append((current, id2label[top_ids[1]]))
            else:
                token_preds.append((current, id2label[top_ids[0]]))

            start_idx = i
            current = token_map[i]
            i += 1

    # check the last token
    if current != -1 and start_idx < len(token_map):
        if len(token_preds) == 0 or token_preds[-1][0] != current:

            end_idx = len(token_map)

            pooled_scores = char_preds[start_idx:end_idx].max(0)

            top_ids = pooled_scores.argsort()[::-1]

            if id2label[top_ids[0]] == "O":
                if pooled_scores[top_ids[0]] < threshold:
                    token_preds.append((current, id2label[top_ids[1]]))
            else:
                token_preds.append((current, id2label[top_ids[0]]))

    return token_preds


def pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5):
    """
    Parameters:
    - pred_df (DataFrame): DataFrame containing predicted PII labels.
    - gt_df (DataFrame): DataFrame containing ground truth PII labels.
    - beta (float): The beta parameter for the F-beta score, controlling the trade-off between precision and recall.

    Returns:
    - float: Micro F-beta score.
    """

    df = pred_df.merge(
        gt_df, how="outer", on=["document", "token"], suffixes=("_pred", "_gt")
    )
    df["cm"] = ""

    df.loc[df.label_gt.isna(), "cm"] = "FP"
    df.loc[df.label_pred.isna(), "cm"] = "FN"

    # df.loc[(df.label_gt.notna()) & (df.label_gt != df.label_pred), "cm"] = "FNFP"
    df.loc[
        (df.label_gt.notna() & df.label_pred.notna()) & (df.label_gt != df.label_pred),
        "cm",
    ] = "FNFP"  # CHANGED

    df.loc[
        (df.label_pred.notna())
        & (df.label_gt.notna())
        & (df.label_gt == df.label_pred),
        "cm",
    ] = "TP"

    FP = (df["cm"].isin({"FP", "FNFP"})).sum()
    FN = (df["cm"].isin({"FN", "FNFP"})).sum()
    TP = (df["cm"] == "TP").sum()
    s_micro = (1 + (beta**2)) * TP / (((1 + (beta**2)) * TP) + ((beta**2) * FN) + FP)

    df.to_csv(
        Path(output_dir) / (datetime.now().strftime("%Y%m%d%H%M%S") + "_pii_fbeta.csv")
    )

    return s_micro


def compute_metrics(
    eval_preds, all_labels, id2label, tokenized_eval_ds, eval_data, gt_df, output_dir
):
    """
    tokenized_eval_ds has multiple rows per sample (due to stride and overflow)

    eval_data has one row per sample
    """

    predictions, labels = eval_preds

    current_idx = tokenized_eval_ds[0]["idx"]
    char_preds = np.zeros((len(eval_data[current_idx]["full_text"]), len(all_labels)))

    threshold = 0.9

    all_preds = []

    for j, idx in enumerate(tokenized_eval_ds["idx"]):
        # j is the index of the tokenized_eval_ds
        # idx is the index of the eval_data

        if idx != current_idx:
            token_map = eval_data[current_idx]["token_map"]

            token_preds = get_token_preds(
                char_preds, token_map, id2label, threshold=threshold
            )

            all_preds.append(token_preds)

            current_idx = idx

            char_preds = np.zeros(
                (len(eval_data[current_idx]["full_text"]), len(all_labels))
            )

        for (start_idx, end_idx), p in zip(
            tokenized_eval_ds[j]["offset_mapping"], predictions[j]
        ):

            if start_idx + end_idx == 0:
                continue

            char_preds[start_idx:end_idx] = np.maximum(char_preds[start_idx:end_idx], p)

    if char_preds.sum() > 0:
        token_map = eval_data[current_idx]["token_map"]

        token_preds = get_token_preds(
            char_preds, token_map, id2label, threshold=threshold
        )

        all_preds.append(token_preds)

    pred_df = make_pred_df(all_preds, eval_data["document"], eval_data["tokens"])

    f5 = pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5)

    return {
        "f5_score": f5,
    }


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

    filter_no_pii_percent_allow: float = field(default=0.2)

    add_newline_token: bool = field(default=True)

    use_lora: bool = field(default=False)
    model_dtype: str = field(
        default="fp16", metadata={"help": "fp32, fp16, int8, int4"}
    )
    lora_r: int = field(default=16)
    lora_modules: str = field(default="all-linear")

    random_seed: bool = field(default=True)

    layer_drop_prob: float = field(default=0.0)

    use_multisample_dropout: bool = field(default=False)

    fold: int = field(default=0)

def main():
    parser = HfArgumentParser((Config,))

    args = parser.parse_args_into_dataclasses()[0]

    if args.random_seed:
        args.seed = datetime.now().microsecond
        set_seed(args.seed)

    if "," in args.lora_modules:
        args.lora_modules = args.lora_modules.split(",")

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

    ds1_train = ds1.filter(lambda x: int(x["document"]) % 4 != args.fold, num_proc=args.num_proc)
    ds1_test = ds1.filter(lambda x: int(x["document"]) % 4 == args.fold, num_proc=args.num_proc)

    ds2 = Dataset.from_parquet("/drive2/kaggle/pii-dd/data/ds-v2.pq")

    ds2 = ds2.rename_column("essay", "full_text")
    ds2 = ds2.rename_column("labels", "provided_labels")

    raw_ds = concatenate_datasets([ds1_train.remove_columns(["document"]), ds2])

    if args.train_on_all_data:
        raw_ds = concatenate_datasets(
            [ds1_test.remove_columns(["document"]), raw_ds]
        )

    # raw_ds = raw_ds.select(range(500))

    # ds = ds1_train

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.add_newline_token:
        # lots of newlines in the text
        # adding this should be helpful
        tokenizer.add_tokens(AddedToken("\n", normalized=False))

    raw_ds = raw_ds.filter(
        filter_no_pii,
        num_proc=args.num_proc,
        fn_kwargs={"percent_allow": args.filter_no_pii_percent_allow},
        keep_in_memory=True,
    )

    keep_cols = ["input_ids", "attention_mask", "labels"]
    remove_cols = [x for x in raw_ds.column_names if x not in keep_cols]

    ds = raw_ds.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": args.max_length,
            "stride": args.stride,
        },
        num_proc=args.num_proc,
        batched=True,
        batch_size=1,
        remove_columns=remove_cols,
        keep_in_memory=True,
    )

    ds1_test = ds1_test.filter(
        filter_no_pii,
        num_proc=args.num_proc,
        keep_in_memory=True,
    )

    ds1_test = ds1_test.add_column("idx", list(range(len(ds1_test))))
    val_ds = ds1_test.map(add_token_map, num_proc=args.num_proc, keep_in_memory=True)

    gt_df = make_gt_dataframe(val_ds)

    tokenized_val_ds = ds1_test.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": args.max_length,
            "stride": args.stride,
        },
        num_proc=args.num_proc,
        batched=True,
        batch_size=1,
        remove_columns=[x for x in ds1_test.column_names if x not in keep_cols],
        keep_in_memory=True,
    )

    if "deberta-v" in args.model_path:
        add_layer_drop(args.layer_drop_prob)

    if args.use_lora:
        model = create_peft_model(
            args.model_path,
            num_labels=len(all_labels),
            id2label=id2label,
            label2id=label2id,
            model_dtype=args.model_dtype,
            gradient_checkpointing=args.gradient_checkpointing,
            lora_r=args.lora_r,
            lora_modules=args.lora_modules,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.1,
            use_multisample_dropout=args.use_multisample_dropout,
        )
    else:
        if args.use_multisample_dropout:
            model = MultiSampleDebertaV2ForTokenClassification.from_pretrained(
                args.model_path,
                num_labels=len(all_labels),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

        else:
            model = AutoModelForTokenClassification.from_pretrained(
                args.model_path,
                num_labels=len(all_labels),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    args.fp16 = True
    args.learning_rate = args.lr
    args.weight_decay = 0.01
    args.report_to = ["wandb"]
    args.evaluation_strategy = "epoch"
    args.save_total_limit = 1
    args.logging_steps = 50
    args.metric_for_best_model = "f5_score"
    args.greater_is_better = True
    args.dataloader_num_workers = 1
    args.lr_scheduler_type = "cosine"

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=tokenized_val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(
            compute_metrics,
            all_labels=all_labels,
            id2label=id2label,
            tokenized_eval_ds=tokenized_val_ds,
            eval_data=val_ds,
            gt_df=gt_df,
            output_dir=args.output_dir,
        ),
    )

    swcb = SwapNameCallback(
        raw_ds,
        partial(
            tokenize,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            stride=args.stride,
        ),
        trainer=trainer,
    )

    trainer.add_callback(swcb)

    trainer.train()


if __name__ == "__main__":
    main()

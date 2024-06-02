import argparse
import json

from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import pandas as pd
from scipy.special import softmax

from piidd.models.span import (
    DebertaV2ForSpanClassification,
    DebertaV1ForSpanClassification,
)
from piidd.processing.pre import (
    create_dataset_for_span_classification,
    tokenize_for_span_classification,
)


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="/kaggle/input/pii-detection-removal-from-educational-data/test.json",
    )
    parser.add_argument("--df_path", type=str)
    parser.add_argument("--upper_threshold", type=float, default=0.999)
    parser.add_argument("--lower_threshold", type=float, default=0.05)
    parser.add_argument("--model_path", type=str, default="deberta-v2-xlarge")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--num_proc",
        type=int,
        default=2,
    )
    parser.add_argument("--output_path", type=str)

    return parser.parse_args()


def main():

    args = get_args()

    df = pd.read_parquet(args.df_path)

    pred_cols = [
        "B-EMAIL",
        "B-ID_NUM",
        "B-NAME_STUDENT",
        "B-PHONE_NUM",
        "B-STREET_ADDRESS",
        "B-URL_PERSONAL",
        "B-USERNAME",
        "I-ID_NUM",
        "I-NAME_STUDENT",
        "I-PHONE_NUM",
        "I-STREET_ADDRESS",
    ]

    df = (
        df[
            (
                (df[pred_cols].max(axis=1) > args.lower_threshold)
                | (df["O"] < args.upper_threshold)
            )
            & (df[pred_cols + ["O"]].sum(axis=1) > 0.0001)
        ]
        .copy()
        .reset_index(drop=True)
    )

    data = json.load(open(args.data_path))

    model_config = AutoConfig.from_pretrained(args.model_path)

    if "deberta-v" in model_config.model_type:
        model_class = DebertaV2ForSpanClassification
    elif "deberta" in model_config.model_type:
        model_class = DebertaV1ForSpanClassification
    else:
        raise ValueError("Model not supported")

    START_TOKEN = "<START>"
    END_TOKEN = "<END>"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    start_token_id, end_token_id = tokenizer.convert_tokens_to_ids(
        [START_TOKEN, END_TOKEN]
    )

    model = model_class.from_pretrained(
        args.model_path, start_token_id=start_token_id, end_token_id=end_token_id
    )

    num_tokens_before = model.config.num_tokens_before
    num_tokens_after = model.config.num_tokens_after


    ds = create_dataset_for_span_classification(
        data,
        df,
        start_token=START_TOKEN,
        end_token=END_TOKEN,
        num_before=num_tokens_before,
        num_after=num_tokens_after,
    )

    ds = ds.map(
        tokenize_for_span_classification,
        batched=True,
        num_proc=args.num_proc,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": model.config.label2id,
        },
    )

    targs = TrainingArguments(
        ".",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        dataloader_num_workers=1,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print(ds)

    preds = softmax(trainer.predict(ds).predictions, axis=-1)

    id2label = model.config.id2label

    for id, label in id2label.items():
        df[label] = preds[:, id]

    df.to_parquet(args.output_path)


if __name__ == "__main__":
    main()

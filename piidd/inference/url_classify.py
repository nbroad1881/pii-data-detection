import argparse
import json

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
import numpy as np
from scipy.special import softmax

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_tokens_before", type=int, default=100)
    parser.add_argument("--num_tokens_after", type=int, default=100)
    parser.add_argument("--start_delimiter", type=str, default="[URL_START]")
    parser.add_argument("--end_delimiter", type=str, default="[URL_END]")
    parser.add_argument("--output_path", type=str, default="url_preds.parquet")

    return parser.parse_args()


def make_dataset(args):

    data = json.load(open(args.dataset_path))
    texts, doc_ids, token_ids = [], [], []

    for d in data:
        max_idx = len(d["tokens"])

        for i, t in enumerate(d["tokens"]):
            if "http" in t or "://" in t or "www." in t:
                before_tokens = d["tokens"][max(0, i - args.num_tokens_before) : i]
                before_text = ""
                for b, bws in zip(
                    before_tokens,
                    d["trailing_whitespace"][max(0, i - args.num_tokens_before) : i],
                ):
                    before_text += b
                    if bws:
                        before_text += " "

                if i + 1 >= max_idx:
                    after = []
                else:
                    after = d["tokens"][i + 1 : i + args.num_tokens_after]

                    after_text = ""
                    for a, aws in zip(
                        after,
                        d["trailing_whitespace"][i + 1 : i + args.num_tokens_after],
                    ):
                        after_text += a
                        if aws:
                            after_text += " "

                excerpt = " ".join(
                    [before_text]
                    + [args.start_delimiter]
                    + [t]
                    + [args.end_delimiter]
                    + [after_text]
                )

                texts.append(excerpt)
                doc_ids.append(d["document"])
                token_ids.append(i)

    return Dataset.from_dict(
        {
            "text": texts,
            "doc_id": doc_ids,
            "token_id": token_ids,
        }
    )


def tokenize(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        padding=False,
        truncation=False,
    )

    tokenized["length"] = [len(t) for t in tokenized.input_ids]

    return tokenized


def main():

    args = get_args()

    ds = make_dataset(args)

    model_paths = args.model_paths.split(",")

    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])

    ds = ds.map(
        tokenize,
        batched=True,
        num_proc=args.num_proc,
        fn_kwargs={"tokenizer": tokenizer},
    )

    ds = ds.sort("length")

    targs = TrainingArguments(
        ".",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        dataloader_num_workers=1,
    )

    all_preds = []
    for mp in model_paths:

        model = AutoModelForSequenceClassification.from_pretrained(
            mp,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=targs,
            data_collator=DataCollatorWithPadding(tokenizer),
        )

        preds = softmax(trainer.predict(ds).predictions, -1)

        all_preds.append(preds)

    average_preds = np.stack(all_preds).mean(axis=0)

    ds = ds.add_column("pred", [x for x in average_preds[:, 1]])

    ds.to_parquet(args.output_path)



if __name__ == "__main__":
    main()

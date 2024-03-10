import json
import os
from dotenv import load_dotenv
from pathlib import Path
import random
from typing import Tuple
from collections import Counter

import pandas as pd
from datasets import Dataset, concatenate_datasets

dotenv_path = Path(__file__).parent.parent.parent / ".env"
if dotenv_path.exists():
    print("Loaded .env file!")
    load_dotenv(str(dotenv_path))


data = json.load(open(Path(os.environ["PROJECT_HOME_DIR"]) / "data/train.json"))

generated_data = json.load(open(Path(os.environ["PROJECT_HOME_DIR"]) / "data/mixtral-8x7b-v1/mixtral-8x7b-v1.json"))

def find_false_positives(
    glob_path: Path,
    num_tokens_before: int,
    num_tokens_after: int,
    indicators: Tuple[str, str],
    threshold: float = 0.9,
):

    files = glob_path.rglob("*.csv")

    contexts, target_tokens, labels, docs = [], [], [], []

    unq_pairs = set()

    for file in files:
        df = pd.read_csv(file, engine="python")

        FP = (df["cm"].isin({"FP", "FNFP"})).sum()
        FN = (df["cm"].isin({"FN", "FNFP"})).sum()
        TP = (df["cm"] == "TP").sum()
        beta = 5
        s_micro = (
            (1 + (beta**2)) * TP / (((1 + (beta**2)) * TP) + ((beta**2) * FN) + FP)
        )

        if s_micro < threshold:
            continue

        df = df[df.cm == "FP"]



        for token_idx, doc_id in zip(df.token, df.document):
            
            temp = [d for d in data if d["document"] == doc_id]
            if len(temp) == 0:
                continue

            d = temp[0]


            if len(d["tokens"][token_idx]) < 2 or (doc_id, token_idx) in unq_pairs:
                continue

            ctx, tok, lbl = add_example(
                token_idx,
                d["tokens"],
                d["labels"],
                d["trailing_whitespace"],
                num_tokens_before,
                num_tokens_after,
                indicators,
            )

            contexts.append(ctx)
            target_tokens.append(tok)
            labels.append(lbl)
            docs.append(str(doc_id))

            unq_pairs.add((doc_id, token_idx))

    dataset = Dataset.from_dict(
        {"context": contexts, "target": target_tokens, "label": labels, "document": docs}
    )

    return dataset


def add_example(
    idx: int,
    tokens: list[str],
    labels: list[str],
    trailing_whitespace: list[bool],
    num_tokens_before: int,
    num_tokens_after: int,
    indicators: Tuple[str, str] = ("[START]", "[END]"),
):
    max_idx = len(tokens)

    start_idx = max(0, idx - num_tokens_before)
    end_idx = min(max_idx, idx + num_tokens_after)

    span = ""

    for j in range(start_idx, idx):
        span += tokens[j]
        if trailing_whitespace[j]:
            span += " "

    token = tokens[idx]
    if indicators:
        if len(span) and span[-1] not in {" ", "\n"}:
            span += " "
        span += indicators[0] + " " + tokens[idx] + " " + indicators[1] + " "

        token = indicators[0] + " " + token + " " + indicators[1]

    if idx + 1 < end_idx:
        for j in range(idx + 1, end_idx):
            span += tokens[j]
            if trailing_whitespace[j]:
                span += " "

    return span, token, labels[idx]


def create_dataset(
    output_path: Path,
    false_positive_path: Path,
    num_tokens_before: int = 50,
    num_tokens_after: int = 50,
    indicators: Tuple[str, str] = ("[START]", "[END]"),
):

    contexts, target_tokens, labels, docs = [], [], [], []
    for d in data+generated_data:
        max_idx = len(d["tokens"])

        for i, ll in enumerate(d["labels"]):

            if ll != "O":
                ctx, tok, lbl = add_example(
                    i,
                    d["tokens"],
                    d["labels"],
                    d["trailing_whitespace"],
                    num_tokens_before,
                    num_tokens_after,
                    indicators,
                )

                contexts.append(ctx)
                target_tokens.append(tok)
                labels.append(lbl)
                docs.append(str(d["document"]))

                # Add a negative example nearby 10% of the time

                if random.random() > 0.05:
                    continue

                start_idx = max(0, i - num_tokens_before)
                end_idx = min(max_idx, i + num_tokens_after)
                options = [
                    j for j in range(start_idx, end_idx) if d["labels"][j] == "O"
                ]

                if options:
                    idx = random.choice(options)

                    ctx, tok, lbl = add_example(
                        idx,
                        d["tokens"],
                        d["labels"],
                        d["trailing_whitespace"],
                        num_tokens_before,
                        num_tokens_after,
                        indicators,
                    )

                    contexts.append(ctx)
                    target_tokens.append(tok)
                    labels.append(lbl)
                    docs.append(str(d["document"]))

    dataset = Dataset.from_dict(
        {"context": contexts, "target": target_tokens, "label": labels, "document": docs}
    )

    fp_dataset = find_false_positives(
        false_positive_path, num_tokens_before, num_tokens_after, indicators
    )

    print("fp dataset size:", len(fp_dataset))

    full_ds = concatenate_datasets([dataset, fp_dataset])

    print("full ds size:", len(full_ds))

    print(Counter(full_ds["label"]))

    full_ds.to_parquet(output_path)


if __name__ == "__main__":

    create_dataset(
        Path(os.environ["PROJECT_HOME_DIR"]) / "data" / "assertion_ds_v1.parquet",
        Path(os.environ["PROJECT_HOME_DIR"]) / "piidd" / "training" / "basic",
    )

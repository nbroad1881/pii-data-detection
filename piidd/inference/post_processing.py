import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.special import softmax


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

    return s_micro


def make_gt_dataframe(eval_data):

    all_token_ids = []
    all_labels = []
    all_token_texts = []
    all_doc_idxs = []
    for tokens, labels, doc_idx in zip(
        eval_data["tokens"], eval_data["provided_labels"], eval_data["document"]
    ):
        for i, l in enumerate(labels):
            if l != "O":
                all_token_ids.append(i)
                all_labels.append(l)
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


def get_preds(
    predictions, all_labels, id2label, tokenized_eval_ds, eval_data, threshold=0.9
):

    current_idx = tokenized_eval_ds[0]["idx"]
    char_preds = np.zeros((len(eval_data[current_idx]["full_text"]), len(all_labels)))

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

    return make_pred_df(all_preds, eval_data["document"], eval_data["tokens"])


def compute_metrics(
    eval_preds, all_labels, id2label, tokenized_eval_ds, eval_data, gt_df, output_dir
):
    """
    tokenized_eval_ds has multiple rows per sample (due to stride and overflow)

    eval_data has one row per sample
    """

    predictions, labels = eval_preds

    pred_df = get_preds(
        softmax(predictions, -1),
        all_labels,
        id2label,
        tokenized_eval_ds,
        eval_data,
        threshold=0.9,
    )

    f5 = pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5)

    return f5


def add_repeated_names(df, data):
    """
    I noticed that during training that the model would often miss names that were repeated later in the essay.
    This will add any name that is repeated and add that as a prediction.

    The name must be capitalized and longer than 1 character to be added.

    Potentially has room for improvement if considering prediction scores.
    """

    tokens2add, labels2add, doc2add, token_texts = [], [], [], []
    for d in data:
        doc = d["document"]

        filtered = df[df["document"] == doc].reset_index(drop=True)

        if len(filtered) > 0:
            names = filtered[filtered["label"].str.contains("NAME_STUDENT")]

            if len(names) > 0:

                id2label = {i: l for i, l in names[["token", "label"]].values}

                name2label = {}

                for id_, label in id2label.items():
                    name2label[d["tokens"][id_]] = label

                for token_id, token_str in enumerate(d["tokens"]):

                    if token_str in name2label and token_id not in id2label:

                        # must start with a capital, must be longer than 1 character
                        # (maybe a "-" got labeled as a name)
                        if token_str.istitle() and len(token_str) > 1:
                            tokens2add.append(token_id)
                            labels2add.append(name2label[token_str])
                            doc2add.append(doc)
                            token_texts.append(token_str)

    if len(tokens2add) > 0:
        temp = pd.DataFrame(
            {
                "token": tokens2add,
                "label": labels2add,
                "document": doc2add,
                "token_text": token_texts
            }
        )
        df = pd.concat([df, temp], ignore_index=True)

    return df


def add_emails_and_urls(df, data):
    """
    Spacy has a built-in tokenizer that can detect emails and urls.

    This will add any emails and certain urls that are detected to the predictions.
    Initial tests on public LB suggest this slightly hurts the score.
    """

    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    from spacy.lang.en import English

    spacy_tokenizer = English().tokenizer

    like_urls = []
    like_email = []
    for d in data:
        for i, token in enumerate(spacy_tokenizer(d["full_text"])):
            if token.like_url:
                if "http" in token.text or "://" in token.text:
                    if "facebook.com" in token.text or "linkedin" in token.text:
                        if len(token.text) < 50:
                            like_urls.append(
                                (d["document"], i, "B-URL_PERSONAL", token.text)
                            )

            if re.search(pattern, token.text):
                like_email.append((d["document"], i, "B-EMAIL", token.text))

    if len(like_urls) + len(like_email) > 0:

        temp_df = pd.DataFrame(
            columns=["document", "token", "label", "token_text"],
            data=like_urls + like_email,
        )

        df = pd.concat([df, temp_df], ignore_index=True).drop_duplicates(
            subset=["document", "token", "label"]
        )


    return df

def check_name_casing(pred_df):
    """
    If a token is labeled as a name, it should be capitalized.
    It should not be all caps, just the first letter.
    """

    pred_df["bad_name_casing"] = ["NAME_STUDENT" in l and not t.istitle() for t, l in zip(pred_df["token_text"], pred_df["label"])]

    return pred_df[~pred_df["bad_name_casing"]].reset_index(drop=True)


def correct_preds(pred_df):
    """
    If a prediction is made with "I-NAME_STUDENT" without another "I-NAME_STUDENT" or "B-NAME_STUDENT" next to it, then it gets fixed to "B-NAME_STUDENT"

    This works well when a "B-NAME_STUDENT" is mistakenly predicted as "I-NAME_STUDENT"
    """

    new_labels = []

    for d in pred_df["document"].unique():

        filtered = pred_df[pred_df["document"] == d].reset_index(drop=True)

        if len(filtered) > 0:
            filtered = filtered.sort_values("token", ascending=True)


            for idx, (doc_idx, pred_label, row_id) in enumerate(filtered[["token", "label", "row_id"]].values):
                if pred_label == "I-NAME_STUDENT":
                    if idx == 0:
                        new_labels.append((row_id, "B-NAME_STUDENT"))
                    else:
                        # if the previous token is not next to this token (by token id)
                        # OR if the label i
                        if filtered["token"][idx-1] != doc_idx -1 or filtered["label"][idx-1] not in {"B-NAME_STUDENT", "I-NAME_STUDENT"}:
                            new_labels.append((row_id, "B-NAME_STUDENT"))
    
    for (row_id, new_label) in new_labels:
        pred_df.loc[pred_df["row_id"] == row_id, "label"] = new_label

    return pred_df


def check_urls(pred_df):
    """
    Rules:
    - urls should not have .gov, .edu
    - coursera.org should not be predicted
    - wikipedia.org should not be predicted
    - must have one of the following (http, ://, www.)
    """

    bad_urls = []

    for i, (doc, label, token_text) in enumerate(zip(pred_df[["document", "label", "token_text"]].values)):
        if label == "B-URL_PERSONAL":
            if "coursera.org" in token_text or "wikipedia.org" in token_text:
                bad_urls.append((doc, i))
            elif ".gov" in token_text or ".edu" in token_text:
                 bad_urls.append((doc, i))
            elif not any(["http" in token_text, "://" in token_text, "www." in token_text]):
                 bad_urls.append((doc, i))
        elif label == "I-URL_PERSONAL":
             bad_urls.append((doc, i))
    
    doc_matches = pred_df["document"].isin([b[0] for b in bad_urls])
    token_matches = pred_df.index.isin([b[1] for b in bad_urls])


    return pred_df[~(doc_matches & token_matches)].reset_index(drop=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/kaggle/input/pii-detection-removal-from-educational-data/test.json",
    )
    parser.add_argument("--model_dir", type=str, default="/kaggle/input/model-dir")
    parser.add_argument("--dataset_path", type=str, default="ds.pq")
    parser.add_argument("--tokenized_ds_path", type=str, default="tds.pq")
    parser.add_argument("--preds_path", type=str, default="preds.npy")
    parser.add_argument("--add_repeated_names", action="store_true")
    parser.add_argument("--add_emails_and_urls", action="store_true")
    parser.add_argument("--output_csv_path", type=str, default="submission.csv")
    parser.add_argument("--include_token_text", action="store_true")

    args = parser.parse_args()

    ds = Dataset.from_parquet(args.dataset_path)
    tds = Dataset.from_parquet(args.tokenized_ds_path)

    predictions = softmax(np.load(args.preds_path), -1)

    config = json.load(open(Path(args.model_dir) / "config.json"))

    id2label = {int(i): l for i, l in config["id2label"].items()}
    all_labels = list(id2label.values())

    pred_df = get_preds(predictions, all_labels, id2label, tds, ds, threshold=0.8)

    data = json.load(open(args.data_path))

    if args.add_repeated_names:
        pred_df = add_repeated_names(pred_df, data)

    if args.add_emails_and_urls:
        pred_df = add_emails_and_urls(pred_df, data)

    pred_df["row_id"] = range(len(pred_df))

    cols2save = ["row_id", "document", "token", "label"]

    if args.include_token_text:
        cols2save.append("token_text")

    pred_df[cols2save].to_csv(
        args.output_csv_path, index=False
    )


if __name__ == "__main__":
    main()

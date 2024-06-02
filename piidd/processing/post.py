import re

import numpy as np
import pandas as pd


def char2token_preds(
    char_preds,
    token_map,
    id2label,
    threshold=0.9,
    pooling_function="max",
    return_all_token_scores=False,
):
    """
    This will return predictions at the token level for a single document.

    Args:
        char_preds: np.array, shape (number of characters, num_labels)
        token_map: np.array, shape (number of characters,)
            This is a mapping from character to token. If token_map[i] == X, then the i-th character belongs to the X-th token.
            -1 indicates whitespace
        id2label: dict, mapping from label id to label
        threshold: float, threshold for predictions
            "O" predictions must be above this otherwise the second highest score will be used.
        pooling_function: str, "max" or "mean"
            How to pool the character predictions to get a token prediction.
        return_all_token_scores: bool, whether to return all scores for each token


    Returns:
        list of tuples: (token_id, label)
    """

    def pool(x):
        if pooling_function == "max":
            return x.max(0)
        elif pooling_function == "mean":
            return x.mean(0)
        else:
            raise ValueError(f"Pooling function {pooling_function} not supported")

    # pair of (token_id, scores)
    token_preds = []

    tm = np.array(token_map)

    token_idxs = sorted(list(set(tm) - {-1}))

    for i in token_idxs:
        pooled_scores = pool(char_preds[tm == i])

        if return_all_token_scores:
            token_preds.append((i, pooled_scores))
        else:
            # can do thresholding here

            top_ids = pooled_scores.argsort()[::-1]  # reverse to go from high to low

            if pooled_scores.sum() < 1e-4:
                continue

            if id2label[top_ids[0]] == "O":
                if pooled_scores[top_ids[0]] < threshold:
                    token_preds.append((i, id2label[top_ids[1]]))
            else:
                token_preds.append((i, id2label[top_ids[0]]))

    return token_preds


def make_pred_df(
    all_preds, doc_idxs, all_tokens, return_all_token_scores=False, id2label=None
):
    """
    all_preds is a list of lists of tuples

    """

    all_token_ids, all_labels, all_doc_idxs, token_texts = [], [], [], []

    if return_all_token_scores:
        all_scores = []

    for preds, doc_idx, tokens in zip(all_preds, doc_idxs, all_tokens):
        for token_id, label_preds in preds:
            if tokens[token_id].isspace() and "\n" not in tokens[token_id]:
                continue

            if return_all_token_scores:
                all_scores.append(label_preds)
            else:
                all_labels.append(label_preds)

            all_token_ids.append(token_id)
            all_doc_idxs.append(doc_idx)
            token_texts.append(tokens[token_id])

    df = pd.DataFrame(
        {
            "token": all_token_ids,
            "document": all_doc_idxs,
            "token_text": token_texts,
        }
    )

    if return_all_token_scores:
        label_cols = [id2label[i] for i in range(len(id2label))]
        df[label_cols] = all_scores
    else:
        df["label"] = all_labels

    df["row_id"] = range(len(df))

    return df


def get_all_preds(
    predictions,
    raw_ds,
    tokenized_ds,
    id2label,
    threshold=0.9,
    return_char_preds=False,
    return_all_token_scores=False,
):

    # each idx in raw_ds corresponds to multiple idx in tokenized_ds (due to stride)
    idx2tidxs = {}
    for i, x in enumerate(tokenized_ds["idx"]):
        if x not in idx2tidxs:
            idx2tidxs[x] = []
        idx2tidxs[x].append(i)

    all_preds = []
    all_char_preds = []

    for idx in raw_ds["idx"]:

        # if idx == 196:
        #     print()

        temp_preds = [predictions[x] for x in idx2tidxs[idx]]
        temp_tds = tokenized_ds.select(idx2tidxs[idx])

        token_map = raw_ds[idx]["token_map"]

        char_preds = np.zeros((len(token_map), len(id2label)))

        for preds, offsets in zip(temp_preds, temp_tds["offset_mapping"]):
            for (start_idx, end_idx), p in zip(offsets, preds):

                # bos, eos, cls, sep tokens
                if start_idx + end_idx == 0:
                    continue

                char_preds[start_idx:end_idx] = np.maximum(
                    char_preds[start_idx:end_idx], p
                )

        token_preds = char2token_preds(
            char_preds,
            token_map,
            id2label,
            threshold=threshold,
            return_all_token_scores=return_all_token_scores,
        )

        all_preds.append(token_preds)

        if return_char_preds:
            all_char_preds.append(char_preds.astype(np.float16))

    pred_df = make_pred_df(
        all_preds,
        raw_ds["document"],
        raw_ds["tokens"],
        return_all_token_scores,
        id2label,
    )

    if return_char_preds:
        return pred_df, all_char_preds
    return pred_df


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
                "token_text": token_texts,
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

    pred_df["bad_name_casing"] = [
        "NAME_STUDENT" in l and not t.istitle()
        for t, l in zip(pred_df["token_text"], pred_df["label"])
    ]

    return pred_df[~pred_df["bad_name_casing"]].reset_index(drop=True)


def remove_name_titles(pred_df):
    """
    Remove predictions like Mr, Mrs, Dr, etc.
    """

    for idx, (token_text, token_idx, pred_label) in enumerate(
        pred_df[["token_text", "token", "label"]].values
    ):
        if pred_label in {"B-NAME_STUDENT", "I-NAME_STUDENT"}:
            if token_text.lower().rstrip(".") in {"mr", "mrs", "dr", "ms", "miss"}:
                pred_df.at[idx, "label"] = "O"

                if idx > 0 and idx + 1 < len(pred_df):
                    if pred_df["token"][idx + 1] == token_idx + 1:
                        if pred_df["label"][idx + 1] == "I-NAME_STUDENT":
                            pred_df.at[idx + 1, "label"] = "B-NAME_STUDENT"

    return pred_df[pred_df["label"] != "O"].reset_index(drop=True)


def correct_name_student_preds(pred_df):
    """
    If a prediction is made with "I-NAME_STUDENT" without another "I-NAME_STUDENT" or "B-NAME_STUDENT" next to it, then it gets fixed to "B-NAME_STUDENT"

    This works well when a "B-NAME_STUDENT" is mistakenly predicted as "I-NAME_STUDENT"
    """

    for d in pred_df["document"].unique():

        filtered = pred_df[pred_df["document"] == d]

        if len(filtered) > 0:
            filtered = filtered.sort_values("token", ascending=True)

            for idx in range(len(filtered)):

                pred_label = filtered["label"].values[idx]
                token_idx = filtered["token"].values[idx]
                df_idx = filtered.index.values[idx]

                # if current pred is "B-NAME_STUDENT"
                # and previous token was also "B-NAME_STUDENT"
                # then change this one to "I-NAME_STUDENT"
                if pred_label == "B-NAME_STUDENT":

                    if idx > 0 and filtered["token"].values[idx - 1] == token_idx - 1:
                        if filtered["label"].values[idx - 1] == "B-NAME_STUDENT":
                            filtered.at[df_idx, "label"] = "I-NAME_STUDENT"
                            pred_df.at[df_idx, "label"] = "I-NAME_STUDENT"

                elif pred_label == "I-NAME_STUDENT":
                    if idx == 0:
                        filtered.at[df_idx, "label"] = "B-NAME_STUDENT"
                        pred_df.at[df_idx, "label"] = "B-NAME_STUDENT"
                    else:
                        # if the previous token is not next to this token (by token id)
                        # then this should be "B"
                        if filtered["token"].values[idx - 1] != token_idx - 1:
                            filtered.at[df_idx, "label"] = "B-NAME_STUDENT"
                            pred_df.at[df_idx, "label"] = "B-NAME_STUDENT"

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

    pred_df = pred_df.reset_index(drop=True)

    for i, (label, token_text) in enumerate(pred_df[["label", "token_text"]].values):
        if label == "B-URL_PERSONAL":
            if "coursera.org" in token_text or "wikipedia.org" in token_text:
                bad_urls.append(i)
            elif ".gov" in token_text or ".edu" in token_text:
                bad_urls.append(i)
            elif not any(
                ["http" in token_text, "://" in token_text, "www." in token_text]
            ):
                bad_urls.append(i)
        elif label == "I-URL_PERSONAL":
            bad_urls.append(i)

    return pred_df[~pred_df.index.isin(bad_urls)].reset_index(drop=True)


def remove_bad_categories(pred_df):
    """
    As of March 14, these categories were always FP.
    """
    bad_categories = set(
        [
            "I-USERNAME",
            "I-EMAIL",
            "I-URL_PERSONAL",
            #         "I-ID_NUM",
        ]
    )

    return pred_df[~pred_df.label.isin(bad_categories)].reset_index(drop=True)


def check_phone_numbers(pred_df):
    """
    Rules:
    - If a token that has been predicted as "B-PHONE_NUM" has a number that
    1) is longer than 4 digits
    2) doesn't have any of {"(", ")", ".", "x", "-"}
    then it should be "B-ID_NUM"
    """

    for i, (doc, label, token_text) in enumerate(
        pred_df[["document", "label", "token_text"]].values
    ):
        if label == "B-PHONE_NUM":
            if len(token_text) > 4:
                if not any([x in token_text for x in {"(", ")", ".", "x", "-", "+"}]):
                    pred_df.at[i, "label"] = "B-ID_NUM"

    return pred_df


def manual_check(
    pred_df,
    thresholds={
        "NAME_STUDENT": 0.15,
        "USERNAME": 0.5,
        "EMAIL": 0.7,
        "URL_PERSONAL": 0.7,
        "PHONE_NUM": 0.5,
        "STREET_ADDRESS": 0.5,
        "ID_NUM": 0.5,
    },
):
    """
    Assumes that the labels do not have B or I
    """

    not_title = ~pred_df["token_text"].str.istitle()
    has_other_chars = pred_df["token_text"].str.contains("[^a-zA-Z\.]")

    pred_df.loc[(not_title | has_other_chars), "NAME_STUDENT"] = 0

    pred_df.loc[~pred_df["token_text"].str.islower(), "USERNAME"] = 0
    pred_df.loc[pred_df["token_text"].str.isspace(), ["USERNAME", "ID_NUM"]] = 0

    pred_df.loc[
        pred_df["token_text"].str.contains("(http)|(www)|(://)"),
        ["NAME_STUDENT", "USERNAME", "ID_NUM", "EMAIL", "PHONE_NUM", "STREET_ADDRESS"],
    ] = 0

    pred_df.loc[
        pred_df["token_text"].isin({"\n", "\n\n"}),
        ["NAME_STUDENT", "USERNAME", "ID_NUM", "EMAIL", "PHONE_NUM", "URL_PERSONAL"],
    ] = 0

    leq_1 = pred_df["token_text"].str.len() <= 1

    pred_df.loc[leq_1, ["NAME_STUDENT", "USERNAME", "EMAIL", "URL_PERSONAL"]] = 0

    leq_4 = pred_df["token_text"].str.len() <= 4

    pred_df.loc[leq_4, ["USERNAME", "EMAIL", "URL_PERSONAL"]] = 0

    pred_df.loc[~pred_df["token_text"].str.contains("@"), "EMAIL"] = 0

    all_preds = []

    for d in pred_df["document"].unique():
        filtered = pred_df[pred_df["document"] == d]

        # NAMES
        temp_name_df = filtered[filtered["NAME_STUDENT"] > thresholds["NAME_STUDENT"]]

        temp_preds = []
        for idx, token_text in temp_name_df[["token", "token_text"]].values:
            if token_text.rstrip(".") in {"Mr", "Mrs", "Dr", "Ms", "Miss", "Prof"}:
                continue

            if len(temp_preds) == 0:
                temp_preds.append((d, idx, "B-NAME_STUDENT", token_text))
            else:
                if temp_preds[-1][1] + 1 == idx:
                    temp_preds.append((d, idx, "I-NAME_STUDENT", token_text))
                else:
                    temp_preds.append((d, idx, "B-NAME_STUDENT", token_text))

        # maybe add repeat names

        # URLS

        temp_url_df = filtered[
            (filtered["URL_PERSONAL"] > thresholds["URL_PERSONAL"])
            & (filtered["token_text"].str.len() > 5)
        ]
        temp_preds.extend(
            [
                (d, idx, "B-URL_PERSONAL", token_text)
                for idx, token_text in temp_url_df[["token", "token_text"]].values
            ]
        )

        # EMAILS

        temp_email_df = filtered[
            (filtered["EMAIL"] > thresholds["EMAIL"])
            & (~filtered["token_text"].str.isspace())
        ]
        temp_preds.extend(
            [
                (d, idx, "B-EMAIL", token_text)
                for idx, token_text in temp_email_df[["token", "token_text"]].values
            ]
        )

        # STREET_ADDRESS

        temp_street_df = filtered[
            filtered["STREET_ADDRESS"] > thresholds["STREET_ADDRESS"]
        ]

        for idx, token_text in temp_street_df[["token", "token_text"]].values:

            if len(temp_preds) == 0:
                temp_preds.append((d, idx, "B-STREET_ADDRESS", token_text))
            else:
                if temp_preds[-1][1] + 1 == idx and temp_preds[-1][2].endswith(
                    "STREET_ADDRESS"
                ):
                    temp_preds.append((d, idx, "I-STREET_ADDRESS", token_text))
                else:
                    temp_preds.append((d, idx, "B-STREET_ADDRESS", token_text))

        # USERNAME

        temp_username_df = filtered[filtered["USERNAME"] > thresholds["USERNAME"]]
        temp_preds.extend(
            [
                (d, idx, "B-USERNAME", token_text)
                for idx, token_text in temp_username_df[["token", "token_text"]].values
            ]
        )

        # PHONE_NUM

        temp_phone_df = filtered[filtered["PHONE_NUM"] > thresholds["PHONE_NUM"]]

        for idx, token_text in temp_phone_df[["token", "token_text"]].values:
            if len(temp_preds) == 0:
                temp_preds.append((d, idx, "B-PHONE_NUM", token_text))
            else:
                if temp_preds[-1][1] + 1 == idx and temp_preds[-1][2].endswith(
                    "PHONE_NUM"
                ):
                    temp_preds.append((d, idx, "I-PHONE_NUM", token_text))
                else:
                    temp_preds.append((d, idx, "B-PHONE_NUM", token_text))

        # ID_NUM

        temp_id_df = filtered[filtered["ID_NUM"] > thresholds["ID_NUM"]]

        for idx, token_text in temp_id_df[["token", "token_text"]].values:
            if len(temp_preds) == 0:
                temp_preds.append((d, idx, "B-ID_NUM", token_text))
            else:
                if temp_preds[-1][1] + 1 == idx and temp_preds[-1][2].endswith(
                    "ID_NUM"
                ):
                    temp_preds.append((d, idx, "I-ID_NUM", token_text))
                else:
                    temp_preds.append((d, idx, "B-ID_NUM", token_text))

        all_preds.extend(temp_preds)

    return pd.DataFrame(all_preds, columns=["document", "token", "label", "token_text"])


def manual_v2(pred_df):

    not_title = ~pred_df["token_text"].str.istitle()
    has_other_chars = pred_df["token_text"].str.contains("[^a-zA-Z\.]")

    pred_df.loc[(not_title | has_other_chars), ["B-NAME_STUDENT", "I-NAME_STUDENT"]] = 0

    pred_df.loc[~pred_df["token_text"].str.islower(), "B-USERNAME"] = 0
    pred_df.loc[pred_df["token_text"].str.isspace(), ["B-USERNAME", "B-ID_NUM"]] = 0

    pred_df.loc[
        pred_df["token_text"].str.contains("(http)|(www)|(://)"),
        [
            f"B-{x}"
            for x in [
                "NAME_STUDENT",
                "USERNAME",
                "ID_NUM",
                "EMAIL",
                "PHONE_NUM",
                "STREET_ADDRESS",
            ]
        ]
        + [f"I-{x}" for x in ["NAME_STUDENT", "ID_NUM", "PHONE_NUM", "STREET_ADDRESS"]],
    ] = 0

    pred_df.loc[
        pred_df["token_text"].isin({"\n", "\n\n"}),
        [
            f"B-{x}"
            for x in [
                "NAME_STUDENT",
                "USERNAME",
                "ID_NUM",
                "EMAIL",
                "PHONE_NUM",
                "URL_PERSONAL",
            ]
        ]
        + [
            f"I-{x}"
            for x in [
                "NAME_STUDENT",
                "ID_NUM",
                "PHONE_NUM",
            ]
        ],
    ] = 0

    leq_1 = pred_df["token_text"].str.len() <= 1

    pred_df.loc[
        leq_1,
        [f"B-{x}" for x in ["NAME_STUDENT", "USERNAME", "EMAIL", "URL_PERSONAL"]]
        + ["I-NAME_STUDENT"],
    ] = 0

    leq_4 = pred_df["token_text"].str.len() <= 3

    pred_df.loc[leq_4, [f"B-{x}" for x in ["USERNAME", "EMAIL", "URL_PERSONAL"]]] = 0

    pred_df.loc[~pred_df["token_text"].str.contains("@"), "B-EMAIL"] = 0

    cols = [
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

    pred_df = pred_df.sort_values(["document", "token"])

    high_scores = pred_df[cols].max(axis=1) > 0.15
    low_o = pred_df["O"] < 0.9

    pred_df["label"] = pred_df[cols+["O"]].idxmax(axis=1)
    pred_df.loc[high_scores&low_o, "label"] = pred_df.loc[high_scores&low_o, cols].idxmax(axis=1)


    url_likely = pred_df["token_text"].str.contains("(http)|(www)|(://)")
    high_url_score = pred_df["B-URL_PERSONAL"] > 0.15

    pred_df.loc[
        url_likely&high_url_score,
        "label",
    ] = "B-URL_PERSONAL"

    pred_df.loc[
        pred_df[cols].sum(axis=1) < 1e-3,
        "label"
    ] = "O"


    return pred_df
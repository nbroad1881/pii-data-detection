# Creating features for second stage model

from itertools import chain

def make_str_features(tokens, idx):

    this_token = tokens[idx]
    prev_token = tokens[idx - 1] if idx > 0 else ""
    prev2_token = tokens[idx - 2] if idx > 1 else ""
    prev3_token = tokens[idx - 3] if idx > 2 else ""
    prev4_token = tokens[idx - 4] if idx > 3 else ""
    next_token = tokens[idx + 1] if idx < len(tokens) - 1 else ""
    next2_token = tokens[idx + 2] if idx < len(tokens) - 2 else ""
    next3_token = tokens[idx + 3] if idx < len(tokens) - 3 else ""
    next4_token = tokens[idx + 4] if idx < len(tokens) - 4 else ""

    # need feature for how many times that token appears

    def make_token_features(token, idx):
        return [
            idx/len(tokens),
            len(token),
            token.isupper(),
            token.islower(),
            token.istitle(),
            token.isdigit(),
            token.isalpha(),
            len([x for x in token if x.isalpha()]),
            len([x for x in token if x.isspace()]),
            len([x for x in token if x.isdigit()]),
            token.count("-"),
            token.count("/"),
            token.count("."),
            token.count(","),
            1 if "http" in token or "www" in token else 0,
            1 if "@" in token else 0,
            1 if ":" in token else 0,
            1 if "name" in token.lower() else 0,
            1 if "address" in token.lower() else 0,
            1 if "phone" in token.lower() else 0,
            1 if "email" in token.lower() else 0,
            1 if "addr" in token.lower() else 0,
            1 if "number" in token.lower() else 0,
            1 if "num" in token.lower() else 0,
            1 if "(" in token else 0,
            1 if ")" in token else 0,
            1 if "\n" in token else 0,
            1 if "+" in token else 0,
            1 if "|" in token else 0,
        ]

    features = [
        make_token_features(this_token, idx),
        make_token_features(prev_token, idx - 1),
        make_token_features(prev2_token, idx - 2),
        make_token_features(prev3_token, idx - 3),
        make_token_features(prev4_token, idx - 4),
        make_token_features(next_token, idx + 1),
        make_token_features(next2_token, idx + 2),
        make_token_features(next3_token, idx + 3),
        make_token_features(next4_token, idx + 4),
    ]

    return list(chain(*features))

"""
Thinking through process.

1. Model makes predictions for each token --> dataframe
2. Loop through json data in dataset to make string features
3. Ignore any tokens that have label scores < 0.01 for all main labels
   - Don't create a feature for those tokens, but use those scores when doing next/prev scores


Group dataframe by document id
- turn into dataset
- can do multiprocessing if one sample is one document

"""


def make_pred_features(preds, idx, num_before=2, num_after=2):
    """
    preds is list of list of floats.
    len(preds) is the number of tokens in the document.
    len(preds[i]) is the number of labels in the model.

    Returns a list of floats that are the predictions for the token at index idx as well as the tokens before and after.

    Returns:
        list of float
    """

    num_labels = len(preds[0])
    all_preds = preds[idx].copy()

    for i in range(1, num_before + 1):
        all_preds.extend(preds[idx - i] if (idx - i) >= 0 else [0.0] * num_labels)
    
    for i in range(1, num_after + 1):
        all_preds.extend(preds[idx + i] if (idx + i) < len(preds) else [0.0] * num_labels)

    return all_preds


def token_featurize_doc(example):
    """
    To be used in map function.
    """

    tokens = example["tokens"]
    idxs = example["idxs"]

    token_features = [make_str_features(tokens, idx) for idx in idxs]

    return {
        "token_features": token_features,
    }


def pred_featurize_doc(example, num_before=2, num_after=2):
    """
    """

    pred_cols = sorted([x for x in example.keys() if "B-" in x or "I-" in x or x == "O"])

    preds = [example[col] for col in pred_cols]
    preds = [list(group) for group in zip(*preds)]

    idxs = example["idxs"]

    return {
        "pred_features": [make_pred_features(preds, idx, num_before=num_before, num_after=num_after) for idx in idxs],
    }


def add_idxs(example):
    pred_cols = sorted([x for x in example.keys() if "B-" in x or "I-" in x or x == "O"])

    preds = [example[col] for col in pred_cols]
    preds = [list(group) for group in zip(*preds)]

    idxs = [i for i, p in enumerate(preds) if any(y > 0.001 for y in p[:-1])]

    return {
        "idxs": idxs,
    }

def get_labels(example):

    idxs = example["idxs"]

    return {
        "labels": [example["label"][idx] for idx in idxs]
    }
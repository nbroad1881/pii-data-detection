# Creating features for second stage model

from itertools import chain

def make_str_features(full_text, tokens, idx):

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

    def make_token_features(token):
        return [
            len(token),
            token.isupper(),
            token.islower(),
            token.istitle(),
            token.isdigit(),
            token.isalpha(),
            len([x for x in token if x.isdigit()]),
            1 if "-" in token else 0,
            1 if "/" in token else 0,
            1 if "." in token else 0,
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
            1 if "instructor" in token.lower() else 0,
            1 if "(" in token else 0,
            1 if ")" in token else 0,
            1 if "\n" in token else 0,
            1 if "+" in token else 0,
            1 if "|" in token else 0,
        ]

    features = [
        make_token_features(t)
        for t in [
            this_token,
            prev_token,
            prev2_token,
            prev3_token,
            prev4_token,
            next_token,
            next2_token,
            next3_token,
            next4_token,
        ]
    ]

    features = list(chain(*features))

def make_pred_features(preds, idx, num_before=2, num_after=2):
    """
    preds is a list of list of lists
    preds[0] is the first token predictions
    preds[0][0] is the first model's predictions
    preds[0][0][0] is the first model's prediction for the first label

    Returns:
        list of float
    """

    preds = [preds[idx]]

    for i in range(1, num_before + 1):
        preds.append(preds[idx - i] if (idx - i) >= 0 else [0.0] * len(preds[0][0]))
    
    for i in range(1, num_after + 1):
        preds.append(preds[idx + i] if (idx + i) < len(preds) else [0.0] * len(preds[0][0]))

    


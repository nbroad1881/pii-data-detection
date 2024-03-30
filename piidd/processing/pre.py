import numpy as np

def strided_tokenize(example, tokenizer, label2id, max_length, stride):
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
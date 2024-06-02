import numpy as np
from datasets import Dataset


def strided_tokenize(example, tokenizer, label2id, max_length, stride):
    text = []
    char_labels = []

    tokens = example["tokens"][0]

    add_labels = "provided_labels" in example

    if add_labels:
        provided_labels = example["provided_labels"][0]
    trailing_whitespace = example["trailing_whitespace"][0]

    zipped = [tokens, trailing_whitespace]
    if add_labels:
        zipped.append(provided_labels)
    else:
        zipped.append([None] * len(tokens))

    for t, ws, label in zip(*zipped):
        text.append(t)
        if add_labels:
            char_labels.extend([label] * len(t))
        if ws:
            text.append(" ")
            if add_labels:
                char_labels.append("O")

    tokenized = tokenizer(
        "".join(text),
        return_offsets_mapping=True,
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        padding=False,
    )

    # When removing B, I from labels, add to temp label2id so we can correctly map the labels
    temp_label2id = label2id.copy()
    if not any([x.startswith("B-") for x in label2id.keys()]):
        temp_label2id.update({f"B-{k}": v for k, v in label2id.items()})
        temp_label2id.update({f"I-{k}": v for k, v in label2id.items()})

    # tokenized is now a list of lists depending on how long the input is, the max length, and the stride

    labels = {}
    if add_labels:
        char_labels = np.array(char_labels)

        text = "".join(text)

        token_labels = []
        for i in range(len(tokenized.input_ids)):

            sample_labels = [temp_label2id["O"]] * len(tokenized.input_ids[i])

            for j, (start_idx, end_idx) in enumerate(tokenized.offset_mapping[i]):

                # CLS  and PAD tokens
                if start_idx == 0 and end_idx == 0:
                    sample_labels[j] = -100

                # case when token starts with space
                while (
                    start_idx < end_idx
                    and start_idx < len(text)
                    and text[start_idx] == " "
                ):
                    start_idx += 1

                # the whole token might be whitespace
                if start_idx >= end_idx:
                    continue

                start_idx = min(start_idx, len(char_labels) - 1)

                sample_labels[j] = temp_label2id[char_labels[start_idx]]

            token_labels.append(sample_labels)

        labels = {"labels": token_labels}

    idxs = {}
    if "idx" in example:
        # only applies to validation data
        idxs = {"idx": [example["idx"][0]] * len(tokenized.input_ids)}

    return {**tokenized, **labels, **idxs}


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


def create_dataset_for_span_classification(
    data, df, start_token, end_token, num_before=100, num_after=100
):
    """
    df: pd.DataFrame
        columns: document, token, token_text, label
    """

    texts = []
    labels = []
    docs = []
    tokens = []

    has_labels = "label" in df.columns

    for d in data:
        doc = d["document"]

        filtered = df[df["document"] == doc]

        if len(filtered) == 0:
            continue

        cols = ["token", "label"] if has_labels else ["token"]

        for row in filtered[cols].values:

            if has_labels:
                idx, label = row
            else:
                idx = row[0]

            text = create_snippet(
                idx,
                d["tokens"],
                d["trailing_whitespace"],
                num_tokens_before=num_before,
                num_tokens_after=num_after,
                start_delimiter=start_token,
                end_delimiter=end_token,
            )

            texts.append(text)
            docs.append(doc)
            tokens.append(idx)

            if has_labels:
                labels.append(label)

    d = {"text": texts, "doc": docs, "token": tokens}
    if has_labels:
        d["label"] = labels  
    return Dataset.from_dict(d)


def create_snippet(
    idx,
    tokens,
    whitespace,
    num_tokens_before=400,
    num_tokens_after=400,
    start_delimiter="",
    end_delimiter="",
):

    max_idx = len(tokens)

    before_tokens = tokens[max(0, idx - num_tokens_before) : idx]

    before_text = ""
    for b, bws in zip(before_tokens, whitespace[max(0, idx - num_tokens_before) : idx]):
        before_text += b
        if bws:
            before_text += " "

    after_text = ""
    if idx + 1 >= max_idx:
        after = []
    else:
        after = tokens[idx + 1 : idx + num_tokens_after]

        for a, aws in zip(after, whitespace[idx + 1 : idx + num_tokens_after]):
            after_text += a
            if aws:
                after_text += " "

    text = before_text + start_delimiter + tokens[idx] + end_delimiter

    if whitespace[idx]:
        text += " "

    text += after_text

    return text


def tokenize_for_span_classification(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["text"],
        padding=False,
        truncation=False,
    )

    if "labels" in examples:
        tokenized["labels"] = [label2id[ll] for ll in examples["label"]]

    return tokenized

import re
from datetime import datetime
from itertools import chain
from dataclasses import dataclass, field

from datasets import Dataset
from transformers import TrainingArguments, HfArgumentParser, set_seed, AutoTokenizer, AutoModelForTokenClassification, Trainer


def tokenize(example, tokenizer, label2id):
    names_patterns = [
        "|".join([r"\b" + x + r"\b" for x in n]) for n in example["names"]
    ]

    all_matches = [
        list(re.finditer(p, t)) for p, t in zip(names_patterns, example["text"])
    ]

    tokenized = tokenizer(
        example["text"], padding=False, truncation=False, return_offsets_mapping=True
    )

    all_labels = []

    for offsets, matches in zip(tokenized.offset_mapping, all_matches):

        labels = [label2id["O"]] * len(offsets)

        if len(matches) == 0:
            all_labels.append(labels)
            continue

        for label_idx, (start, end) in enumerate(offsets):
            for m in matches:
                if (
                    start <= m.start() < end
                    or start < m.end() <= end
                    or (m.start() < start and m.end() > end)
                ):
                    labels[label_idx] = label2id["NAME"]
                    break

        all_labels.append(labels)

    tokenized["labels"] = all_labels

    return tokenized


def group_texts(examples, max_length):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result


@dataclass
class Config(TrainingArguments):

    model_path: str = field(
        default="roberta-base",
    )
    max_length: int = field(default=384)
    num_proc: int = field(default=8)

    train_on_all_data: bool = field(default=False)

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

    dataset_path: str = field(default="data/names.parquet")


def compute_metrics(eval_predictions):

    preds, label_ids = eval_predictions

    preds = preds.argmax(axis=-1)

    return {"accuracy": (preds == label_ids).mean()}

def main():

    parser = HfArgumentParser((Config,))

    args = parser.parse_args_into_dataclasses()[0]

    if args.random_seed:
        args.seed = datetime.now().microsecond
        set_seed(args.seed)

    ds = Dataset.from_parquet(args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    label2id = {"O": 0, "NAME": 1}
    id2label = {v: k for k, v in label2id.items()}

    tokenized_ds = ds.map(
        lambda x: tokenize(x, tokenizer, label2id), batched=True, num_proc=args.num_proc, remove_columns=ds.column_names
    ).map(lambda x: group_texts(x, max_length=args.max_length), batched=True, num_proc=args.num_proc)

    print(tokenized_ds)

    split = tokenized_ds.train_test_split(test_size=0.15)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()

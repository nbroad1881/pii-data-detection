import json
import random
from pathlib import Path

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback

from piidd.models.layerdrop_deberta import MultiSampleDebertaV2ForTokenClassification

this_dir = Path(__file__).resolve().parent

FIRST_NAMES_FILEPATH = this_dir.parent.parent / "data/first_names.json"
LAST_NAMES_FILEPATH = this_dir.parent.parent / "data/surnames.json"


def swap_names(ds: Dataset, num_proc: int = 8):
    first_names = json.load(open(FIRST_NAMES_FILEPATH))
    last_names = json.load(open(LAST_NAMES_FILEPATH))

    def swap(example):
        mapping = {}

        new_tokens = []

        for token, label in zip(example["tokens"], example["provided_labels"]):
            if "NAME" in label:
                if label[0] == "B":
                    if token not in mapping:
                        mapping[token] = random.choice(first_names).title()
                elif label[0] == "I":
                    if token not in mapping:
                        mapping[token] = random.choice(last_names).title()

            new_tokens.append(mapping.get(token, token))

        return {"tokens": new_tokens}

    return ds.map(swap, num_proc=num_proc, keep_in_memory=True)


class PiiTrainer(Trainer):

    def __init__(self, raw_train_ds, tokenize_func, **kwargs):
        self.raw_train_ds = raw_train_ds
        self.tokenize_func = tokenize_func

    def get_train_dataloader(self) -> DataLoader:
        return super().get_train_dataloader()


class SwapNameCallback(TrainerCallback):

    def __init__(self, raw_train_ds, tokenize_func, trainer):
        self.raw_train_ds = raw_train_ds
        self.tokenize_func = tokenize_func
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):

        if state.epoch == 0:
            return

        keep_cols = ["input_ids", "attention_mask", "labels"]
        remove_cols = [x for x in self.raw_train_ds.column_names if x not in keep_cols]

        ds = swap_names(self.raw_train_ds)
        ds = ds.map(
            self.tokenize_func,
            num_proc=8,
            batched=True,
            batch_size=1,
            remove_columns=remove_cols,
        )

        idx = random.choice(range(len(ds)))
        x = ds[idx]

        b_name_student = self.trainer.model.config.label2id["B-NAME_STUDENT"]

        while b_name_student not in x["labels"]:
            idx = random.choice(range(len(ds)))
            x = ds[idx]

        tokens = self.trainer.tokenizer.convert_ids_to_tokens(x["input_ids"])
        print(list(zip(tokens, x["labels"])))

        self.trainer.train_dataset = ds

        kwargs["train_dataloader"] = self.trainer.get_train_dataloader()


def create_peft_model(
    model_path,
    num_labels,
    id2label,
    label2id,
    model_dtype,
    gradient_checkpointing=True,
    lora_r=16,
    lora_modules="all-linear",
    lora_alpha=32,
    lora_dropout=0.1,
    use_multisample_dropout=False,
):

    import torch
    from transformers import BitsAndBytesConfig, AutoModelForTokenClassification
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_kwargs = {}
    if model_dtype in {"int8", "int4"}:

        if model_dtype == "int8":
            raise NotImplementedError("int8 not supported yet")
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    if model_dtype == "fp32":
        model_kwargs["torch_dtype"] = torch.float32
    elif model_dtype == "fp16":
        model_kwargs["torch_dtype"] = torch.float16


    if use_multisample_dropout:
        model = MultiSampleDebertaV2ForTokenClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            **model_kwargs,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            **model_kwargs,
        )

    if model_dtype in {"int8", "int4"}:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="TOKEN_CLS",
        target_modules=lora_modules,
        bias="none",
    )

    return get_peft_model(model, config)


def filter_dataset_for_labels(example, labels2keep):
    """
    Focus only on certain labels. All other labels will be set to "O"
    """

    new_labels = []

    for label in example["provided_labels"]:
        if label in labels2keep:
            new_labels.append(label)
        else:
            new_labels.append("O")

    return {
        "provided_labels": new_labels,
    }


# https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/468844
def filter_no_pii(example, percent_allow=0.2):
    # Return True if there is PII
    # Or 20% of the time if there isn't

    has_pii = set("O") != set(example["provided_labels"])

    return has_pii or (random.random() < percent_allow)


def chunk_for_spanmarker(ds, max_length, num_proc=8):


    def chunk(example):

        new_tokens = []
        new_labels = []


        batch_tokens = example["tokens"]
        batch_labels = example["provided_labels"]
        batch_whitespace = example["trailing_whitespace"]

        for tokens, labels, ws in zip(batch_tokens, batch_labels, batch_whitespace):
            # tokens = [t+" " if has_trailing_whitespace else t for t, has_trailing_whitespace in zip(tokens, ws)]

            for i in range(0, len(tokens), max_length):
                temp = tokens[i : i + max_length]

                # if last chunk is smaller than max length,
                # take max length from the end
                if len(temp) < max_length:
                    new_tokens.append(tokens[-max_length:])
                    new_labels.append(labels[-max_length:])
                else:
                    new_tokens.append(tokens[i : i + max_length])
                    new_labels.append(labels[i : i + max_length])

        return {
            "tokens": new_tokens,
            "provided_labels": new_labels,
        }

    return ds.map(chunk, batched=True, num_proc=num_proc, remove_columns=ds.column_names)
import os
import json

from datetime import datetime
from functools import partial
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    set_seed,
)
from tokenizers import AddedToken
from datasets import Dataset, concatenate_datasets

from piidd.training.utils import (
    SwapNameCallback,
    create_peft_model,
    filter_no_pii,
    compute_metrics,
    make_gt_dataframe,
)
from piidd.models.layerdrop_deberta import (
    add_layer_drop,
    MultiSampleDebertaV2ForTokenClassification,
)
from piidd.processing.pre import strided_tokenize, add_token_map

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Config(TrainingArguments):

    model_path: str = field(
        default="roberta-base",
    )
    max_length: int = field(default=384)
    stride: int = field(default=64)
    num_proc: int = field(default=8)
    lr: float = field(default=1e-5)

    train_on_all_data: bool = field(default=False)

    filter_no_pii_percent_allow: float = field(default=0.2)

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

    main_dataset_path: str = field(default="train.json")
    extra_dataset_path: str = field(default=None)
    remove_classes: str = field(default="I-URL_PERSONAL,I-EMAIL,I-USERNAME")


def main():
    parser = HfArgumentParser((Config,))

    args = parser.parse_args_into_dataclasses()[0]

    if args.random_seed:
        args.seed = datetime.now().microsecond
        set_seed(args.seed)

    if "," in args.lora_modules:
        args.lora_modules = args.lora_modules.split(",")

    args.run_name = None

    data = json.load(open(args.main_dataset_path))

    base_labels = {
        "EMAIL",
        "ID_NUM",
        "NAME_STUDENT",
        "PHONE_NUM",
        "STREET_ADDRESS",
        "URL_PERSONAL",
        "USERNAME",
    }
    all_labels = []
    for ll in base_labels:
        all_labels.append(f"B-{ll}")
        all_labels.append(f"I-{ll}")
    all_labels.append("O")

    if args.remove_classes is not None:
        remove_classes = args.remove_classes.split(",")
        for rc in remove_classes:
            all_labels = [x for x in all_labels if rc not in x]

    all_labels = sorted(all_labels)
    label2id = {ll: i for i, ll in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}

    ds1 = Dataset.from_dict(
        {
            "full_text": [x["full_text"] for x in data],
            "document": [x["document"] for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data],
        }
    )

    ds1_train = ds1.filter(
        lambda x: int(x["document"]) % 4 != args.fold, num_proc=args.num_proc
    )
    ds1_test = ds1.filter(
        lambda x: int(x["document"]) % 4 == args.fold, num_proc=args.num_proc
    )

    if args.extra_dataset_path is not None:
        if args.extra_dataset_path.endswith(".json"):
            data2 = json.load(open(args.extra_dataset_path))
            ds2 = Dataset.from_dict(
                {
                    "full_text": ["" for x in data2],
                    "document": [x["document"] for x in data2],
                    "tokens": [x["tokens"] for x in data2],
                    "trailing_whitespace": [x["trailing_whitespace"] for x in data2],
                    "provided_labels": [x["labels"] for x in data2],
                }
            )
        elif args.extra_dataset_path.endswith(".pq"):
            ds2 = Dataset.from_parquet(args.extra_dataset_path)

    if "essay" in ds2.column_names:
        ds2 = ds2.rename_column("essay", "full_text")
    if "labels" in ds2.column_names:
        ds2 = ds2.rename_column("labels", "provided_labels")

    raw_ds = concatenate_datasets([ds1_train.remove_columns(["document"]), ds2])

    if args.train_on_all_data:
        raw_ds = concatenate_datasets([ds1_test.remove_columns(["document"]), raw_ds])

    # raw_ds = raw_ds.select(range(500))

    # ds = ds1_train

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.add_newline_token:
        # lots of newlines in the text
        # adding this should be helpful
        tokenizer.add_tokens(AddedToken("\n", normalized=False))

    raw_ds = raw_ds.filter(
        filter_no_pii,
        num_proc=args.num_proc,
        fn_kwargs={"percent_allow": args.filter_no_pii_percent_allow},
        keep_in_memory=True,
    )

    keep_cols = ["input_ids", "attention_mask", "labels"]
    remove_cols = [x for x in raw_ds.column_names if x not in keep_cols]

    ds = raw_ds.map(
        strided_tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": args.max_length,
            "stride": args.stride,
        },
        num_proc=args.num_proc,
        batched=True,
        batch_size=1,
        remove_columns=remove_cols,
        keep_in_memory=True,
    )

    ds1_test = ds1_test.filter(
        filter_no_pii,
        num_proc=args.num_proc,
        keep_in_memory=True,
    )

    ds1_test = ds1_test.add_column("idx", list(range(len(ds1_test))))
    val_ds = ds1_test.map(add_token_map, num_proc=args.num_proc, keep_in_memory=True)

    gt_df = make_gt_dataframe(val_ds)

    tokenized_val_ds = ds1_test.map(
        strided_tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": args.max_length,
            "stride": args.stride,
        },
        num_proc=args.num_proc,
        batched=True,
        batch_size=1,
        remove_columns=[x for x in ds1_test.column_names if x not in keep_cols],
        keep_in_memory=True,
    )

    if "deberta-v" in args.model_path:
        add_layer_drop(args.layer_drop_prob)

    if args.use_lora:
        model = create_peft_model(
            args.model_path,
            num_labels=len(all_labels),
            id2label=id2label,
            label2id=label2id,
            model_dtype=args.model_dtype,
            gradient_checkpointing=args.gradient_checkpointing,
            lora_r=args.lora_r,
            lora_modules=args.lora_modules,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.1,
            use_multisample_dropout=args.use_multisample_dropout,
        )
    else:
        if args.use_multisample_dropout:
            model = MultiSampleDebertaV2ForTokenClassification.from_pretrained(
                args.model_path,
                num_labels=len(all_labels),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

        else:
            model = AutoModelForTokenClassification.from_pretrained(
                args.model_path,
                num_labels=len(all_labels),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    args.fp16 = True
    args.learning_rate = args.lr
    args.weight_decay = 0.01
    args.report_to = ["wandb"]
    args.evaluation_strategy = "epoch"
    args.save_total_limit = 1
    args.logging_steps = 50
    args.metric_for_best_model = "f5_score"
    args.greater_is_better = True
    args.dataloader_num_workers = 1
    args.lr_scheduler_type = "cosine"

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=tokenized_val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(
            compute_metrics,
            id2label=id2label,
            tokenized_eval_ds=tokenized_val_ds,
            eval_data=val_ds,
            gt_df=gt_df,
            output_dir=args.output_dir,
        ),
    )

    swcb = SwapNameCallback(
        raw_ds,
        partial(
            strided_tokenize,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            stride=args.stride,
        ),
        trainer=trainer,
    )

    trainer.add_callback(swcb)

    trainer.train()


if __name__ == "__main__":
    main()

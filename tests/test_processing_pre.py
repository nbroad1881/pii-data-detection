from transformers import AutoTokenizer
from tokenizers import AddedToken
from datasets import Dataset


from piidd.processing.pre import strided_tokenize

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
tokenizer.add_special_tokens(AddedToken("\n", normalized=False))


def test_strided_tokenize():

    example = {
        "tokens": [
            [
                "Hello",
                "world",
                "this",
                "is",
                "Nicholas",
                "Broad",
                ".",
                "2837",
                "Apple",
                "Ln",
                "\n",
                "CA",
            ]
        ],
        "provided_labels": [
            [
                "O",
                "O",
                "O",
                "O",
                "B-NAME_STUDENT",
                "I-NAME_STUDENT",
                "O",
                "B-STREET_ADDRESS",
                "I-STREET_ADDRESS",
                "I-STREET_ADDRESS",
                "I-STREET_ADDRESS",
                "I-STREET_ADDRESS",
            ]
        ],
        "trailing_whitespace": [
            [True, True, True, True, True, False, True, True, True, True, True, True]
        ],
    }

    dataset = Dataset.from_dict(example)

    label2id = {
        "O": 0,
        "B-NAME_STUDENT": 1,
        "I-NAME_STUDENT": 2,
        "B-STREET_ADDRESS": 3,
        "I-STREET_ADDRESS": 4,
    }
    max_length = 10
    stride = 5

    ds2 = dataset.map(
        strided_tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": max_length,
            "stride": stride,
        },
        batched=True,
        batch_size=1,
        remove_columns=dataset.column_names,
    )

    gt_tokens = [
        [
            "[CLS]",
            "▁Hello",
            "▁world",
            "▁this",
            "▁is",
            "▁Nicholas",
            "▁Broad",
            ".",
            "▁28",
            "[SEP]",
        ],
        [
            "[CLS]",
            "▁is",
            "▁Nicholas",
            "▁Broad",
            ".",
            "▁28",
            "37",
            "▁Apple",
            "▁Ln",
            "[SEP]",
        ],
        ["[CLS]", ".", "▁28", "37", "▁Apple", "▁Ln", "\n", "▁CA", "[SEP]"],
    ]
    gt_labels = [
        [-100, 0, 0, 0, 0, 1, 2, 0, 3, -100],
        [-100, 0, 1, 2, 0, 3, 3, 4, 4, -100],
        [-100, 0, 3, 3, 4, 4, 4, 4, -100, 0],
    ]

    for i, ids in enumerate(ds2["input_ids"]):
        assert tokenizer.convert_ids_to_tokens(ids) == gt_tokens[i]

    assert ds2["labels"] == gt_labels 
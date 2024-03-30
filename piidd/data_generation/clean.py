import os
import re
from pathlib import Path

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from datasets import Dataset, concatenate_datasets

load_dotenv("../../.env")

project_dir = Path(os.getenv("PROJECT_HOME_DIR"))


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:

    files = list((project_dir / cfg.data_dir).glob(cfg.file_glob))

    ds = concatenate_datasets([Dataset.from_parquet(str(f)) for f in files])

    def clean(example):

        if cfg.functions.get("remove_note"):
            example[cfg.essay_column] = remove_note(example[cfg.essay_column])

        if cfg.functions.get("remove_word_count"):
            example[cfg.essay_column] = remove_word_count(example[cfg.essay_column])

        if cfg.functions.get("remove_here_is"):
            example[cfg.essay_column] = remove_here_is(example[cfg.essay_column])

        return {
            cfg.essay_column: example[cfg.essay_column],
        }

    ds = ds.map(clean, num_proc=cfg.num_proc)

    if cfg.filters.get("url_outside_angle_brackets"):
        ds = ds.filter(
            lambda x: not url_outside_angle_brackets(x[cfg.essay_column]),
            num_proc=cfg.num_proc,
        )

    if cfg.filters.get("no_pii"):
        ds = ds.filter(
            lambda x: not no_pii(x, cfg.names_column, cfg.famous_column, cfg.url_pattern),
            num_proc=cfg.num_proc,
        )

    if cfg.filters.get("no_pii_haiku"):
        ds = ds.filter(
            lambda x: no_pii_haiku(x),
            num_proc=cfg.num_proc,
        )

    if cfg.filters.get("no_errors"):
        ds = ds.filter(
            lambda x: no_errors(x[cfg.essay_column]),
            num_proc=cfg.num_proc,
        )

    full_output_path = project_dir / cfg.output_path

    print(ds)
    print(f"Dataset saved to {str(full_output_path)}")

    ds.to_parquet(str(full_output_path))


def remove_note(x):

    patterns = [
        "\nNote:",
        "\n(Note:",
        "\n[Note:",
        "\nPlease note",
    ]

    for p in patterns:
        if p in x and x.rindex(p) > 0.8 * len(x):
            x = x[: x.index(p)]

    return x


def remove_word_count(x):

    patterns = ["\nWord count:", "\nWord Count:", "\n(Word Count:", "\nEssay Length:"]
    for p in patterns:
        if p in x and x.rindex(p) > 0.8 * len(x):
            x = x[: x.index(p)]

    # (1092 words)

    return x


def url_outside_angle_brackets(x):

    temp = re.sub(r"<<.*>>", "<<POTATO>>", x)

    return "URL" in temp


def no_pii(example, names_col, famous_col, url_pattern):
    if len(example[names_col]) > 0:
        return False

    if len(example[famous_col]) > 0:
        return False
    
    if url_pattern in example:
        return False
    
    return True

def no_pii_haiku(example):
    """
    Checks for <personal and <public tags in the haiku dataset.
    """

    if "<personal" in example:
        return False
    
    if "<public" in example:
        return False
    
    return True


def remove_here_is(x):

    if x.strip().startswith("Here"):
        return x.split("\n\n", maxsplit=1)[1].strip()
    
    return x

def no_errors(x):
    return "<|ERROR|>" not in x

if __name__ == "__main__":
    main()

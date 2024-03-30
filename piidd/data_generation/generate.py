import os
import json
import random


from dotenv import load_dotenv
from datasets import Dataset

from piidd.data_generation.utils import (
    random_string,
    inspirational_people,
    first_names,
    bios,
    relations,
)

load_dotenv("../../.env")


basic_prompt = open("./essay_instructions/basic_blog_post.txt", "r").read()


def make_full_prompt(example):

    temp = "# Instructions\n\n"

    temp += basic_prompt + "\n\n"

    bio = random.choice(bios["bio"])

    temp += "Write from the standpoint of the following person:\n\n" + bio + "\n\n"

    temp += "## Guidelines\n\n"

    temp += " - Write in 1st person\n"
    temp += " - Mention your website in the middle of the essay and use <<URL>> as a placeholder.\n"

    relation = random.choice(relations)
    rel_name = random.choice(first_names)
    temp += f" - Mention somewhere in the essay how your {relation} {rel_name} helped you.\n"

    famous = random.choice(inspirational_people)

    temp += f" - Mention somewhere in the essay how {famous} affected you.\n"

    temp += " - The essay should be 500-1000 words\n\n"

    temp += "# Essay\n\n"

    return {
        "starting_prompt": "basic_blog_post",
        "rel_name": rel_name,
        "famous": famous,
        "bio": bio,
        "full_prompt": temp,
    }



oai_key = os.environ["OPENAI_KEY"]
from openai import OpenAI

client = OpenAI(
    api_key=oai_key,
)


def map(example):

    x = make_full_prompt(example)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": x["full_prompt"]},
            ],
        )
        text = response.choices[0].message.content
    except Exception as e:
        print(e)
        text = "<|ERROR|>"

    return {**x, "response": text}

ds = Dataset.from_dict({"id": [random_string(10) for _ in range(500)]})
ds = ds.map(map, batched=False, num_proc=16)

ds.to_parquet("gpt-3.5-turbo-v1-part4.pq")
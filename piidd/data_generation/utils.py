import json
import string
import random
from pathlib import Path

from datasets import Dataset

def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

this_dir = Path(__file__).parent.resolve()

inspirational_people = list(set(open(this_dir / "names/inspirational_people.txt", "r").read().split("\n")))

first_names = list(set(json.load(open(this_dir / "names/first_names.json"))))
last_names = list(set(json.load(open(this_dir / "names/surnames.json"))))
bios = Dataset.from_json(str(this_dir / "names/professional_bios.json"))
relations = [
    "friend",
    "colleague",
    "family member",
    "mentor",
    "coworker",
    "teammate",
    "acquaintance",
    "partner",
    "classmate",
    "roommate",
    "neighbor",
    "associate",
    "advisor",
    "club member",
    "peer",
    "companion",
    "fellow volunteer",
    "protege",
    "sponsor",
    "confidant",
    "collaborator",
    "mentor",
    "ally",
    "consort",
    "disciple",
    "follower",
    "mentee",
    "subordinate",
    "supervisor",
    "trainee",
    "apprentice",
    "comrade",
]


prompts = list(map(str, (this_dir / "essay_instructions").glob("*.txt")))

# remove original prompt as it is not good for LLMs
prompts = [open(p).read().strip() for p in prompts if "original" not in p]
import random
import re
import string
import unicodedata
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from faker import Faker
from spacy.lang.en import English

dotenv_path = Path("../../.env")
if dotenv_path.exists():
    print("Loaded .env file!")
    load_dotenv(str(dotenv_path))

this_dir = Path(__file__).parent

first_names = json.load(
    open(Path(os.environ["PROJECT_HOME_DIR"]) / "data/first_names.json")
)
last_names = json.load(
    open(Path(os.environ["PROJECT_HOME_DIR"]) / "data/surnames.json")
)

domains = open(this_dir / "adding_pii" / "domains.txt").read().split("\n")
website_terms = open(this_dir / "adding_pii" / "website_terms.txt").read().split("\n")
all_endings = open(this_dir / "adding_pii" / "endings.txt").read().split("\n")
email_endings = open(this_dir / "adding_pii" / "emails.txt").read().split("\n")
website_terms = open(this_dir / "adding_pii" / "website_terms.txt").read().split("\n")
file_extensions = (
    open(this_dir / "adding_pii" / "file_extensions.txt").read().split("\n")
)

fake = Faker()
en_tokenizer = English().tokenizer


def make_last_name(last_names):
    # 10% of the time, hypenate

    if random.random() < 0.1:
        names = random.sample(last_names, k=2)

        return names[0].title() + "-" + names[1].title()

    return random.choice(last_names).title()


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def make_website_name(first_name, last_name):

    # 40% of the time, use first name last name
    # 15% of the time, use first initial last name
    # 15% of the time, use first name, last initial
    # 30% of the time, use last name only

    name = ""

    if random.random() < 0.4:
        delimiter = ["", "-", "_"]

        name = first_name + random.choice(delimiter) + last_name

    elif random.random() < 0.15:
        delimiter = ["", "-", "_"]

        name = first_name[0] + random.choice(delimiter) + last_name

    elif random.random() < 0.15:
        delimiter = ["", "-", "_"]

        name = first_name + random.choice(delimiter) + last_name[0]

    else:
        name = last_name

    cleaned = remove_accents(name)

    return re.sub(r"[^a-zA-Z0-9\.\-\_]", "", cleaned)


def generate_youtube_link():

    base = [
        "https://www.youtube.com/user/",
        "https://www.youtube.com/channel/",
        "https://www.youtube.com/watch?v=",
        "https://youtu.be/",
    ]

    random_string = "".join(
        random.choices(string.ascii_letters + string.digits, k=random.randint(8, 12))
    )

    return random.choice(base) + random_string


def generate_website_path():

    first_name = random.choice(first_names)
    while first_name == "":
        first_name = random.choice(first_names)
    last_name = make_last_name(last_names)

    # Sample names, domains, directories, filenames, and extensions

    url_starts = [
        "http://",
        "https://",
        "http://www.",
        "https://www.",
    ]

    start = random.choice(url_starts)

    name = make_website_name(first_name, last_name).lower().replace(" ", "")

    domain = random.choice(domains)

    path = generate_random_url_path(website_terms, length=random.choice([1, 2, 3, 4]))

    return start + name + "." + domain + "/" + path


def generate_random_url_path(terms, length=3):
    """
    Generates a random website URL path by combining multiple terms.

    :param terms: List of terms to choose from.
    :param length: Length of the URL path (number of terms to combine).
    :return: A randomly generated URL path.
    """
    if length < 1 or length > len(terms):
        raise ValueError("Length must be between 1 and the number of terms provided.")

    # Shuffle the terms list to make the selection random
    random.shuffle(terms)

    # Select a random subset of terms based on the specified length
    selected_terms = terms[:length]

    # Join the selected terms to create the URL path
    url_path = "/".join(selected_terms)

    return url_path


def generate_social_username():

    first_name = random.choice(first_names)
    while first_name == "":
        first_name = random.choice(first_names)
    last_name = make_last_name(last_names)

    name = make_website_name(first_name, last_name).lower().replace(" ", "")

    if random.random() < 0.5:
        num = str(int(100 * random.random()))

        name += num

    return name


def generate_social_media_url():

    base = [
        "https://www.facebook.com/",
        "https://www.linkedin.com/",
        "https://www.github.com/",
        "https://www.instagram.com/",
        "https://www.tiktok.com/@",
        "https://www.twitter.com/",
    ]

    return random.choice(base) + generate_social_username()


def generate_email():

    return generate_social_username() + "@" + random.choice(email_endings)


def random_header_template():
    # Placeholders
    placeholders = [
        "[STREET_ADDRESS]",
        "[PHONE_NUM]",
        "[ID_NUM]",
        "[EMAIL]",
        "[USERNAME]",
        "[SOCIAL]",
    ]

    labels = {
        "[STREET_ADDRESS]": ["address", "location", "residence", "addr.", "addr"],
        "[PHONE_NUM]": ["phone", "telephone", "contact", "mobile", "cell"],
        "[EMAIL]": ["email", "e-mail", "contact", "message", "mail"],
        "[USERNAME]": ["username", "handle", "alias", "user"],
        "[SOCIAL]": ["social", "media", "profile", "account", "page"],
        "[NAME]": ["name"],
    }

    # Possible delimiters
    delimiters = ["\n", " | ", " - ", ", ", "; "]

    # Shuffle placeholders
    random.shuffle(placeholders)

    # Choose a random number of placeholders to include in the template
    num_placeholders = random.randint(2, len(placeholders))
    selected_placeholders = ["[NAME]"] + placeholders[:num_placeholders]

    # Assemble the template
    template = ""
    for i, placeholder in enumerate(selected_placeholders):
        delimiter = random.choice(delimiters) if i > 0 else ""
        if placeholder != "[ID_NUM]" and random.random() < 0.5:
            label = random.choice(labels[placeholder])
            template += f"{delimiter} {label}"

            if random.random() < 0.5:
                template += ":"

            template += " "

            template += placeholder
        else:
            template += f"{delimiter}{placeholder}"

    return template


def random_closing_template():
    return random.choice(all_endings)


def generate_id():
    # made with gpt-4

    id_starts = ["ID", "", "PIN", "Pin", "Pin", "Roll", "interviewer", "interviewee"]

    nums = ["No", "No.", "Num", "Num.", "Number", "#", " "]

    funcs = [
        str.upper,
        str.title,
        str.lower,
    ]

    # Define the patterns observed
    patterns = [
        lambda: "".join(
            random.choices(string.digits, k=random.randint(6, 12))
        ),  # Pure Numeric
        lambda: "".join(
            random.choices(
                string.ascii_letters + string.digits, k=random.randint(8, 15)
            )
        ),  # Alphanumeric
        lambda: "".join(random.choices(string.ascii_letters, k=random.randint(2, 4)))
        + ":"
        + "".join(
            random.choices(
                string.ascii_letters + string.digits, k=random.randint(5, 10)
            )
        ),  # Random Prefix Alphanumeric
        lambda: ",".join(
            [
                "".join(random.choices(string.digits, k=random.randint(1, 2)))
                for _ in range(4)
            ]
        ),  # Mixed Format
        lambda: ".".join(random.choices(string.ascii_letters, k=2))
        + ". "
        + "".join(
            random.choices(string.digits, k=random.randint(6, 8))
        ),  # With space and period
        # lambda: "".join(random.choices(string.ascii_letters, k=2))+ " " +  "".join(
        #     random.choices(string.digits, k=random.randint(6, 8))
        # ),  # With space and no period,
        lambda: "".join(
            random.choices(string.ascii_letters + string.digits, k=random.randint(6, 8))
        )
        + "|"
        + "".join(
            random.choices(string.digits, k=random.randint(6, 8))
        ),  # Alphanumeric | numeric
    ]

    # Randomly select a pattern and generate an ID

    f1 = random.choice(funcs)
    f2 = random.choice(funcs)

    add_colon = random.random() < 0.5

    colon_str = ":" if add_colon else ""
    colon_str += " " 
    colon_str = colon_str or " "

    prefix = f1(random.choice(id_starts)) + " " + f2(random.choice(nums)) + colon_str

    while prefix.startswith(":"):
        prefix = (
            f1(random.choice(id_starts)) + " " + f2(random.choice(nums)) + colon_str
        )

    id_num = random.choice(patterns)()

    return {
        "prefix": re.sub(" {2,}", " ", prefix).lstrip(),
        "id_num": id_num,
    }


BRACKETS = ["parentheses", "square brackets", "curly braces", "none"]
def add_brackets(url, bracket_type):
    """
    Adds a space in front
    """

    if bracket_type == "parentheses":
        return f" ({url})"
    elif bracket_type == "square brackets":
        return f" [{url}]"
    elif bracket_type == "curly braces":
        return f" {{{url}}}"
    else:
        return " " + url + " "
        
def add_info_to_claude(essay, generated_urls):
    """
    Haiku will sometimes add a name in between xml tags (<personal_name>Jane Doe</personal_name>).
    This happens 1% of the time, so it is easiest to just drop the sample entirely.
    """

    # Add names
    first_name = random.choice(first_names)
    while first_name == "":
        first_name = random.choice(first_names)

    last_name = None
    if random.random() < 0.5:
        last_name = make_last_name(last_names)

    occurrences = essay.count("<personal_name>")

    for i in range(occurrences):

        if i == 0 and last_name is not None:
            essay = essay.replace("<personal_name>", f"{first_name} {last_name}", 1)
        else:
            if last_name is not None and random.random() < 0.5:
                essay = essay.replace("<personal_name>", last_name, 1)
            else:
                essay = essay.replace("<personal_name>", first_name, 1)

    # claude will sometimes insert <student_name>

    student_name = ""
    if "<student_name>" in essay:
        student_name = random.choice(first_names)
        while student_name == "":
            student_name = random.choice(first_names)

        essay = essay.replace("<student_name>", student_name)

    # public urls
    # extract urls from generated

    temp_urls = generated_urls.split("\n")

    if "<replacements>" in temp_urls[0]:
        temp_urls = temp_urls[1:]

    if "</replacements>" in temp_urls[-1]:
        temp_urls = temp_urls[:-1]

    try:
        while (
            "http" not in temp_urls[0]
            and "www" not in temp_urls[0]
            and ".com" not in temp_urls[0]
        ):
            temp_urls = temp_urls[1:]
    except Exception as e:
        print(e)
        print(generated_urls)

    temp_urls = [x for x in temp_urls if x != ""]

    try:
        if "<public" in temp_urls[0]:
            temp_urls = [x.split(">")[1].split("<")[0] for x in temp_urls]
    except Exception as e:
        print(e)
        print(generated_urls)
        print(temp_urls)

    idx = 0

    if "<public_url>" in essay:

        if "</public_url>" in essay:
            essay = essay.replace("<public_url>", "")
            essay = essay.replace(
                "</public_url>", add_brackets(temp_urls[idx], random.choice(BRACKETS))
            )
        else:
            essay = essay.replace(
                "<public_url>", add_brackets(temp_urls[idx], random.choice(BRACKETS))
            )

        idx += 1

    i = 1

    while f"<public_url{i}>" in essay:

        if f"</public_url{i}>" in essay:
            essay = essay.replace(f"<public_url{i}>", "")
            essay = essay.replace(
                f"</public_url{i}>",
                add_brackets(temp_urls[idx], random.choice(BRACKETS)),
            )
        else:
            essay = essay.replace(
                f"<public_url{i}>",
                add_brackets(temp_urls[idx], random.choice(BRACKETS)),
            )

        i += 1
        idx += 1

    # personal urls

    personal_url = ""
    if "<personal_url>" in essay:
        personal_url = generate_website_path()

        start_idx = essay.index("<personal_url>")

        if any(
            [
                x in essay[max(start_idx - 40, 0) : start_idx + 40].lower()
                for x in ["youtube", "video", "channel", "vlog", "stream"]
            ]
        ):
            personal_url = generate_youtube_link()

        if "</personal_url>" in essay:
            essay = essay.replace("<personal_url>", "")
            essay = essay.replace(
                "</personal_url>", add_brackets(personal_url, random.choice(BRACKETS))
            )
        else:
            essay = essay.replace(
                "<personal_url>", add_brackets(personal_url, random.choice(BRACKETS))
            )

    header_details = add_header(essay)

    return {
        **header_details,
        "personal_first_name": first_name,
        "personal_last_name": last_name or "",
        "public_urls": temp_urls,
        "personal_url": personal_url,
        "new_essay": re.sub(" {2,}", " ", essay),
        "student_name": student_name,
    }


def add_info_to_mixtral(details, url_pattern="<<URL>>"):

    essay = details["essay"]

    header_details = add_header(essay)

    essay = header_details.pop("essay")

    personal_url = generate_website_path()

    essay = re.sub(url_pattern, add_brackets(personal_url, random.choice(BRACKETS)), essay)

    name = header_details["name"]

    essay = add_ending(essay, name)

    return {
        **header_details,
        "personal_url": personal_url,
        "essay": re.sub(" {2,}", " ", essay),
        "names": details["names"],
    }



def add_random_social(essay):
    another_social = ""
    if random.random() < 0.2 and len(essay) > 400:
        # add another social url somewhere
        another_social = generate_social_media_url()

        idx = random.randint(300, len(essay))

        while idx < len(essay) and not essay[idx].isspace():
            idx += 1

        if idx >= len(essay):
            idx = len(essay) - 1

        essay = essay[:idx] + f" ({another_social}) " + essay[idx:]


def add_another_personal(essay):
    another_personal = ""
    if random.random() < 0.2 and len(essay) > 400:
        # add another personal website

        another_personal = generate_website_path()

        idx = random.randint(300, len(essay))

        while idx < len(essay) and not essay[idx].isspace():
            idx += 1

        if idx >= len(essay):
            idx = len(essay) - 1

        essay = essay[:idx] + f" ({another_personal}) " + essay[idx:]


def add_header(essay):
    first_name = random.choice(first_names)
    while first_name == "":
        first_name = random.choice(first_names)
    last_name = make_last_name(last_names)

    header_template = random_header_template()

    address = fake.address()

    username = generate_social_username()

    username = re.sub(r"[^a-zA-Z0-9]", "", username)

    email = generate_email()

    id_nums = generate_id()

    phone_number = fake.phone_number()

    social = generate_social_media_url()

    header = header_template.replace("[NAME]", f"{first_name} {last_name}")
    header = header.replace("[STREET_ADDRESS]", address)
    header = header.replace("[PHONE_NUM]", phone_number)
    header = header.replace("[ID_NUM]", id_nums["prefix"] + id_nums["id_num"])
    header = header.replace("[EMAIL]", email)
    header = header.replace("[SOCIAL]", social)
    header = header.replace("[USERNAME]", username)

    essay = header + "\n\n" + essay

    if random.random() < 0.5 and len(essay) > 1000:
        # add another header between 1000 and 2000 chars

        idx = random.randint(1000, min(len(essay), 2000))

        essay = essay[:idx] + "\n\n" + header + "\n\n" + essay[idx:]

    if random.random() < 0.5 and len(essay) > 2000:
        # add another header between 2000 and 4000 chars

        idx = random.randint(2000, min(len(essay), 4000))

        essay = essay[:idx] + "\n\n" + header + "\n\n" + essay[idx:]

    if random.random() < 0.5 and len(essay) > 4000:
        # add another header between 4000 and 8000 chars

        idx = random.randint(4000, min(len(essay), 8000))

        essay = essay[:idx] + "\n\n" + header + "\n\n" + essay[idx:]

    return {
        "essay": essay,
        "name": f"{first_name} {last_name}",
        "address": address,
        "username": username,
        "email": email,
        "id_num": id_nums["id_num"],
        "phone_number": phone_number,
        "social": social,
    }


def add_ending(essay, name):
    ending = random_closing_template()

    if random.random() < 0.5:
        ending = ending.replace("[NAME]", "\n\n" + name)

    else:
        ending = ending.replace("[NAME]", name)

    essay = essay + "\n\n" + ending

    return essay


def add_pii(essay):
    """
    Used to make v1 mixtral dataset
    https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated
    """

    header_details = add_header(essay)
    essay = header_details.pop("essay")
    essay = add_ending(essay, header_details["name"])

    return {
        "essay": essay,
        **header_details,
    }


key2label = {
    "name": "NAME_STUDENT",
    "personal_url": "URL_PERSONAL",
    "address": "STREET_ADDRESS",
    "username": "USERNAME",
    "email": "EMAIL",
    "id_num": "ID_NUM",
    "phone_number": "PHONE_NUM",
    "social": "URL_PERSONAL",
    "another_social": "URL_PERSONAL",
    "another_personal": "URL_PERSONAL",
    "names": "NAME_STUDENT",
}


def to_tokens(x):

    char_labels = ["O"] * len(x["essay"])

    for key, value in x.items():
        if key == "essay":
            continue
        if value == "":
            continue

        if key == 'names' and "|" in value:
            matches = []
            for n in value.split("|"):
                matches.extend(list(re.finditer(r"\b" + re.escape(n) + r"\b", x["essay"])))

        else:
            pattern = re.escape(value)

            if "name" in key:
                pattern = r"\b" + pattern + r"\b"

            matches = re.finditer(pattern, x["essay"])

        for m in matches:
            for i in range(m.start(), m.end()):
                char_labels[i] = key2label[key]

    token_labels = []
    tokens = []
    trailing_whitespace = []

    doc = en_tokenizer(x["essay"])

    for token in doc:
        tokens.append(token.text)
        trailing_whitespace.append(bool(token.whitespace_))

        # 0: if "O" then "O"
        # scenario 1: nothing before ("B-")
        # scenario 2: "O" before ("B-")
        # 3: "B-label" before ("I-label")
        # 4: "B-labelx" or "I-labelx" before ("B-labely")

        idx = token.idx

        label = char_labels[idx]

        if label == "O":
            token_labels.append("O")

        else:

            # if "O" came before, then do "B-"
            if len(token_labels) == 0:
                token_labels.append("B-" + label)

            elif token_labels[-1] == "O":
                token_labels.append("B-" + label)

            # different labels, do "B-"
            elif token_labels[-1][2:] != label:
                token_labels.append("B-" + label)

            # same labels, do "I-"
            else:
                token_labels.append("I-" + label)

    return {
        "tokens": tokens,
        "labels": token_labels,
        "trailing_whitespace": trailing_whitespace,
    }

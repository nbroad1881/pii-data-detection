import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

def main():

    data = json.load(open("data.json"))

    model = AutoModelForCausalLM.from_pretrained("")

if __name__ == "__main__":
    main()
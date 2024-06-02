import os
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import upload_folder, create_repo

this_dir = Path(__file__).parent

dotenv_path = this_dir.parent / ".env"
if dotenv_path.exists():
    print("Loaded .env file!")
load_dotenv(str(dotenv_path))

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_TOKEN"] = ""


def upload_model(path, repo_id, private=True):

    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="model",
            token=os.environ["HF_TOKEN"],
        )
    except:
        pass

    upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.pt", "*.pth"],
        token=os.environ["HF_TOKEN"],
    )



upload_model("/drive2/kaggle/pii-dd/piidd/training/basic/multirun/2024-04-20/21-44-52/2/checkpoint-4740", "nbroad/floral-bird-887")
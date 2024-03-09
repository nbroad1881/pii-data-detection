import os
import json
from pathlib import Path


def upload_kaggle_dataset(storage_dir, dataset_name, owner):
    """
    :param storage_dir: upload storage dir to kaggle as dataset
    :param dataset_name: name of the dataset
    :param owner: name of the dataset owner
    """
    print("creating metadata...")
    os.system(f"kaggle datasets init -p {storage_dir}")

    print("updating metadata...")
    with open(os.path.join(storage_dir, "dataset-metadata.json"), "r") as f:
        metadata = json.load(f)

    metadata["title"] = dataset_name
    metadata["id"] = f"{owner}/{dataset_name}".replace("_", "-")

    print("saving updated metadata...")
    with open(os.path.join(storage_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("Deleting optimizer.pt, scheduler.pt, and rng.pth ...")
    d = Path(storage_dir)
    if (d / "optimizer.pt").exists():
        os.remove(d / "optimizer.pt")

    if (d / "scheduler.pt").exists():
        os.remove(d / "scheduler.pt")

    if (d / "rng_state.pth").exists():
        os.remove(d / "rng_state.pth")

    print("uploading the dataset ...")
    os.system(f"kaggle datasets create -p {storage_dir}")
    print("done!")


upload_kaggle_dataset(
    "/drive2/kaggle/pii-dd/piidd/training/assertion/mistral-7b/checkpoint-4887",
    dataset_name="pii-dd-mistral7b-multiclass-v1",
    owner="nbroad"
)

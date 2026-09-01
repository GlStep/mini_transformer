from pathlib import Path
from datasets import DatasetDict, load_dataset

LANG_PAIRS = ["de-en", "en-ru", "en-zh"]
RAW_DIR = Path(__file__).parent / "raw"
DATASET_STRING = "Helsinki-NLP/opus-100"

def download(sample_size: int | None = None):
    for pair in LANG_PAIRS:
        train_split = f"train[:{sample_size}]" if sample_size else "train"
        ds = DatasetDict({
            "train": load_dataset(DATASET_STRING, pair, split=train_split),
            "validation": load_dataset(DATASET_STRING, pair, split="validation"),
            "test": load_dataset(DATASET_STRING, pair, split="test"),
        })
        output_dir = RAW_DIR / pair
        output_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(output_dir)
        print(f"Downloaded {pair} dataset to {output_dir}")

if __name__ == "__main__":
    download(sample_size=200000)

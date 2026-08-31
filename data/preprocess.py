import unicodedata
from pathlib import Path
from typing import cast

from datasets import DatasetDict, load_from_disk

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"

LANG_PAIRS = ["de-en", "en-ru", "en-zh"]

LEN_RATIO_BOUNDS = {
    "de-en": (0.34, 3.0),
    "en-ru": (0.34, 3.0),
    "en-zh": (0.8, 7.0),
}

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text

def length_ratio_ok(src: str, tgt: str, pair: str) -> bool:
    src_len = len(src) # Important for chinese, as there are no whitespaces and otherwise would just be 1
    tgt_len = len(tgt)
    if src_len == 0 or tgt_len == 0:
        return False
    ratio = src_len / tgt_len
    min_ratio, max_ratio = LEN_RATIO_BOUNDS[pair]
    return min_ratio <= ratio <= max_ratio

def preprocess_pair(pair: str):
    src_lang, tgt_lang = pair.split("-")
    ds = cast(DatasetDict, load_from_disk(RAW_DIR / pair))

    def clean_example(example):
        translation = example["translation"]
        return {
            "translation": {
                src_lang: normalize_text(translation[src_lang]),
                tgt_lang: normalize_text(translation[tgt_lang]),
            }
        }

    def filter_example(example):
        translation = example["translation"]
        return length_ratio_ok(translation[src_lang], translation[tgt_lang], pair)

    def dedup(dataset):
        seen = set()

        def is_new(example):
            translation = example["translation"]
            key = (translation[src_lang], translation[tgt_lang])
            if key in seen:
                return False
            seen.add(key)
            return True

        return dataset.filter(is_new)

    cleaned = ds.map(clean_example)
    filtered = cleaned.filter(filter_example)

    deduped = DatasetDict({
        split: dedup(filtered[split]) for split in filtered.keys()
    })

    output_dir = PROCESSED_DIR / pair
    deduped.save_to_disk(output_dir)
    print(f"Preprocessed {pair} dataset saved to {output_dir}")

if __name__ == "__main__":
    for pair in LANG_PAIRS:
        preprocess_pair(pair)

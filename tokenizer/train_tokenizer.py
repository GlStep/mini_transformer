from pathlib import Path
from typing import Any, cast

import sentencepiece as spm
from datasets import DatasetDict, load_from_disk

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
TOKENIZER_DIR = Path(__file__).parent / "trained"
CORPUS_PATH = TOKENIZER_DIR / "corpus.txt"

LANG_PAIRS = ["de-en", "en-ru", "en-zh"]
VOCAB_SIZE = 16000

SPECIAL_TOKENS = ["<2en>", "<2de>", "<2ru>", "<2zh>"]


def build_corpus():
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for pair in LANG_PAIRS:
            ds = cast(DatasetDict, load_from_disk(PROCESSED_DIR / pair))
            train = ds["train"]
            for example in train:
                example = cast(dict[str, Any], example)
                translation = example["translation"]
                f.write(translation[pair.split("-")[0]] + "\n")
                f.write(translation[pair.split("-")[1]] + "\n")

def train_tokenizer() -> None:
    spm.SentencePieceTrainer.train(
        input=str(CORPUS_PATH),
        model_prefix=str(TOKENIZER_DIR / "spm"),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9999,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=SPECIAL_TOKENS,
    )

if __name__ == "__main__":
    build_corpus()
    train_tokenizer()
    print(f"Tokenizer trained and saved to {TOKENIZER_DIR}")


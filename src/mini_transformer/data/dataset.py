from pathlib import Path
from typing import Any, cast

import sentencepiece as spm
import torch
from datasets import DatasetDict, load_from_disk
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
            self,
            pairs: list[str],
            processed_dir: Path,
            sp_model_path: Path,
            split: str = "train"
            ):
        self.sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        self.examples: list[tuple[str, str, str]] = []

        for pair in pairs:
            lang_a, lang_b = pair.split("-")
            ds = cast(DatasetDict, load_from_disk(processed_dir / pair))[split]

            for example in ds:
                example = cast(dict[str, Any], example)
                translation = example["translation"]
                text_a = translation[lang_a]
                text_b = translation[lang_b]
                self.examples.append((text_a, text_b, lang_b))
                self.examples.append((text_b, text_a, lang_a))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        src_text, tgt_text, tgt_lang = self.examples[idx]
        src_ids = [self.sp.piece_to_id(f"<2{tgt_lang}>")] + self.sp.encode(src_text, out_type=int)
        tgt_ids = [self.bos_id] + self.sp.encode(tgt_text, out_type=int) + [self.eos_id]
        return {"src_ids": src_ids, "tgt_ids": tgt_ids}

def collate_fn(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    src_lens = [len(item["src_ids"]) for item in batch]
    tgt_lens = [len(item["tgt_ids"]) for item in batch]

    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    src_batch = torch.full((len(batch), max_src_len), pad_id, dtype=torch.long)
    tgt_batch = torch.full((len(batch), max_tgt_len), pad_id, dtype=torch.long)

    for i, item in enumerate(batch):
        src_len = len(item["src_ids"])
        tgt_len = len(item["tgt_ids"])
        src_batch[i, :src_len] = torch.tensor(item["src_ids"], dtype=torch.long)
        tgt_batch[i, :tgt_len] = torch.tensor(item["tgt_ids"], dtype=torch.long)

    return {
        "src_ids": src_batch,
        "tgt_ids": tgt_batch,
    }

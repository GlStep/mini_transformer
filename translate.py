import argparse

import torch
import sentencepiece as spm

from mini_transformer.model.transformer import Transformer
from mini_transformer.config import load_config
from pathlib import Path

cfg = load_config("config.yaml")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(vocab_size: int, pad_id: int, device: torch.device) -> Transformer:
    model = Transformer(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        n_layers=cfg.model.n_layers,
        pad_id=pad_id,
        vocab_size=vocab_size,
    )

    state_dict = torch.load(Path(cfg.training.checkpoint_dir) / "transformer_epoch_8.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

@torch.no_grad()
def greedy_decode(model: Transformer, sp: spm.SentencePieceProcessor, src_sentence: str, tgt_lang: str, device: torch.device) -> str:
    tag_id = sp.piece_to_id(f"<2{tgt_lang}>")
    src_ids = [tag_id] + sp.encode(src_sentence, out_type=int)
    src = torch.tensor([src_ids], device=device)

    src_mask = model.make_padding_mask(src)
    encoder_output = model.encode(src, src_mask=src_mask)

    tgt_ids = [sp.bos_id()]

    for _ in range(cfg.model.max_len):
        tgt = torch.tensor([tgt_ids], device=device)
        tgt_self_mask = model.make_causal_mask(tgt.size(1))

        decoder_output = model.decode(tgt, encoder_output, self_mask=tgt_self_mask, cross_mask=src_mask)
        logits = model.output_proj(decoder_output)

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()

        tgt_ids.append(int(next_token_id))

        if next_token_id == sp.eos_id():
            break

    return sp.decode(tgt_ids[1:], out_type=str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Text to translate")
    parser.add_argument("--lang", type=str, default="en", help="Target language (e.g.: en, de, ru, zh)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    sp = spm.SentencePieceProcessor(model_file=str(Path(cfg.tokenizer.model_path)))
    vocab_size = sp.get_piece_size()

    model = load_model(vocab_size=vocab_size, pad_id=sp.pad_id(), device=device)

    translated_text = greedy_decode(model, sp, args.text, args.lang, device)
    print(f"Translated text: {translated_text}")


if __name__ == "__main__":
    main()

import argparse

import torch
import sentencepiece as spm

from mini_transformer.model.transformer import Transformer
from pathlib import Path


SP_MODEL_PATH = Path("tokenizer/trained/spm.model")
CHECKPOINT_DIR = Path("checkpoints")

D_MODEL = 256
N_HEADS = 4
D_FF = 1024
N_LAYERS = 4
MAX_LEN = 96


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(vocab_size: int, pad_id: int, device: torch.device) -> Transformer:
    model = Transformer(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        pad_id=pad_id,
        vocab_size=vocab_size,
    )

    state_dict = torch.load(CHECKPOINT_DIR / "transformer_epoch_8.pt", map_location=device)
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

    for _ in range(MAX_LEN):
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

    sq = spm.SentencePieceProcessor(model_file=str(SP_MODEL_PATH))
    vocab_size = sq.get_piece_size()

    model = load_model(vocab_size=vocab_size, pad_id=0, device=device)

    translated_text = greedy_decode(model, sq, args.text, args.lang, device)
    print(f"Translated text: {translated_text}")


if __name__ == "__main__":
    main()

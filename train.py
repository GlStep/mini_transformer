from pathlib import Path
from tqdm import tqdm

import sentencepiece as spm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mini_transformer.data.dataset import TranslationDataset, collate_fn
from mini_transformer.model.transformer import Transformer

# PAIRS = ["de-en", "en-ru", "en-zh"]
PAIRS = ["de-en"]
PROCESSED_DIR = Path("data/processed")
SP_MODEL_PATH = Path("tokenizer/trained/spm.model")
CHECKPOINT_DIR = Path("checkpoints")

D_MODEL = 256
N_HEADS = 4
D_FF = 1024
N_LAYERS = 4
DROPOUT = 0.1

BATCH_SIZE = 32
LR = 3e-4
NUM_EPOCHS = 1
GRAD_CLIP = 1.0

WARMUP_STEPS = 400


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def make_masks(model: Transformer, src: torch.Tensor, tgt_input: torch.Tensor):
    src_mask = model.make_padding_mask(src)
    tgt_padding_mask = model.make_padding_mask(tgt_input)
    causal_mask = model.make_causal_mask(tgt_input.size(1))

    tgt_self_mask = tgt_padding_mask | causal_mask
    cross_mask = src_mask

    return src_mask, tgt_self_mask, cross_mask

def make_lr_lambda(d_model: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return lr_lambda


def run_epoch(model, loader, optimizer, loss_fn, device, pad_id, scheduler, train: bool):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    step_losses = []

    desc = "train" if train else "validation"
    for batch in tqdm(loader, desc=desc):
        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        src_mask, tgt_self_mask, cross_mask = make_masks(model, src, tgt_input)

        with torch.set_grad_enabled(train):
            logits = model(src, tgt_input, src_mask=src_mask, self_mask=tgt_self_mask, cross_mask=cross_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()

        n_tokens = (tgt_target != pad_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        step_losses.append(loss.item())

    return total_loss / total_tokens, step_losses

def main():
    device = get_device()
    print(f"Using device: {device}")

    all_train_losses = []

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL_PATH))
    pad_id = sp.pad_id()
    vocab_size = sp.get_piece_size()

    train_ds = TranslationDataset(PAIRS, PROCESSED_DIR, SP_MODEL_PATH, split="train")
    val_ds = TranslationDataset(PAIRS, PROCESSED_DIR, SP_MODEL_PATH, split="validation")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    model = Transformer(vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, n_layers=N_LAYERS, dropout=DROPOUT, pad_id=pad_id).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_lr_lambda(D_MODEL, WARMUP_STEPS))
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_losses = run_epoch(model, train_loader, optimizer, loss_fn, device, pad_id, scheduler, train=True)
        val_loss, _ = run_epoch(model, val_loader, optimizer, loss_fn, device, pad_id, scheduler, train=False)

        all_train_losses.extend(train_losses)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        checkpoint_path = CHECKPOINT_DIR / f"transformer_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    plt.plot(all_train_losses)
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(CHECKPOINT_DIR / "training_loss.png")
    print(f"Saved training loss plot: {CHECKPOINT_DIR / 'training_loss.png'}")

if __name__ == "__main__":
    main()

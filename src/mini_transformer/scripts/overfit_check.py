import torch
import torch.nn as nn
from typing import Any
from mini_transformer.model.transformer import Transformer


def main():
    torch.manual_seed(0)
    loss: Any = None

    vocab_size = 50
    d_model = 32
    n_heads = 4
    d_ff = 64
    n_layers = 2
    pad_id = 0
    batch_size = 4
    seq_len = 8

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        pad_id=pad_id
    )

    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, vocab_size, (batch_size, seq_len))

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    causal_mask = model.make_causal_mask(tgt_input.size(1))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    for step in range(1000):
        optimizer.zero_grad()

        logits = model(src, tgt_input, self_mask=causal_mask)
        loss = loss_fn(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print("Final Loss:", loss.item())

if __name__ == "__main__":
    main()


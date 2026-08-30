import torch

from mini_transformer.model.transformer import Transformer


def test_transformer_forward_shape():
    vocab_size = 50
    d_model = 32
    n_heads = 4
    d_ff = 64
    n_layers = 2
    pad_id = 0
    batch_size = 2
    src_len = 7
    tgt_len = 5

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        pad_id=pad_id,
    )

    src = torch.randint(1, vocab_size, (batch_size, src_len))
    src[:, -2:] = pad_id
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
    tgt[:, -1:] = pad_id

    src_mask = model.make_padding_mask(src)          # (batch, 1, 1, src_len)
    tgt_padding_mask = model.make_padding_mask(tgt)  # (batch, 1, 1, tgt_len)
    causal_mask = model.make_causal_mask(tgt_len)    # (1, 1, tgt_len, tgt_len)

    tgt_self_mask = tgt_padding_mask | causal_mask

    logits = model(
        src, tgt,
        src_mask=src_mask,
        self_mask=tgt_self_mask,
        cross_mask=src_mask,
    )

    assert logits.shape == (batch_size, tgt_len, vocab_size)


def test_transformer_weight_tying():
    model = Transformer(
        vocab_size=50, d_model=32, n_heads=4, d_ff=64, n_layers=2,
    )
    assert model.output_proj.weight is model.embedding.weight

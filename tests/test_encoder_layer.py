import pytest
import torch

from mini_transformer.model.encoder_layer import EncoderLayer


def test_encoder_layer_shape():
    d_model = 512
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_length = 10

    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
    x = torch.rand(batch_size, seq_length, d_model)

    output = encoder_layer(x)

    assert output.shape == (batch_size, seq_length, d_model)

def test_encoder_layer_with_and_without_mask():
    d_model = 512
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_length = 10

    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
    x = torch.rand(batch_size, seq_length, d_model)
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()

    output_with_mask = encoder_layer(x, mask)
    output_without_mask = encoder_layer(x)

    assert output_with_mask.shape == (batch_size, seq_length, d_model)
    assert output_without_mask.shape == (batch_size, seq_length, d_model)
    assert not torch.allclose(output_with_mask, output_without_mask)


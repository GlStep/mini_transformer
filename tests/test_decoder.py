import pytest
import torch

from mini_transformer.model.decoder import Decoder


def test_decoder_shape():
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 6
    seq_len = 10
    batch_size = 2

    decoder = Decoder(d_model, n_heads, d_ff, n_layers)
    x = torch.rand(batch_size, seq_len, d_model)
    memory = torch.rand(batch_size, seq_len, d_model)
    output = decoder(x, memory)

    assert output.shape == (batch_size, seq_len, d_model)

def test_decoder_with_mask():
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 6
    seq_len = 10
    batch_size = 2

    decoder = Decoder(d_model, n_heads, d_ff, n_layers)
    x = torch.rand(batch_size, seq_len, d_model)
    memory = torch.rand(batch_size, seq_len, d_model)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    output = decoder(x, memory, self_mask=mask)

    assert output.shape == (batch_size, seq_len, d_model)

def test_decoder_with_different_layer_sizes():
    d_model = 256
    n_heads = 4
    d_ff = 1024
    n_layers = 1
    seq_len = 8
    batch_size = 2

    decoder = Decoder(d_model, n_heads, d_ff, n_layers)
    x = torch.rand(batch_size, seq_len, d_model)
    memory = torch.rand(batch_size, seq_len, d_model)
    output = decoder(x, memory)

    assert output.shape == (batch_size, seq_len, d_model)

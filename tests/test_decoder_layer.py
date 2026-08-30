import pytest
import torch

from mini_transformer.model.decoder_layer import DecoderLayer


def test_decoder_layer_shape_with_different_src_tgt_lengths():
    d_model = 512
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    tgt_seq_length = 10
    src_seq_length = 15

    decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
    x = torch.rand(batch_size, tgt_seq_length, d_model)
    enc_output = torch.rand(batch_size, src_seq_length, d_model)

    self_mask = torch.triu(torch.ones(tgt_seq_length, tgt_seq_length), diagonal=1).bool()
    cross_mask = torch.triu(torch.ones(tgt_seq_length, src_seq_length), diagonal=1).bool()

    output = decoder_layer(x=x, enc_output=enc_output, self_mask=self_mask, cross_mask=cross_mask)

    assert output.shape == (batch_size, tgt_seq_length, d_model)


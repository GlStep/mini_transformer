import pytest
import torch

from mini_transformer.model.feed_forward import PositionwiseFeedForward


def test_positionwise_feed_forward_shape():
    d_model = 512
    d_ff = 2048
    batch_size = 2
    seq_len = 10

    ff = PositionwiseFeedForward(d_model, d_ff)
    x = torch.rand(batch_size, seq_len, d_model)
    output = ff(x)

    assert output.shape == (batch_size, seq_len, d_model)


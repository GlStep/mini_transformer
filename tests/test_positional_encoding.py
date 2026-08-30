import pytest
import torch

from mini_transformer.model.positional_encoding import PositionalEncoding


def test_positional_encoding_output_shape():
    d_model = 8
    seq_len = 5
    batch_size = 2
    pe = PositionalEncoding(d_model, max_len=50)

    x = torch.zeros(batch_size, seq_len, d_model)
    output = pe(x)

    assert output.shape == (batch_size, seq_len, d_model)

def test_positional_encoding_differs_by_position():
    d_model = 8
    seq_len = 5
    batch_size = 2
    pe = PositionalEncoding(d_model, max_len=50)

    x = torch.zeros(batch_size, seq_len, d_model)

    output = pe(x)

    pos1 = output[:, 0]
    pos2 = output[:, 1]

    # The outputs should differ because of the positional encoding
    assert not torch.allclose(pos1, pos2)

def test_positional_encoding_position_zero_closed_form():
    d_model = 8
    seq_len = 5
    batch_size = 2
    pe = PositionalEncoding(d_model, max_len=50)

    x = torch.zeros(batch_size, seq_len, d_model)

    output = pe(x)

    # The first position should have the same encoding for all batches
    pos0_batch1 = output[0, 0]
    pos0_batch2 = output[1, 0]
    expected_pos0 = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])  # sin(0) = 0, cos(0) = 1

    assert torch.allclose(pos0_batch1, expected_pos0)
    assert torch.allclose(pos0_batch2, expected_pos0)

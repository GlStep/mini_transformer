import pytest
import torch

from mini_transformer.model.attention import attention, MultiHeadAttention

def test_attention_output_shape():
    query = torch.rand(2, 3, 4)  # (batch_size, seq_len, d_model)
    key = torch.rand(2, 3, 4)
    value = torch.rand(2, 3, 4)
    output, attn = attention(query, key, value)
    assert output.shape == (2, 3, 4)
    assert attn.shape == (2, 3, 3)

def test_attention_with_mask():
    seq_len = 4
    query = torch.rand(2, seq_len, 4)
    key = torch.rand(2, seq_len, 4)
    value = torch.rand(2, seq_len, 4)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # Upper triangular mask
    attn = attention(query, key, value, mask=mask)[1]
    assert torch.all(attn[..., mask] < 1e-6)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, seq_len), atol=1e-6)  # Ensure attention weights sum to 1 along the last dimension

def test_multihead_attention_shape():
    d_model = 8
    num_heads = 2
    mha = MultiHeadAttention(d_model, num_heads)
    query = torch.rand(2, 3, d_model)
    key = torch.rand(2, 3, d_model)
    value = torch.rand(2, 3, d_model)
    output, attn_weights = mha(query, key, value)
    assert output.shape == (2, 3, d_model)
    assert attn_weights.shape == (2, num_heads, 3, 3)

def test_multihead_attention_head_dimensions():
    d_model = 9
    num_heads = 2
    with pytest.raises(AssertionError):
        mha = MultiHeadAttention(d_model, num_heads)

def test_multihead_attention_respects_causal_mask():
    d_model = 8
    num_heads = 2
    seq_len = 4
    batch_size = 2
    mha = MultiHeadAttention(d_model, num_heads)

    query = torch.rand(batch_size, seq_len, d_model)
    key = torch.rand(batch_size, seq_len, d_model)
    value = torch.rand(batch_size, seq_len, d_model)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # Upper triangular mask
    attn_weights = mha(query, key, value, mask=mask)[1]
    assert torch.all(attn_weights[..., mask] < 1e-6)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, num_heads, seq_len), atol=1e-6)
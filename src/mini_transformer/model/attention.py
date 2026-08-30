import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask = None) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.split_heads(self.w_q(query))
        key = self.split_heads(self.w_k(key))
        value = self.split_heads(self.w_v(value))

        attention_output, attention_weights = attention(query, key, value, mask)
        output = self.merge_heads(attention_output)
        return self.w_o(output), attention_weights


def attention(query, key, value, mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / query.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, value), attn

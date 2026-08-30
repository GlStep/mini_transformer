import torch
import torch.nn as nn

from mini_transformer.model.attention import MultiHeadAttention
from mini_transformer.model.feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_output, _ = self.self_attn(normed, normed, normed, mask)
        x = x + self.dropout(attn_output)

        # Feed-forward sublayer
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)

        return x


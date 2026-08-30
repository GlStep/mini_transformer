import torch
import torch.nn as nn

from mini_transformer.model.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

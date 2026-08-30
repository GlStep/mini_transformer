import torch
import torch.nn as nn


from mini_transformer.model.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, self_mask = None, cross_mask = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)

        return self.norm(x)

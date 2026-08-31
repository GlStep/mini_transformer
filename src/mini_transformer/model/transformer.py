import torch
import torch.nn as nn

from mini_transformer.model.encoder import Encoder
from mini_transformer.model.decoder import Decoder
from mini_transformer.model.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            n_heads: int,
            d_ff: int,
            n_layers: int,
            max_len: int = 5000,
            dropout: float = 0.1,
            pad_id: int = 0
            ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers, dropout)

        self.output_proj = nn.Linear(d_model, vocab_size)
        self.output_proj.weight = self.embedding.weight

    def encode(self, src: torch.Tensor, src_mask=None) -> torch.Tensor:
        x = self.embedding(src) * (self.d_model ** 0.5)
        x = self.dropout(self.positional_encoding(x))
        return self.encoder(x, src_mask)

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, self_mask=None, cross_mask=None) -> torch.Tensor:
        x = self.embedding(tgt) * (self.d_model ** 0.5)
        x = self.dropout(self.positional_encoding(x))
        return self.decoder(x, encoder_output, self_mask, cross_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask=None, self_mask=None, cross_mask=None) -> torch.Tensor:
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, self_mask, cross_mask)
        return self.output_proj(decoder_output)

    def make_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        return (seq == self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_causal_mask(self, seq_len: int) -> torch.Tensor:
        device = next(self.parameters()).device
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return causal_mask.unsqueeze(0).unsqueeze(1)

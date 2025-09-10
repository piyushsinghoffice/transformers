import torch
import torch.nn as nn
from .layers import MultiHeadAttention, PositionwiseFFN, PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, memory, memory, memory_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len, tied_embeddings=None):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model) if tied_embeddings is None else tied_embeddings
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src_ids, src_mask=None):
        x = self.pos(self.embed(src_ids))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len, tied_embeddings=None):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model) if tied_embeddings is None else tied_embeddings
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, tgt_ids, memory, tgt_mask=None, memory_mask=None):
        x = self.pos(self.embed(tgt_ids))
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
        tie_embeddings=False,
    ):
        super().__init__()
        tied = nn.Embedding(tgt_vocab_size, d_model) if tie_embeddings else None

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len,
                               tied_embeddings=tied if tie_embeddings else None)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len,
                               tied_embeddings=tied if tie_embeddings else None)
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if tie_embeddings:
            self.generator.weight = self.decoder.embed.weight  # weight tying

        self.src_pad_id = 0
        self.tgt_pad_id = 0

    @staticmethod
    def make_pad_mask(seq, pad_id):
        # (B,T) -> (B,1,1,T)
        return (seq == pad_id).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_causal_mask(size: int, device=None):
        # (1,1,T,T) upper-triangular True = masked
        m = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
        return m.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt_inp, src_pad_id=0, tgt_pad_id=0):
        device = src.device
        src_mask = self.make_pad_mask(src, src_pad_id)                        # (B,1,1,Ts)
        tgt_pad_mask = self.make_pad_mask(tgt_inp, tgt_pad_id)                # (B,1,1,Tt)
        causal = self.make_causal_mask(tgt_inp.size(1), device=device)        # (1,1,Tt,Tt)
        tgt_mask = tgt_pad_mask | causal

        memory = self.encoder(src, src_mask)                                  # (B,Ts,D)
        memory_mask = src_mask.expand(-1, 1, tgt_inp.size(1), -1)             # (B,1,Tt,Ts)
        dec_out = self.decoder(tgt_inp, memory, tgt_mask, memory_mask)        # (B,Tt,D)
        logits = self.generator(dec_out)                                      # (B,Tt,V)
        return logits

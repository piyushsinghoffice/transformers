import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout (B, T, D) -> (B, T, D)."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Standard multi-head scaled dot-product attention."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q,k,v: (B, T, D)
        B, Tq, _ = q.shape
        Tk = k.shape[1]

        q = self.q_proj(q).view(B, Tq, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tq,d_k)
        k = self.k_proj(k).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tk,d_k)
        v = self.v_proj(v).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tk,d_k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,H,Tq,Tk)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = attn @ v  # (B,H,Tq,d_k)

        ctx = ctx.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  # (B,Tq,D)
        out = self.o_proj(ctx)
        return self.proj_drop(out)


class PositionwiseFFN(nn.Module):
    """Two-layer feedforward network applied position-wise."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

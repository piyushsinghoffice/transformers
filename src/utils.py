import torch
from typing import Tuple

PAD, BOS, EOS = 0, 1, 2

def add_bos(tgt: torch.Tensor, bos_id: int = BOS) -> torch.Tensor:
    """Prepend <BOS> to every sequence and drop last token (teacher forcing)."""
    b = tgt.size(0)
    bos = torch.full((b, 1), bos_id, dtype=tgt.dtype, device=tgt.device)
    return torch.cat([bos, tgt[:, :-1]], dim=1)

def synthetic_copy_batch(
    batch_size: int = 32,
    seq_len: int = 12,
    vocab_size: int = 100,
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple copy task:
      src = random ints in [3, vocab_size)
      tgt = src.clone()
      tgt_inp = <BOS> + tgt[:-1]
    """
    src = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    tgt = src.clone()
    tgt_inp = add_bos(tgt)
    return src, tgt, tgt_inp

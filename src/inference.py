import torch
from .utils import BOS, EOS

@torch.no_grad()
def greedy_decode(model, src, max_len: int, bos_id: int = BOS, eos_id: int = EOS):
    """
    src: (B, Ts)
    Returns token ids (B, T_out) including BOS and up to EOS or max_len.
    """
    device = src.device
    B = src.size(0)
    ys = torch.full((B, 1), bos_id, dtype=src.dtype, device=device)  # start with BOS

    for _ in range(max_len - 1):
        logits = model(src, ys, src_pad_id=model.src_pad_id, tgt_pad_id=model.tgt_pad_id)  # (B,T,V)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)                  # (B,1)
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token == eos_id).all():
            break
    return ys

import torch
import torch.nn as nn
from tqdm import trange
from src.model import Transformer
from src.utils import synthetic_copy_batch, PAD

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Small model for quick sanity check
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        tie_embeddings=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)

    steps = 1000
    batch_size = 64
    seq_len = 16

    for step in trange(steps):
        src, tgt, tgt_inp = synthetic_copy_batch(batch_size, seq_len, vocab_size=100, device=device)
        logits = model(src, tgt_inp, src_pad_id=PAD, tgt_pad_id=PAD)   # (B,T,V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"\nstep {step+1}: loss={loss.item():.4f}")

    # quick decode demo
    model.eval()
    src, tgt, tgt_inp = synthetic_copy_batch(4, 10, 100, device)
    from src.inference import greedy_decode
    outs = greedy_decode(model, src, max_len=12)
    print("SRC:\n", src)
    print("PRED:\n", outs[:, 1:1+src.size(1)])   # drop BOS and align length

if __name__ == "__main__":
    main()

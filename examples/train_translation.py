import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import trange
import os

from src.model import Transformer
from src.utils import PAD, BOS, EOS, add_bos
from src.inference import greedy_decode

DATA_DIR = "data/opus_de_en"
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------
# Tokenizer
# ---------------------------
def prepare_tokenizer(dataset, vocab_size=8000):
    src_file = os.path.join(DATA_DIR, "src.txt")
    tgt_file = os.path.join(DATA_DIR, "tgt.txt")

    # Dump text for training tokenizers if not already done
    if not os.path.exists(os.path.join(DATA_DIR, "src-vocab.json")):
        with open(src_file, "w", encoding="utf-8") as fsrc, \
             open(tgt_file, "w", encoding="utf-8") as ftgt:
            for ex in dataset["train"]:
                fsrc.write(ex["translation"]["de"] + "\n")
                ftgt.write(ex["translation"]["en"] + "\n")

        tok_src = ByteLevelBPETokenizer()
        tok_src.train(files=src_file, vocab_size=vocab_size,
                      special_tokens=["<pad>", "<bos>", "<eos>"])
        tok_src.save_model(DATA_DIR, "src")

        tok_tgt = ByteLevelBPETokenizer()
        tok_tgt.train(files=tgt_file, vocab_size=vocab_size,
                      special_tokens=["<pad>", "<bos>", "<eos>"])
        tok_tgt.save_model(DATA_DIR, "tgt")

    # Reload tokenizers
    tok_src = ByteLevelBPETokenizer(
        os.path.join(DATA_DIR, "src-vocab.json"),
        os.path.join(DATA_DIR, "src-merges.txt")
    )
    tok_tgt = ByteLevelBPETokenizer(
        os.path.join(DATA_DIR, "tgt-vocab.json"),
        os.path.join(DATA_DIR, "tgt-merges.txt")
    )
    return tok_src, tok_tgt


# ---------------------------
# Encoding + batching
# ---------------------------
def encode_batch(batch, tok_src, tok_tgt, max_len=64):
    # German → English
    src_ids = tok_src.encode(batch["translation"]["de"]).ids[:max_len - 1] + [EOS]
    tgt_ids = tok_tgt.encode(batch["translation"]["en"]).ids[:max_len - 1] + [EOS]
    return {"src": src_ids, "tgt": tgt_ids}


def collate_fn(batch):
    max_src = max(len(x["src"]) for x in batch)
    max_tgt = max(len(x["tgt"]) for x in batch)
    src_batch, tgt_batch, tgt_inp_batch = [], [], []

    for x in batch:
        src = x["src"] + [PAD] * (max_src - len(x["src"]))
        tgt = x["tgt"] + [PAD] * (max_tgt - len(x["tgt"]))
        tgt_inp = add_bos(torch.tensor([tgt], dtype=torch.long))[0].tolist()

        src_batch.append(src)
        tgt_batch.append(tgt)
        tgt_inp_batch.append(tgt_inp)

    return (
        torch.tensor(src_batch, dtype=torch.long),
        torch.tensor(tgt_batch, dtype=torch.long),
        torch.tensor(tgt_inp_batch, dtype=torch.long),
    )


def safe_decode(tok, ids):
    """Decode while removing special tokens <pad>=0, <bos>=1, <eos>=2."""
    return tok.decode([i for i in ids if i > 2])


# ---------------------------
# Main training loop
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load dataset
    raw_dataset = load_dataset("opus100", "de-en")
    print(raw_dataset)

    # 2. Tokenizer
    tok_src, tok_tgt = prepare_tokenizer(raw_dataset)

    # 3. Encode dataset
    dataset = raw_dataset.map(
        lambda b: encode_batch(b, tok_src, tok_tgt),
        remove_columns=raw_dataset["train"].column_names
    )

    # 4. DataLoader
    train_loader = DataLoader(
        dataset["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 5. Model
    vocab_src = tok_src.get_vocab_size()
    vocab_tgt = tok_tgt.get_vocab_size()
    model = Transformer(
        vocab_src,
        vocab_tgt,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9
    )

    # 6. Training
    for epoch in range(1, 3):  # demo = 2 epochs
        model.train()
        loop = trange(len(train_loader), desc=f"Epoch {epoch}")
        for i, (src, tgt, tgt_inp) in zip(loop, train_loader):
            src, tgt, tgt_inp = src.to(device), tgt.to(device), tgt_inp.to(device)
            logits = model(src, tgt_inp, src_pad_id=PAD, tgt_pad_id=PAD)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        # 7. Eval sample
        model.eval()
        src, tgt, tgt_inp = next(iter(train_loader))
        src, tgt, tgt_inp = src.to(device), tgt.to(device), tgt_inp.to(device)
        out = greedy_decode(model, src[:1], max_len=30)
        print("\nSRC:", safe_decode(tok_src, src[0].cpu().tolist()))
        print("TGT:", safe_decode(tok_tgt, tgt[0].cpu().tolist()))
        print("PRED:", safe_decode(tok_tgt, out[0].cpu().tolist()))


if __name__ == "__main__":
    main()

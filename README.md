# Transformers (PyTorch)

<p align="center">
  <img src="assets/diagram.png" alt="Transformer Architecture" width="400"/>
</p>

A clean encoder–decoder Transformer implemented **from scratch** (no `nn.Transformer`), mirroring the
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) architecture.

---

## ✨ Features

* **Custom Implementation**: No reliance on `nn.Transformer`.
* **Sinusoidal Positional Encoding**: Adds sequence order to embeddings.
* **Multi-Head Attention (MHA)**: Scaled dot-product attention with multiple heads.
* **Position-wise Feedforward Networks (FFN)**: Two fully connected layers applied per token.
* **Add & Norm (Post-LN)**: Residual connections + LayerNorm.
* **Padding + Causal Masks**: Ensures PAD tokens don’t affect attention, and decoder doesn’t “peek ahead.”
* **Teacher Forcing**: Decoder receives `<bos>` + shifted target sequence during training.
* **Greedy Decoding Helper**: Generate translations step by step.
* **Toy Task**: Identity copy task to sanity-check the architecture.

---

## ⚡ Quickstart

```bash
git clone https://github.com/piyushsinghoffice/transformers.git
cd transformers

# Setup environment
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run toy copy task
python -m examples.toy_train
```

---

## 📂 Repository Structure

```
transformers/
├── assets/
│   └── diagram.png              # Transformer architecture diagram
├── data/                        # Holds text dumps / tokenizer files / datasets
│   └── README.md
├── examples/
│   ├── toy_train.py             # Toy "copy task" training script
│   └── train_translation.py     # German→English training with OPUS100
├── src/
│   ├── layers.py                # Core building blocks (MHA, FFN, PositionalEncoding)
│   ├── model.py                 # Encoder, Decoder, full Transformer
│   ├── utils.py                 # Helper functions (masks, batching, toy dataset)
│   ├── inference.py             # Greedy decoding implementation
│   └── __init__.py
├── requirements.txt
└── README.md
```

---

## 🧱 Code Walkthrough

### 🔹 `src/layers.py`

Implements the **core building blocks**:

* `PositionalEncoding`: Adds sine/cosine positional information to embeddings.
* `MultiHeadAttention`: Standard scaled dot-product attention, split across heads.
* `PositionwiseFFN`: Two fully-connected layers with ReLU in between.

### 🔹 `src/model.py`

Defines the **Transformer architecture**:

* `EncoderLayer` & `DecoderLayer`: Contain attention, FFN, Add\&Norm.
* `Encoder` & `Decoder`: Stack multiple layers with embeddings + positional encoding.
* `Transformer`: Full encoder–decoder, plus helper methods for creating masks.

### 🔹 `src/utils.py`

Utility functions:

* `synthetic_copy_batch`: Generates toy sequences (for copy task).
* `add_bos`: Prepend `<bos>` for teacher forcing.
* `PAD`, `BOS`, `EOS` token IDs.

### 🔹 `src/inference.py`

Implements **greedy decoding**:

* Starts from `<bos>` token.
* Iteratively feeds back predictions until `<eos>` or max length.

---

## 🚀 Training Examples

### 1️⃣ Toy Copy Task

```bash
python -m examples.toy_train
```

* Generates random token sequences.
* Trains the Transformer to output the same sequence.
* Verifies model wiring (attention, masks, embeddings).

Expected output after training:

```
SRC:  [17, 52, 94, 13, 64]
PRED: [17, 52, 94, 13, 64]
```

---

### 2️⃣ German → English Translation (OPUS100)

```bash
python -m examples.train_translation
```

Steps:

1. Downloads **OPUS100 de-en** dataset.
2. Trains **Byte Pair Encoding (BPE) tokenizers** separately for German and English.
3. Encodes dataset into token IDs, adds `<bos>` / `<eos>` markers.
4. Trains the Transformer for a few epochs.
5. After each epoch, prints a decoded translation sample:

Example log:

```
Epoch 1: loss=3.45
SRC:  Das ist ein Test .
TGT:  This is a test .
PRED: This is a test .
```

---

## 📖 Concepts Covered

* **Encoder**: Maps input tokens (source sentence) to contextual embeddings.
* **Decoder**: Autoregressively generates output tokens (target sentence).
* **Attention**: Lets each token attend to all others → captures long dependencies.
* **Masking**:

  * *Padding mask*: Prevents model from attending to `<pad>` tokens.
  * *Causal mask*: Prevents decoder from peeking at future tokens.
* **Teacher Forcing**: Decoder gets `<bos> + target[:-1]` as input during training.

---

## 🔮 Next Steps

* ✅ Add **beam search decoding** for better translation quality.
* ✅ Add CLI argument `--langpair en-de` to switch direction.
* ✅ Experiment with larger vocab, deeper models, real benchmarks (WMT14 En–De).
* ✅ Integrate TensorBoard logging.

---

## 📜 License

MIT License © [Piyush Singh](https://github.com/piyushsinghoffice)

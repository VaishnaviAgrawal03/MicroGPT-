

# MicroGPT 🧠✨

*A tiny, readable GPT-style language model you can train and tinker with in an afternoon.*

---

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [Project Structure](#project-structure)
* [Quickstart](#quickstart)
* [Training](#training)
* [Text Generation](#text-generation)
* [Config & Hyperparameters](#config--hyperparameters)
* [Results & Samples](#results--samples)
* [Roadmap](#roadmap)
* [Contributing](#contributing)


---

## Overview

**MicroGPT** is a compact implementation of a causal Transformer (GPT-style) trained on small, domain-specific corpora (e.g., WineMag reviews). The code walks through the complete pipeline—tokenization, tf.data batching, Transformer blocks, and sampling—aiming for clarity over cleverness.

## Key Features

* **Causal Transformer core**: Multi-Head Self-Attention, token & position embeddings, FFN blocks, LayerNorm, dropout.
* **End-to-end pipeline**: tf.data + TextVectorization for efficient GPU training (10K vocab, 80-token context by default).
* **Live monitoring**: ModelCheckpoint, TensorBoard, and an epoch-end sampling callback for instant qualitative evals.
* **Config-driven**: One YAML/JSON to tweak vocab, context length, heads, dims, LR, etc.
* **Hackable**: Swap in new datasets, tokenizers, or decoding strategies (temp, top-k, top-p) with minimal changes.

## Project Structure

microgpt/

├── microgpt.py               # Core model: embeddings, TransformerBlock, forward pass

├── train.py                  # Training loop & logging hooks

├── sample.py                 # Text generation / inference script

├── utils/

│   ├── tokenizer.py          # Simple tokenizer or BPE/WordPiece wrapper

│   ├── callbacks.py          # TextGenerator, custom callbacks

│   └── data_utils.py         # Helpers for loading/preprocessing

├── data/

│   ├── raw/                  # Raw corpus (e.g., winemag-data-130k-v2.json)

│   └── processed/            # Tokenized/serialized datasets

├── configs/

│   └── microgpt.yaml         # Hyperparameters & paths

├── checkpoints/              # Saved weights & logs

└── README.md

## Quickstart

 1. Create & activate a virtual env
    
python -m venv .venv

source .venv/bin/activate   # Windows: .venv\Scripts\activate

 2. Install requirements
    
pip install -r requirements.txt

 3. Prep data (example: tokenize WineMag reviews)
    
python utils/tokenizer.py \

  --input data/raw/winemag-data-130k-v2.json \
  
  --out   data/processed/winemag_tokens.pkl


## Text Generation

python sample.py \

  --checkpoint checkpoints/best.ckpt \
  
  --prompt "wine review : italy" \
  
  --max_tokens 80 \
  
  --temperature 0.8 \
  
  --top_k 50
`


## Results & Samples

| Dataset          | Epochs  | Qualitative Result               |
| ---------------- | ------  | -------------------------------- |
| WineMag 130k     | 5       | Coherent, on-topic tasting notes |
| Tiny Shakespeare | 10      | Shakespeare-esque prose          |

> Drop in a few generated snippets here to showcase quality.




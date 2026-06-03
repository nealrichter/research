# microgpt

A minimal GPT implementation in pure Python with zero dependencies. Trains a transformer language model, generates text, and demonstrates LoRA supervised fine-tuning — all using only the standard library.

Originally by [@karpathy](https://github.com/karpathy). SFT extension added for teaching post-training concepts.

## Usage

```bash
# Pretrain the base model (trains on names, saves model.json)
python3 microgpt.py

# Inference only from saved model (different results each run)
python3 microgpt.py -i
python3 microgpt.py -i model_sft.json

# LoRA SFT (loads model.json, trains adapters, saves model_sft.json)
python3 microgpt_sft.py

# SFT inference only
python3 microgpt_sft.py -i
python3 microgpt_sft.py -i model_sft.json
```

## Files

| File | Purpose |
|------|---------|
| `microgpt.py` | Pretraining + inference |
| `microgpt_sft.py` | LoRA supervised fine-tuning + inference |
| `input.txt` | Pretraining corpus (names, auto-downloaded) |
| `input_sft.txt` | SFT instruction-response pairs |
| `model.json` | Saved pretrained weights |
| `model_sft.json` | Saved SFT-merged weights |

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 1 |
| Embedding dim | 16 |
| Attention heads | 4 |
| Context length | 16 (pretrain) / 32 (SFT) |
| MLP hidden dim | 64 |

Notable differences from GPT-2: RMSNorm instead of LayerNorm, no biases, ReLU instead of GeLU.

## How It Works

### Pretraining (`microgpt.py`)

1. Loads a list of names, tokenizes each character to an integer ID
2. Feeds tokens through a GPT-2-style transformer, building a computation graph
3. Cross-entropy loss + backpropagation via topological sort
4. Adam optimizer with linear LR decay (1000 steps)
5. Saves weights to `model.json`, generates 20 hallucinated names

### SFT (`microgpt_sft.py`)

1. Loads pretrained `model.json`, freezes all base weights
2. Attaches LoRA adapters (rank 4) to Q and V attention projections
3. Trains only the 256 adapter parameters on instruction→response pairs
4. Loss computed only on response tokens (instruction is masked)
5. Merges LoRA into base weights and saves `model_sft.json`

### LoRA Details

| Parameter | Value |
|-----------|-------|
| Rank | 4 |
| Alpha | 8 |
| Target modules | attn_wq, attn_wv |
| Trainable params | 256 (5.7% of base) |
| Init | W_down random, W_up zeros |

The adapter computes: `h = W_base @ x + (α/r) * W_up @ W_down @ x`

### SFT Data Format (`input_sft.txt`)

Pipe-delimited instruction-response pairs:
```
fa|Alice
mb|Benjamin
```

## Requirements

Python 3.6+ (no external dependencies).

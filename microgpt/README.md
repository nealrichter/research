# microgpt

A minimal GPT implementation in pure Python with zero dependencies. Trains a transformer language model, generates text, and demonstrates LoRA supervised fine-tuning and DPO alignment — all using only the standard library.

Originally by [@karpathy](https://github.com/karpathy). SFT and DPO extensions added for teaching post-training concepts.

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

# DPO alignment (loads model_sft.json, trains on preferences, saves model_dpo.json)
python3 microgpt_dpo.py

# DPO inference only
python3 microgpt_dpo.py -i
python3 microgpt_dpo.py -i model_dpo.json
```

## Files

| File | Purpose |
|------|---------|
| `microgpt.py` | Pretraining + inference |
| `microgpt_sft.py` | LoRA supervised fine-tuning + inference |
| `microgpt_dpo.py` | DPO preference alignment + inference |
| `input.txt` | Pretraining corpus (names, auto-downloaded) |
| `input_sft.txt` | SFT instruction-response pairs |
| `input_dpo.txt` | DPO preference triples |
| `model.json` | Saved pretrained weights |
| `model_sft.json` | Saved SFT-merged weights |
| `model_dpo.json` | Saved DPO-merged weights |

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 1 |
| Embedding dim | 16 |
| Attention heads | 4 |
| Context length | 16 (pretrain) / 32 (SFT/DPO) |
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

### DPO (`microgpt_dpo.py`)

1. Loads SFT model `model_sft.json` as both reference policy (frozen) and policy init
2. Attaches LoRA adapters to the policy model
3. For each (prompt, chosen, rejected) triple, computes log-probs under both policy and reference
4. Optimizes the DPO loss: `-log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))`
5. Merges LoRA into base weights and saves `model_dpo.json`

### LoRA Details

| Parameter | Value |
|-----------|-------|
| Rank | 4 |
| Alpha | 8 |
| Target modules | attn_wq, attn_wv |
| Trainable params | 256 (5.7% of base) |
| Init | W_down random, W_up zeros |

The adapter computes: `h = W_base @ x + (α/r) * W_up @ W_down @ x`

### DPO Details

| Parameter | Value |
|-----------|-------|
| β (KL penalty) | 0.3 |
| Reference model | Frozen SFT checkpoint |
| Learning rate | 0.005 |
| Epochs | 30 |

### Data Formats

**SFT** (`input_sft.txt`) — pipe-delimited instruction-response pairs:
```
fa|Alice
mb|Benjamin
```

**DPO** (`input_dpo.txt`) — pipe-delimited preference triples:
```
fa|Alice|Bob
mb|Benjamin|Alice
```

## Requirements

Python 3.6+ (no external dependencies).

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

# Visualize training (ASCII loss sparkline + attention + embedding heat maps)
python3 microgpt.py --viz        # clean live line; loss chart + heat maps at the end
python3 microgpt.py --viz 100    # multi-row loss sparkline + attention matrix every 100 steps
python3 microgpt.py -h           # usage
```

## Files

| File | Purpose |
|------|---------|
| `microgpt.py` | Pretraining + inference |
| `microgpt_sft.py` | LoRA supervised fine-tuning + inference |
| `microgpt_dpo.py` | DPO preference alignment + inference |
| `microgpt_viz.py` | ASCII visualization helpers (loss sparkline + attention heat map) |
| `input.txt` | Pretraining corpus (names, auto-downloaded) |
| `input_sft.txt` | SFT instruction-response pairs |
| `input_dpo.txt` | DPO preference triples |
| `model.json` | Saved pretrained weights |
| `model_sft.json` | Saved SFT-merged weights |
| `model_dpo.json` | Saved DPO-merged weights |
| `train.log` | Appended run log — mirrors stdout (transient per-step progress excluded) |

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 1 |
| Embedding dim | 16 |
| Attention heads | 4 |
| Context length | 16 (pretrain) / 32 (SFT/DPO) |
| MLP hidden dim | 64 |

Notable differences from GPT-2: RMSNorm instead of LayerNorm, no biases, ReLU instead of GeLU.

## Visualization

`microgpt.py` has an optional `--viz [N]` flag, backed by the dependency-free
`microgpt_viz.py` module. All visualization state and logic live in the module; the
core script just calls a few hooks (each guarded so an absent `--viz` runs none of it).

- By default (no flag) training prints a one-line end-to-end summary: `Loss X -> Y (-21.3%)`.
- `--viz` (or `--viz 0`) keeps the live progress line clean (`loss V (D)`) and, at the end,
  dumps a multi-row loss chart, an attention heat map, and a token-embedding similarity map.
- `--viz N` (N > 0) prints a multi-row loss sparkline + attention matrix snapshot every N steps.

The per-step progress line is augmented to `loss V (D)`, where `D` is the change since the
last snapshot. Multi-row charts are only shown where vertical space is free (periodic
snapshots and end-of-run), never crammed onto the live carriage-return line. Loss is
EMA-smoothed and bucket-mean downsampled for display only (the reported numbers are never
altered), and normalized over the whole run so snapshots stay comparable.

The **token-embedding similarity** map (shown by `microgpt.py` at the end) is the classic
"did the embeddings learn meaning?" view: a char-vs-char cosine-similarity grid, z-scored
to ±2σ so structure (e.g. vowels clustering) stands out instead of washing out.

The sparkline renderer is a single flexible function — `sparkline(history, height=N)` —
where `height` sets how many character-rows tall the chart is (1 = inline line, 3 = chart).

| Glyph | Loss bucket (sparkline) |   | Glyph | Attention weight (heat map) |
|-------|-------------------------|---|-------|-----------------------------|
| ` ` … `█` | low … high (8 levels per row) |   | `█` `▓` `░` | high → low; `□` = masked (causal future, not computed) |

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

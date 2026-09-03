# microgpt

A minimal GPT implementation in pure Python with zero dependencies. Trains a transformer language model, generates text, and demonstrates LoRA supervised fine-tuning, DPO alignment, and GRPO reinforcement learning — all using only the standard library.

Originally by [@karpathy](https://github.com/karpathy). SFT, DPO, and GRPO extensions added for teaching post-training concepts.

## Pipeline (it's a branch, not a chain)

Pretraining and SFT run in sequence, but **DPO and GRPO are alternative
alignment steps** — both branch off the *same* SFT checkpoint. You do **not**
run DPO and then GRPO; you pick one.

```
                                          ┌── DPO  ──► model_dpo.json   (preference pairs)
pretrain ──► SFT ──► model_sft.json ──────┤
 model.json    (instruction following)    └── GRPO ──► model_grpo.json  (rule-based reward)
```

**DPO vs GRPO — which alignment step?**

|            | DPO | GRPO |
|------------|-----|------|
| Signal | **Preference pairs** you provide: `(prompt, chosen, rejected)` | **A reward function** that scores responses |
| Best when | You have human/AI preference judgments but no explicit scoring rule | You can *programmatically score* an answer (correctness, format, length) |
| Data cost | Must curate preference triples up front | Just prompts (+ a reward fn); groups sampled online, or pre-generated offline |
| Critic/value net | None (closed-form loss) | None (group-relative baseline replaces the critic) |
| Intuition | "Prefer the answer a judge liked" | "Reinforce the answers my reward function scored highest, relative to their peers" |

Rule of thumb: reach for **DPO** when the notion of "better" lives in comparisons
you've collected; reach for **GRPO** when you can *write a function* that grades
an answer. Both start from the SFT model and only train small LoRA adapters.

## Usage

> The alignment steps branch off the same `model_sft.json` — run **either** DPO
> **or** GRPO, not both in sequence (see the Pipeline section above).

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

# GRPO alignment (loads model_sft.json, group-relative RL, saves model_grpo.json)
python3 microgpt_grpo.py                              # online: samples groups at runtime
python3 microgpt_grpo.py --offline data/input_grpo_offline.jsonl   # offline: pre-generated groups

# GRPO inference only
python3 microgpt_grpo.py -i
python3 microgpt_grpo.py -i model_grpo.json

# Use a custom dataset with any script
python3 microgpt.py -d data/input_piglatin.txt
python3 microgpt_sft.py -d data/input_sft_add1.txt

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
| `microgpt_grpo.py` | GRPO reinforcement learning (online + offline) + inference |
| `microgpt_viz.py` | ASCII visualization helpers (loss sparkline + attention heat map) |
| `data/input_sft.txt` | SFT instruction-response pairs |
| `data/input_dpo.txt` | DPO preference triples |
| `data/input_grpo_offline.jsonl` | GRPO pre-generated response groups (offline mode) |
| `input.txt` | Pretraining corpus (names, auto-downloaded to CWD) |
| `model.json` | Saved pretrained weights |
| `model_sft.json` | Saved SFT-merged weights |
| `model_dpo.json` | Saved DPO-merged weights |
| `model_grpo.json` | Saved GRPO-merged weights |
| `train.log` | Appended run log — mirrors stdout (transient per-step progress excluded) |

Example datasets live in `data/`. The scripts' built-in defaults read from the
current directory, so pass datasets under `data/` explicitly, e.g.
`-d data/input_sft_add1.txt` or `--offline data/input_grpo_offline.jsonl`.

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 1 |
| Embedding dim | 16 |
| Attention heads | 4 |
| Context length | 16 (pretrain) / 32 (SFT/DPO/GRPO) |
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

### GRPO (`microgpt_grpo.py`)

1. Loads SFT model `model_sft.json` as both reference policy (frozen) and policy init
2. Attaches LoRA adapters to the policy model — **no separate critic/value network**
3. For each prompt, forms a *group* of G candidate responses:
   - **Online** (default): samples G completions from the current policy at runtime
   - **Offline** (`--offline FILE`): reads pre-generated groups from a JSONL file
4. Scores each response with a rule-based reward, then computes group-relative
   advantages: `A_i = (r_i - mean(r)) / std(r)`
5. Optimizes the clipped surrogate objective with a KL penalty to the reference:
   `-1/G · Σ [ min(ρ·A_i, clip(ρ, 1-ε, 1+ε)·A_i) - β·KL(π_θ ‖ π_ref) ]` where `ρ = π_θ/π_old`
6. Merges LoRA into base weights and saves `model_grpo.json`

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

### GRPO Details

| Parameter | Value |
|-----------|-------|
| Group size (G) | 8 (online; offline uses file) |
| β (KL penalty) | 0.1 |
| ε (clip) | 0.2 |
| Sampling temperature | 1.2 (online) |
| Reward | Rule-based (no learned reward model) |
| Learning rate | 0.001 |
| Epochs | 5 |
| Critic/value network | None (group-relative baseline) |

### Data Formats

**SFT** (`data/input_sft.txt`) — pipe-delimited instruction-response pairs:
```
fa|Alice
mb|Benjamin
```

**DPO** (`data/input_dpo.txt`) — pipe-delimited preference triples:
```
fa|Alice|Bob
mb|Benjamin|Alice
```

**GRPO offline** (`data/input_grpo_offline.jsonl`) — one pre-generated response group per line:
```json
{"prompt": "fa", "target": "Alice", "responses": ["Alice", "Bob", "Alicia", "Alice"]}
{"prompt": "mb", "target": "Benjamin", "responses": ["Ben", "Benjamin", "Carl", "Benjamin"]}
```

## Requirements

Python 3.6+ (no external dependencies).

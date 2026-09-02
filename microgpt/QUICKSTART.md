# microgpt Quickstart

A minimal GPT in pure Python (zero dependencies) that you can train, fine-tune, and
align end-to-end in a few minutes. This runs the full pipeline — **pretrain → LoRA SFT →
DPO/GRPO** — with live ASCII visualization (loss sparkline + attention heat map) at each stage.

Each stage feeds the next, so run them **in order**:

```bash
python3 microgpt.py --viz 250            # pretrain on names      -> model.json
python3 microgpt_sft.py --viz 250        # LoRA instruction SFT   -> model_sft.json
python3 microgpt_dpo.py --viz 250 -n 1000  # DPO preference align -> model_dpo.json
```

What the flags do:
- `--viz 250` — print a loss sparkline + attention matrix snapshot every 250 steps, plus
  end-of-run charts (and, for `microgpt.py`, a character-embedding similarity map).
- `-n 1000` — cap DPO to 1000 training steps (the full run is ~4200; `-n` keeps it short).
- `-d FILE` — override the default training data file for any script.

Notes:
- Requires only **Python 3.6+** — no external packages.
- Each run appends its console output to `train.log` (the live per-step counter stays on
  your terminal but is kept out of the log).
- Run inference only from a saved model with `-i`, e.g. `python3 microgpt.py -i`.
- See all options with `-h`, e.g. `python3 microgpt_dpo.py -h`.

For full details, see [README.md](README.md).

---

## Task Examples

The default pipeline trains on names. Below are alternative tasks that demonstrate
learnable patterns — the model can actually generalize, not just memorize.

### Single-Digit Addition

The model learns `a+b` → `sum`. 100 pairs, fits easily in context, verifiable by hand.

```bash
# Step 1: Pretrain on addition strings (learns digit/operator character patterns)
python3 microgpt.py -d input_sft_add1.txt --viz 250

# Step 2: SFT on the completion task (learns a+b → answer)
python3 microgpt_sft.py -d input_sft_add1.txt --viz 250

# Step 3: GRPO alignment (reinforce correct sums, suppress wrong ones)
python3 microgpt_grpo.py --offline input_grpo_add1.jsonl --viz 250
```

### Double-Digit Addition

Harder — the model must learn carrying. 210 pairs, answers up to 3 digits.

```bash
# Step 1: Pretrain
python3 microgpt.py -d input_sft_add2.txt --viz 250

# Step 2: SFT
python3 microgpt_sft.py -d input_sft_add2.txt --viz 250

# Step 3: GRPO
python3 microgpt_grpo.py --offline input_grpo_add2.jsonl --viz 250
```

### Pluralization Rules

The model learns English plural rules: `+s`, `+es`, `y→ies`, `f→ves`, irregulars.

```bash
# Step 1: Pretrain
python3 microgpt.py -d input_sft_plural.txt --viz 250

# Step 2: SFT
python3 microgpt_sft.py -d input_sft_plural.txt --viz 250

# Step 3: GRPO
python3 microgpt_grpo.py --offline input_grpo_plural.jsonl --viz 250
```

### Pig Latin

Character-level string transformation: move consonant cluster to end + `ay`, or add `way`.

```bash
# Step 1: Pretrain
python3 microgpt.py -d input_sft_piglatin.txt --viz 250

# Step 2: SFT
python3 microgpt_sft.py -d input_sft_piglatin.txt --viz 250

# Step 3: GRPO
python3 microgpt_grpo.py --offline input_grpo_piglatin.jsonl --viz 250
```

### Airport Codes (Memorization)

Pure lookup task — `US, Denver` → `DEN`. 96 major international airports.
Demonstrates memorization limits of tiny models.

```bash
# Step 1: Pretrain on raw airport lines (6,072 entries, learns format)
python3 microgpt.py -d airport_codes.txt --viz 250

# Step 2: SFT on country+city → code
python3 microgpt_sft.py -d input_airports_sft.txt --viz 250

# Step 3: GRPO alignment
python3 microgpt_grpo.py --offline input_grpo_airports.jsonl --viz 250
```

# microgpt Quickstart

A minimal GPT in pure Python (zero dependencies) that you can train, fine-tune, and
align end-to-end in a few minutes. This runs the full pipeline — **pretrain → LoRA SFT →
DPO** — with live ASCII visualization (loss sparkline + attention heat map) at each stage.

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

Notes:
- Requires only **Python 3.6+** — no external packages.
- Each run appends its console output to `train.log` (the live per-step counter stays on
  your terminal but is kept out of the log).
- Run inference only from a saved model with `-i`, e.g. `python3 microgpt.py -i`.
- See all options with `-h`, e.g. `python3 microgpt_dpo.py -h`.

For full details, see [README.md](README.md).

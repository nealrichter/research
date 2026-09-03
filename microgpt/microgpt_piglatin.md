# Pig Latin with microgpt — Teaching Notes

A worked example of using `microgpt` to learn a **character-level string
transformation** (Pig Latin) instead of just generating names. It's a good
teaching vehicle because the rule is simple, deterministic, and verifiable by
hand — yet it exposes the real limits of a tiny model.

## The task

Pig Latin has two rules:

1. **Consonant-start** → move the leading consonant cluster to the end, append `ay`
   - `hello` → `ellohay`, `string` → `ingstray`
2. **Vowel-start** → append `way`
   - `apple` → `appleway`, `egg` → `eggway`

This is a *copy-and-permute* task: the output reuses the input's characters in a
reordered form. That property is what makes it hard for a small model (more on
that below).

## Files

| File | Format | Role |
|------|--------|------|
| `input_piglatin_tiny.txt` | `bat atbay` | Pretraining, PAIRED (pretrain solves task) |
| `input_piglatin_tiny_pretrain.txt` | `bat` / `atbay` | Pretraining, words-only UNPAIRED (enables real SFT demo) |
| `input_piglatin_tiny_sft.txt` | `bat\|atbay` | SFT (tiny balanced set) |
| `input_piglatin_tiny_test.txt` | `cat` | Held-out eval words (tiny balanced) |
| `input_piglatin_imbal.txt` | `bat atbay` | Pretraining (imbalanced set) |
| `input_piglatin_imbal_sft.txt` | `bat\|atbay` | SFT (imbalanced set) |
| `input_piglatin_imbal_test.txt` | `at` | Held-out eval words (imbalanced) |
| `input_piglatin.txt` | `zebra ebrazay` | Pretraining (larger dictionary set) |
| `input_piglatin_sft.txt` | `zebra\|ebrazay` | SFT (larger set) |
| `input_piglatin_test.txt` | `bloomy` | Held-out eval words (larger) |

Generators:
- `make_piglatin_tiny.py` → **balanced** tiny set. Regular structure for BOTH
  rules (consonants `onset×rime`, vowels `vowel-onset×body`), balanced class
  sizes. Both rules generalize to held-out words at ~100%.
- `make_piglatin_tiny_imbalanced.py` → **imbalanced** tiny set. ~120 regular
  consonant words vs ~19 varied vowel words. Learns the consonant rule perfectly
  but fails the vowel-stem copy (0% exact / ~0.30 CER on vowels).
- The larger set was built from the system dictionary (`/usr/share/dict/words`),
  3–5 letter words only, transformed with the same `pig_latin()` rule.

All datasets and the `make_*.py` generators live under / write to `data/`. The
training scripts' built-in defaults point at the current directory, so pass
datasets explicitly, e.g. `-d data/input_piglatin_tiny.txt`.

## How to run

### Tiny set — pretraining solves it (SFT redundant)

```bash
# 1. Generate the data
python3 make_piglatin_tiny.py

# 2. Pretrain on PAIRED 'word pigword' lines
python3 microgpt.py -d data/input_piglatin_tiny.txt --viz 250

# 3. Evaluate on held-out words (feeds each word, scores completion vs the rule)
python3 microgpt_piglatin_eval.py model.json data/input_piglatin_tiny_test.txt
```

Expected: both `vowel-start (+way)` and `consonant-start (+ay)` rows near 100%.
Pretraining alone already learns the transformation (see "How pretraining learns
the mapping" below), so a subsequent SFT step is redundant — BEFORE-SFT and
AFTER-SFT outputs are identical.

### Tiny set — a MEANINGFUL SFT demo (create a capability gap first)

To make SFT actually teach something, pretrain on the **words-only, unpaired**
corpus so the base model learns the vocabulary but NOT the input→output mapping:

```bash
python3 make_piglatin_tiny.py

# Pretrain on UNPAIRED words -> base model learns letters, not the mapping
python3 microgpt.py -d data/input_piglatin_tiny_pretrain.txt --viz 250

# SFT then teaches the word->piglatin transformation
python3 microgpt_sft.py -d data/input_piglatin_tiny_sft.txt --viz 1000

# Eval the SFT model
python3 microgpt_piglatin_eval.py model_sft.json data/input_piglatin_tiny_test.txt
```

Now BEFORE-SFT is wrong (the base model emits generic `ay`-ending noise because
it only saw pig-latin *outputs*, never the mapping), and AFTER-SFT is correct.
SFT visibly adds a capability the base model lacked.

### Imbalanced set (demonstrates class imbalance + the copy bottleneck)

```bash
python3 make_piglatin_tiny_imbalanced.py
python3 microgpt.py -d data/input_piglatin_imbal.txt --viz 250
python3 microgpt_piglatin_eval.py model.json data/input_piglatin_imbal_test.txt
```

Expected: `consonant-start (+ay)` ~100% exact / CER 0.000, but
`vowel-start (+way)` ~0% exact / CER ~0.30. Note that *every* vowel prediction
still ends in `way` — the model learned the rule, it just can't copy the stem of
the undertrained, less-regular vowel class (e.g. `at → apway`, `and → antway`).

### Larger set (demonstrates the failure mode)

```bash
python3 microgpt.py -d data/input_piglatin.txt --viz 250
python3 microgpt_piglatin_eval.py model.json data/input_piglatin_test.txt
```

The eval's model file argument is optional (defaults to `model.json`), so this
also works:

```bash
python3 microgpt_piglatin_eval.py data/input_piglatin_test.txt
```

## Reading the eval output

`microgpt_piglatin_eval.py` reports two metrics, split by rule:

- **Exact match** — did it produce the exact correct Pig Latin string? (harsh, binary)
- **Char error rate (CER)** — Levenshtein distance / length (0 = perfect, 1 = all wrong).
  This is the useful one: it reveals *partial* learning that exact-match hides.

A model can score 0% exact while sitting at ~0.5 CER — meaning it learned the
*shape* of the transformation (move a letter, append `ay`) but can't reproduce
the stem precisely.

## Why it will / won't work

### Why the tiny set can work

- **Short words (2–4 chars):** the whole `word pigword` line fits well inside the
  16-token context, so nothing is truncated.
- **Massive shared structure:** the set is built from a small grid of consonant
  onsets × rime bodies (`bat, cat, hat, mat, …` all share `-at`). The model sees
  dozens of near-identical examples that differ only in the onset, so it can
  induce "move the single leading consonant, append `ay`" from repetition.
- **Few distinct patterns to memorize:** ~120 short words is well within the
  capacity of the model, and 1000 steps gives ~8 passes over the data.
- **Both rules are cleanly separated and simple:** the vowel words (`at → atway`)
  teach the `+way` branch with equally trivial examples.

### Why the larger set won't (with the stock model)

- **It's a copy-and-permute task.** A 1-layer, 16-dim transformer has no explicit
  copy mechanism. Reproducing an arbitrary word's exact characters in reordered
  form is exactly the kind of operation small autoregressive models struggle with.
- **Too much variety for the parameter budget.** ~3000 diverse dictionary words
  over a 27-char vocab, encoded in ~4.2k parameters, is far past the point where
  the model can memorize stems. It learns the *shape* but not the content
  (observed: 0% exact, ~0.49 CER).
- **Undertraining.** With `num_steps = 1000` and 3000 docs, the model completes
  only ~1/3 of one epoch. It literally never sees most of the data even once at
  full weight. The loss trajectory bounces (e.g. `3.34 → 1.61 → 2.20 → 1.63`)
  rather than converging.
- **Rare/obscure words.** The raw dictionary is weighted toward uncommon words
  with unusual letter statistics, which makes the character distribution harder
  to model than natural text.

## Model & data limitations (the general lesson)

| Constraint | Value (stock `microgpt.py`) | Consequence for Pig Latin |
|------------|-----------------------------|---------------------------|
| Layers | 1 | No depth to compose "find onset" → "move it" → "emit" |
| Embedding dim | 16 | Little room to represent per-position copy state |
| Attention heads | 4 | Limited routing for copy/permute |
| Context length | 16 | Fine for short words; truncates long ones |
| Params | ~4.2k | Enough to memorize a small structured set, not 1000s of stems |
| Training steps | 1000 | ~8 epochs on ~120 words; <1 epoch on 3000 words |

**Levers to make the hard version work** (in rough order of impact):

1. **Shrink and structure the data** — the tiny set above. Repetition beats volume.
2. **Train longer** — bump `num_steps` so the model sees several full epochs
   (for 3000 docs, aim for 6000–9000+ steps).
3. **Add capacity** — `n_layer = 2–3`, `n_embd = 32–64`. Copy/permute tasks
   benefit disproportionately from depth and width.
4. **Raise `block_size`** if you keep longer words, so nothing is truncated.

## How pretraining learns the mapping (no special machinery)

Nothing special makes pretraining learn `word → pigword`. It falls out of plain
**next-character prediction** over the paired lines. Each line
`bat atbay` is tokenized `[BOS] b a t ␣ a t b a y [BOS]` and the model is trained
to predict every next character. At the space, the objective is literally
"given `bat `, predict the next char" — so over many examples the model learns
"word + space → emit that word's pig-latin form." **The space acts as an
implicit `→` operator**, and at inference we feed `[BOS] b a t ␣` and let the
model continue.

This is the same phenomenon that lets real LLMs pick up tasks during
pretraining: the web is full of `question … answer`, `English: X French: Y`,
`def f(): … return` patterns, and next-token prediction absorbs those mappings
with no explicit supervision. Here it works cleanly because (a) the separator is
in a consistent position, (b) the pairs share structure so the model learns the
*rule* not the pairs, and (c) the whole pair fits inside the 16-token context so
the output positions can attend back to the input characters.

## Pretrain-on-pairs vs pretrain-then-SFT (the real lesson)

Two ways to reach a working model, with very different efficiency:

| Approach | What happens | Result |
|----------|-------------|--------|
| **Pretrain on paired `word pigword`** | Full (unmasked) next-token loss; every character gives gradient, incl. the mapping | ~100% in ~1000 steps. SFT afterward is redundant (BEFORE == AFTER). |
| **Pretrain on words-only, then SFT** | Base model learns vocabulary but not the mapping; SFT teaches it with a **response-only (masked)** loss on ~256 LoRA params | Learns, but slower/weaker — many more steps, less gradient signal per step. |

### Measured results (tiny balanced set, 1-layer / 16-dim / ~4k params)

| Regime | Steps | Overall exact | Consonant (+ay) | Vowel (+way) | Loss path |
|--------|-------|---------------|-----------------|--------------|-----------|
| Paired pretrain | ~1000 | **32/32 = 100%** | 18/18 (100%) | 14/14 (100%) | smooth `3.3 → 0.5` |
| Words-only pretrain → SFT | 7200 | **~24/32 ≈ 75%** | 15/18 (~83%) | 9/14 (~64%) | noisy `6.24 → 0.42`, oscillating |

So SFT took ~7× the steps, followed a noisier loss path, and still landed ~25
points *below* the paired-pretrain baseline. It genuinely learned the task from a
base model that scored 0% (BEFORE-SFT emitted a degenerate `ay` for every input),
but it is clearly the weaker route here.

Every SFT error was a **stem-copy corruption, not a rule failure** — the suffix
and move-onset structure were right, but arbitrary characters got mangled:

```
rag -> agfay   (expected agray)   # moved r, +ay correct; stem consonant r->f
rug -> ogbay   (expected ugray)   # right _g_ay shape, two chars wrong
ega -> enaway  (expected egaway)  # +way correct; stem eg->en
ita -> inaway  (expected itaway)  # +way correct; stem it->in
```

This is the **same copy bottleneck** seen in the imbalanced and large-dataset
runs, now appearing a third time under a masked/low-parameter objective. Note the
errors drift toward high-frequency characters (`n`, `a`) when the copy circuit is
uncertain — classic low-capacity fallback.

The key observation: **for a task you fully control at toy scale, pretraining on
paired data is strictly more efficient than pretrain-then-SFT.** SFT is the
harder, slower path here because:

- It **masks the loss to response tokens only**, discarding gradient from the
  prompt that full pretraining uses.
- It updates just the **256 LoRA parameters**, not the full ~4k.
- It's deliberately built on a **handicapped base model** (words-only pretrain).

So why does SFT exist at all? Because in the real world you **cannot re-pretrain
a large model for every task.** SFT earns its keep when (a) the task can't be
baked into the pretraining corpus, or (b) you have a general pretrained model and
want to cheaply specialize it. It is *not* a better learner than pretraining —
it's a *cheaper* way to add a capability when re-pretraining is off the table.
At this scale, where you can just pretrain on pairs, doing so wins.

## Takeaway for students

Pig Latin is a clean demonstration that **task structure interacts with model
capacity**. Three regimes, same architecture:

1. **Balanced tiny set** → both rules generalize to held-out words at ~100%.
   Regular structure + enough examples per class = the model learns *and* copies.
2. **Imbalanced tiny set** → the consonant rule is perfect, but the vowel rule
   learns the `+way` suffix yet fails the stem copy (0% exact / ~0.30 CER). This
   isolates two distinct skills — **learning a rule** vs **executing a copy** —
   and shows how **class imbalance** starves the minority class.
3. **Large dictionary set** → both rules fail the copy. Too much stem variety
   for 4k parameters and a single layer; the model learns the *shape* of the
   transformation but not the content.

The through-line: copying arbitrary characters through a permutation is the hard
part for a tiny model, and it degrades gracefully as data gets less regular or
less balanced. The **char error rate (CER)** is what lets you *see* this partial
learning that a pass/fail (exact-match) metric would completely hide.

And a second, orthogonal lesson from the same task: **pretraining vs SFT is
about *when* you can bake the task into the corpus, not about which algorithm
learns better.** Pretraining on paired data solves this task fastest; SFT is the
slower, weaker path that only earns its keep when re-pretraining isn't an option.
Seeing BEFORE-SFT == AFTER-SFT (paired pretrain) vs BEFORE-SFT wrong / AFTER-SFT
right (words-only pretrain) makes that tradeoff concrete.

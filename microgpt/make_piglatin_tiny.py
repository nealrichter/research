"""
Generate a tiny, highly-repetitive Pig Latin dataset for microgpt (BALANCED).

Design goal: maximize learnability for a 1-layer, 16-dim model by using
- short words (2-4 letters)
- lots of SHARED structure for BOTH rules:
    consonant-start: single-consonant onset + shared rime body (bat, cat, hat, ...)
    vowel-start:     vowel onset + shared body (aba, ada, ata, ...)
- balanced class sizes so neither rule is undertrained.

This gives the model many near-identical examples so it can generalize the
two rules:  consonant-start -> move onset to end + 'ay';  vowel-start -> + 'way'.
Expected result: both rules generalize to held-out words at ~100%.

For the deliberately UNBALANCED / harder variant (which learns the consonant
rule but fails the vowel copy), see make_piglatin_tiny_imbalanced.py.

## Two ways to pretrain (this matters for the SFT demo)

- input_piglatin_tiny.txt  — PAIRED 'word pigword' lines. Pretraining on this
  ALREADY solves the transformation, so a subsequent SFT step is redundant
  (BEFORE-SFT == AFTER-SFT). Use this to show pretraining alone learning a rule.

- input_piglatin_tiny_pretrain.txt — words-only, UNPAIRED lines (inputs and
  pig-latin forms shuffled, never shown as a pair). Pretraining on this teaches
  the model the character vocabulary but NOT the input->output mapping. THEN an
  SFT step genuinely teaches the mapping: BEFORE-SFT is word-like noise,
  AFTER-SFT is correct Pig Latin. Use this for a meaningful SFT demonstration.

Writes:
  input_piglatin_tiny.txt           paired 'word pigword' (pretrain solves task)
  input_piglatin_tiny_pretrain.txt  words-only unpaired   (SFT then teaches mapping)
  input_piglatin_tiny_sft.txt       'word|pigword' pairs  (SFT)
  input_piglatin_tiny_test.txt      held-out words        (eval)

Run:  python3 make_piglatin_tiny.py
"""

import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
def data_path(name):
    return os.path.join(DATA_DIR, name)

def pig_latin(word):
    vowels = "aeiou"
    w = word.lower()
    if w[0] in vowels:
        return w + "way"
    i = 0
    while i < len(w) and w[i] not in vowels:
        i += 1
    return w[i:] + w[:i] + "ay"

# Highly regular structure: single-consonant onsets + shared rime bodies.
onsets = ["b", "c", "d", "f", "h", "m", "n", "p", "r", "s", "t"]
rimes  = ["at", "an", "ap", "ag", "ed", "en", "ig", "op", "ot", "un", "ug"]

# Vowel-start words (the +way rule). Built with the SAME regular structure as
# the consonant set — a vowel onset + a shared 2-letter body — so the model
# gets equally repetitive, copy-friendly examples for the +way branch.
vowel_onsets = ["a", "e", "i", "o", "u"]
vbodies = ["ba", "da", "ta", "na", "ma", "pa", "ra", "sa", "ka", "ga", "la"]
vowel_words = [vo + b for vo in vowel_onsets for b in vbodies]  # e.g. aba, ada, ...

# --- Train/test split ---
# Consonant combos: hold out every 7th for testing.
all_combos = [(o, r) for o in onsets for r in rimes]
cons_test = [o + r for i, (o, r) in enumerate(all_combos) if i % 7 == 0]
cons_train = [o + r for (o, r) in all_combos if (o + r) not in set(cons_test)]

# Vowel words: hold out every 4th for testing so the +way rule is also evaluated.
vowel_test = [w for i, w in enumerate(vowel_words) if i % 4 == 0]
vowel_train = [w for w in vowel_words if w not in set(vowel_test)]

train_words = cons_train + vowel_train
test_words = cons_test + vowel_test

# De-dup, keep order
def dedup(seq):
    seen, out = set(), []
    for w in seq:
        if w not in seen:
            seen.add(w); out.append(w)
    return out

train_words = dedup(train_words)
test_words = dedup([w for w in test_words if w not in set(train_words)])

with open(data_path("input_piglatin_tiny.txt"), "w") as f:
    for w in train_words:
        f.write(f"{w} {pig_latin(w)}\n")

with open(data_path("input_piglatin_tiny_sft.txt"), "w") as f:
    for w in train_words:
        f.write(f"{w}|{pig_latin(w)}\n")

with open(data_path("input_piglatin_tiny_test.txt"), "w") as f:
    for w in test_words:
        f.write(f"{w}\n")

# --- Words-only pretraining corpus (for a MEANINGFUL SFT demo) ---
# Contains the bare input words AND the pig-latin forms, but as SEPARATE,
# UNPAIRED lines. The model learns the character statistics of both vocabularies
# but never sees the input->output MAPPING. That mapping is what SFT then teaches
# -- so BEFORE-SFT output is word-like noise and AFTER-SFT output is correct.
# (Contrast with input_piglatin_tiny.txt, which pairs them and lets pretraining
#  already solve the task, making SFT redundant.)
pretrain_lines = []
for w in train_words:
    pretrain_lines.append(w)               # the input word
    pretrain_lines.append(pig_latin(w))    # the pig-latin form, unpaired
# shuffle so pairs aren't adjacent (avoid leaking the mapping via ordering)
import random as _r
_r.seed(0)
_r.shuffle(pretrain_lines)
with open(data_path("input_piglatin_tiny_pretrain.txt"), "w") as f:
    for line in pretrain_lines:
        f.write(line + "\n")

mx = max(len(f"{w} {pig_latin(w)}") for w in train_words)
print(f"train: {len(train_words)} words, test: {len(test_words)} words (disjoint)")
print(f"max pretrain line length: {mx} chars (context=16 ok)")
print(f"wrote to {DATA_DIR}/:")
print("  input_piglatin_tiny.txt          paired 'word pigword'  (pretrain solves task; SFT redundant)")
print("  input_piglatin_tiny_pretrain.txt words-only, unpaired    (SFT then teaches the mapping)")
print("  input_piglatin_tiny_sft.txt      'word|pigword'          (SFT / eval)")
print("  input_piglatin_tiny_test.txt     held-out words          (eval)")
print("samples (paired):")
for w in train_words[:8]:
    print(f"  {w} {pig_latin(w)}")

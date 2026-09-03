"""
Generate a tiny Pig Latin dataset for microgpt (IMBALANCED / harder variant).

This is the DELIBERATELY UNBALANCED counterpart to make_piglatin_tiny.py. It is
designed to fail on the vowel-start rule so you can teach two lessons at once:

  1. Class imbalance: ~120 highly-regular consonant words vs only ~19 varied
     vowel words. The consonant copy-path is well trained; the vowel one isn't.
  2. The copy bottleneck: the vowel words are a grab-bag (and, opt, off, own)
     with little shared structure, so the model can't reliably COPY their stems
     even though it clearly learns the '+way' suffix rule.

Observed result (1-layer, 16-dim, 1000 steps):
  consonant-start (+ay)   100% exact, CER 0.000   (rule learned AND copied)
  vowel-start   (+way)      0% exact, CER ~0.30    (rule learned, copy fails)
Every vowel prediction ends in 'way' -- the RULE is right -- but the stem is
mangled (at -> apway, and -> antway). The CER metric makes this visible; a
pass/fail metric would just say "0%".

For the balanced version that learns BOTH rules at ~100%, see
make_piglatin_tiny.py.

Writes:
  input_piglatin_imbal.txt        'word pigword' pairs (pretraining)
  input_piglatin_imbal_sft.txt    'word|pigword' pairs (SFT)
  input_piglatin_imbal_test.txt   held-out words (both rules)

Run:  python3 make_piglatin_tiny_imbalanced.py
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

# Consonant set: highly regular (onset x rime) -> lots of repetition, easy copy.
onsets = ["b", "c", "d", "f", "h", "m", "n", "p", "r", "s", "t"]
rimes  = ["at", "an", "ap", "ag", "ed", "en", "ig", "op", "ot", "un", "ug"]

# Vowel set: a small, VARIED grab-bag (little shared structure) -> undertrained
# and hard to copy. This is what breaks the vowel branch.
vowel_words = [
    "at", "an", "ap", "ag", "ed", "en", "ig", "op", "ot", "un", "ug",
    "ant", "and", "end", "elf", "imp", "opt", "urn", "add", "ill", "off",
    "art", "eat", "ink", "own", "use",
]

# --- Train/test split ---
all_combos = [(o, r) for o in onsets for r in rimes]
cons_test = [o + r for i, (o, r) in enumerate(all_combos) if i % 7 == 0]
cons_train = [o + r for (o, r) in all_combos if (o + r) not in set(cons_test)]

vowel_test = [w for i, w in enumerate(vowel_words) if i % 4 == 0]
vowel_train = [w for w in vowel_words if w not in set(vowel_test)]

train_words = cons_train + vowel_train
test_words = cons_test + vowel_test

def dedup(seq):
    seen, out = set(), []
    for w in seq:
        if w not in seen:
            seen.add(w); out.append(w)
    return out

train_words = dedup(train_words)
test_words = dedup([w for w in test_words if w not in set(train_words)])

with open(data_path("input_piglatin_imbal.txt"), "w") as f:
    for w in train_words:
        f.write(f"{w} {pig_latin(w)}\n")

with open(data_path("input_piglatin_imbal_sft.txt"), "w") as f:
    for w in train_words:
        f.write(f"{w}|{pig_latin(w)}\n")

with open(data_path("input_piglatin_imbal_test.txt"), "w") as f:
    for w in test_words:
        f.write(f"{w}\n")

n_vowel_train = sum(1 for w in train_words if w[0] in "aeiou")
n_cons_train = len(train_words) - n_vowel_train
mx = max(len(f"{w} {pig_latin(w)}") for w in train_words)
print(f"train: {len(train_words)} words ({n_cons_train} consonant, {n_vowel_train} vowel)")
print(f"test:  {len(test_words)} words (disjoint)")
print(f"max pretrain line length: {mx} chars (context=16 ok)")
print("NOTE: vowel class is intentionally small/varied -> expect the +way branch to fail the copy.")

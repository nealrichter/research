"""
Pig Latin error function for microgpt.

Feeds each real word from a file as a prompt, lets the model complete it,
and scores the completion against the true Pig Latin rule. Use a held-out
word list (words never seen in training) to measure generalization.

Usage:
  python3 microgpt_piglatin_eval.py [MODEL.json] WORDS.txt
  python3 microgpt_piglatin_eval.py model.json data/input_piglatin_test.txt
"""

import json, math, random, sys

if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 2:
    print("usage: python3 microgpt_piglatin_eval.py [MODEL.json] WORDS.txt\n")
    print("Feed each word from WORDS.txt as a prompt, score completion vs Pig Latin rule.")
    sys.exit(0)

# Parse args: optional MODEL.json (ends in .json), required WORDS.txt
model_file = 'model.json'
words_file = None
for arg in sys.argv[1:]:
    if arg.endswith('.json'):
        model_file = arg
    else:
        words_file = arg
if words_file is None:
    print("error: provide a words file, e.g. data/input_piglatin_test.txt")
    sys.exit(1)

# --- The ground-truth Pig Latin rule ---
def pig_latin(word):
    """Consonant cluster -> end + 'ay'; vowel start -> + 'way'."""
    vowels = 'aeiou'
    w = word.lower()
    if not w:
        return ''
    if w[0] in vowels:
        return w + 'way'
    i = 0
    while i < len(w) and w[i] not in vowels:
        i += 1
    return w[i:] + w[:i] + 'ay'

# --- Load model ---
with open(model_file) as f:
    model = json.load(f)

uchars = model['vocab']
cfg = model['config']
n_layer, n_embd, n_head = cfg['n_layer'], cfg['n_embd'], cfg['n_head']
block_size = cfg['block_size']
head_dim = n_embd // n_head
vocab_size = len(model['weights']['wte'])
weights = model['weights']

# The model declares its prompt format AND its special-token ids in the saved
# JSON (written by the training scripts). We read them directly rather than
# guessing/recomputing.
#   'sft'      -> prompt is [BOS] + word + [SEP], answer follows.
#   'pretrain' -> free-form 'word pigword'; no SEP, space separates the pair.
model_format = model.get('format')
if model_format is None:
    # Old file without metadata: fall back to legacy conventions and warn.
    model_format = 'pretrain'
    print("warning: model has no 'format' field (old file?); assuming 'pretrain'. "
          "Retrain to embed format/bos/sep metadata.")

BOS = model.get('bos', len(uchars))          # default matches legacy convention
SEP = model.get('sep')                        # None for pretrain
SPACE = uchars.index(' ') if ' ' in uchars else None
print(f"model format: {model_format}  (bos={BOS}, sep={SEP})")

# --- Forward pass (plain floats, no grad) ---
def linear(x, w):
    return [sum(wo[j] * x[j] for j in range(len(x))) for wo in w]

def softmax(logits):
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    tot = sum(exps)
    return [e / tot for e in exps]

def rmsnorm(x):
    scale = (sum(xi * xi for xi in x) / len(x) + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    x = [t + p for t, p in zip(weights['wte'][token_id], weights['wpe'][pos_id])]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)
        q = linear(x, weights[f'layer{li}.attn_wq'])
        k = linear(x, weights[f'layer{li}.attn_wk'])
        v = linear(x, weights[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                          for t in range(len(k_h))]
            aw = softmax(attn_logits)
            x_attn.extend([sum(aw[t] * v_h[t][j] for t in range(len(v_h)))
                          for j in range(head_dim)])
        x = [a + b for a, b in zip(linear(x_attn, weights[f'layer{li}.attn_wo']), x_res)]
        x_res = x
        x = [max(0, xi) for xi in linear(rmsnorm(x), weights[f'layer{li}.mlp_fc1'])]
        x = [a + b for a, b in zip(linear(x, weights[f'layer{li}.mlp_fc2']), x_res)]
    return linear(x, weights['lm_head'])

# --- Prompted generation: feed the word, complete its Pig Latin ---
def complete(word, temperature=0.5, max_len=10):
    """
    Feed `word` as a prompt and generate the completion, using the prompt
    format the model declared:
      sft      -> [BOS] word [SEP]  ...answer...   (stop on BOS/SEP)
      pretrain -> [BOS] word SPACE  ...answer...   (stop on BOS/SPACE)
    """
    prompt = [BOS] + [uchars.index(c) for c in word if c in uchars]
    if SEP is not None:
        prompt.append(SEP)          # SFT delimiter
    elif SPACE is not None:
        prompt.append(SPACE)        # pretrain 'word pigword' separator
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    for pos_id, tid in enumerate(prompt[:-1]):
        gpt(tid, pos_id, keys, values)
    tid, pos_id = prompt[-1], len(prompt) - 1
    out = []
    for _ in range(max_len):
        logits = gpt(tid, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        tid = random.choices(range(vocab_size), weights=probs)[0]
        if tid == BOS or tid == SEP or tid == SPACE:
            break
        if tid < len(uchars):
            out.append(uchars[tid])
        pos_id += 1
        if pos_id >= block_size - 1:
            break
    return ''.join(out)

# --- Character error rate (Levenshtein / max length) ---
def char_error_rate(a, b):
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    return prev[n] / max(m, n)

# --- Evaluate ---
# Pig Latin is two-fold: vowel-start words (+way) vs consonant-start words (move cluster + ay).
# Score each case separately so we can see if the model learned both rules.
random.seed()
test_words = [w.strip() for w in open(words_file) if w.strip().isalpha()]

def is_vowel_start(w):
    return w.lower()[0] in 'aeiou'

# Counters per rule
stats = {'vowel': [0, 0, 0.0], 'consonant': [0, 0, 0.0]}  # [count, exact, cer_sum]
examples = []

for word in test_words:
    produced = complete(word)
    expected = pig_latin(word)
    cer = char_error_rate(produced, expected)
    key = 'vowel' if is_vowel_start(word) else 'consonant'
    stats[key][0] += 1
    stats[key][1] += (produced == expected)
    stats[key][2] += cer
    if len(examples) < 25:
        examples.append((word, expected, produced))

n = len(test_words)
exact = stats['vowel'][1] + stats['consonant'][1]
total_cer = stats['vowel'][2] + stats['consonant'][2]

print(f"Model: {model_file}")
print(f"Words: {words_file} ({n} words)")
print()
print(f"  Overall exact match:    {exact}/{n} ({100*exact/n:.1f}%)")
print(f"  Overall char err rate:  {total_cer/n:.3f}  (0=perfect, 1=all wrong)")
print()
print("  By rule:")
for key, label in (('vowel', 'vowel-start (+way)'), ('consonant', 'consonant-start (+ay)')):
    cnt, ex, cer = stats[key]
    if cnt:
        print(f"    {label:24s} {ex}/{cnt} exact ({100*ex/cnt:.1f}%), CER {cer/cnt:.3f}")
print()
print("Sample completions (word -> produced, expected):")
for word, expected, produced in examples:
    mark = "\u2713" if produced == expected else "\u2717"
    print(f"  {word} -> {produced}  (expected {expected}) {mark}")

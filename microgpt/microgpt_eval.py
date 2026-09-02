"""
Evaluate a trained microgpt model on an SFT-format dataset.
Runs inference on every prompt and scores against the expected completion.

Usage:
  python3 microgpt_eval.py MODEL.json DATA.txt
  python3 microgpt_eval.py model_sft.json input_sft_add1.txt
  python3 microgpt_eval.py model_grpo.json input_sft_add1.txt
"""

import json, math, random, sys

if len(sys.argv) < 3 or '-h' in sys.argv or '--help' in sys.argv:
    print("usage: python3 microgpt_eval.py MODEL.json DATA.txt\n")
    print("Evaluate model accuracy on every prompt|expected pair in DATA.txt")
    sys.exit(0)

model_file = sys.argv[1]
data_file = sys.argv[2]

# --- Load model ---
with open(model_file) as f:
    model = json.load(f)

uchars = model['vocab']
cfg = model['config']
n_layer, n_embd, n_head = cfg['n_layer'], cfg['n_embd'], cfg['n_head']
head_dim = n_embd // n_head
BOS = len(uchars)
SEP = len(uchars) + 1
vocab_size = len(uchars) + 2

weights = model['weights']

# Ensure SEP token row exists
for key in ('wte', 'lm_head'):
    if len(weights[key]) < vocab_size:
        weights[key].append([0.0] * n_embd)

block_size = len(weights['wpe'])

# --- Forward pass (plain floats, no grad) ---
def linear(x, w):
    return [sum(wo[j] * x[j] for j in range(len(x))) for wo in w]

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

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
            attn_weights = softmax(attn_logits)
            x_attn.extend([sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                          for j in range(head_dim)])
        x = [a + b for a, b in zip(linear(x_attn, weights[f'layer{li}.attn_wo']), x_res)]
        x_res = x
        x = [max(0, xi) for xi in linear(rmsnorm(x), weights[f'layer{li}.mlp_fc1'])]
        x = [a + b for a, b in zip(linear(x, weights[f'layer{li}.mlp_fc2']), x_res)]
    return linear(x, weights['lm_head'])

# --- Helpers ---
def encode(text):
    return [uchars.index(ch) for ch in text if ch in uchars]

def generate(prompt_str, max_len=20):
    """Generate a completion greedily (argmax, deterministic)."""
    prompt = [BOS] + encode(prompt_str) + [SEP]
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    for pos_id, tid in enumerate(prompt[:-1]):
        gpt(tid, pos_id, keys, values)
    tid, pos_id = prompt[-1], len(prompt) - 1
    out = []
    for _ in range(max_len):
        logits = gpt(tid, pos_id, keys, values)
        tid = logits.index(max(logits))  # greedy argmax
        if tid in (BOS, SEP):
            break
        if tid < len(uchars):
            out.append(uchars[tid])
        pos_id += 1
        if pos_id >= block_size - 1:
            break
    return ''.join(out)

# --- Load eval data ---
pairs = []
for line in open(data_file):
    if '|' in line:
        prompt, expected = line.strip().split('|', 1)
        pairs.append((prompt, expected))

# --- Evaluate ---
correct = 0
off_by_one = 0
total = len(pairs)
errors = []

for prompt, expected in pairs:
    got = generate(prompt, max_len=len(expected) + 5)
    if got == expected:
        correct += 1
    else:
        # Check off-by-one for numeric answers
        try:
            diff = int(got) - int(expected)
            if abs(diff) == 1:
                off_by_one += 1
        except ValueError:
            pass
        errors.append((prompt, expected, got))

# --- Report ---
print(f"Model: {model_file}")
print(f"Data:  {data_file} ({total} pairs)")
print(f"")
print(f"  Correct:    {correct}/{total} ({100*correct/total:.1f}%)")
if off_by_one > 0:
    print(f"  Off-by-one: {off_by_one}/{total} ({100*off_by_one/total:.1f}%)")
print(f"  Wrong:      {total - correct}/{total} ({100*(total-correct)/total:.1f}%)")
print()

# Show sample errors (up to 20)
if errors:
    print(f"Sample errors (showing {min(20, len(errors))}/{len(errors)}):")
    for prompt, expected, got in errors[:20]:
        print(f"  {prompt} -> expected '{expected}', got '{got}'")

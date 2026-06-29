"""
microgpt_viz: tiny, dependency-free ASCII visualization helpers for microgpt.

This module is kept separate from the core training scripts so the model code
(microgpt.py) stays small and readable for learning. Everything here is pure
Python standard library -- no numpy, no matplotlib.

It is organized in two layers:

1. Pure rendering helpers (no global state, easy to unit-test):
     - sparkline(history, height)      -> smoothed Unicode loss chart, 1..N rows tall
     - visualize_matrix(matrix_2d)     -> heat-mapped grid of scalars / Value nodes
     - visualize_heads(heads)          -> all heads' causal attention matrices, side by side
     - build_causal_matrix(record)     -> pad ragged attention rows into a square grid
     - embedding_similarity(wte, chars)-> cosine-similarity heat map of token embeddings

2. A stateful hook layer (module globals) so microgpt.py only needs a few one-line
   calls instead of threading visualization state through gpt() and the loops:
     - configure(argv)                 -> parse `--viz [N]` / `-i`, set `enabled`
     - attn(pos_id, head, weights)     -> cache head-0 attention during a forward pass
     - step(step_idx, num_steps, loss) -> per-training-step progress line + sparkline + dump
     - finish()                        -> end-of-run tall sparkline + attention matrix

Every hook is guarded in the core by `viz.enabled`, so when `--viz` is absent none
of this logic runs and the core behaves exactly as the original.

Design choices worth knowing:
- Loss is smoothed for *display only* (EMA + bucket-mean). The raw history is kept
  intact; we never fabricate the reported numbers.
- The sparkline is normalized over the whole history-so-far, so its top stays near
  the early (high) loss and bars shrink as loss falls -- this keeps periodic
  snapshots visually comparable across a run.
"""

# Eighth-block elements U+2581..U+2588 indexed 0..8 (index 0 == empty cell).
_EIGHTHS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


# --------------------------------------------------------------------------
# Pure rendering helpers
# --------------------------------------------------------------------------
def _ema(data, alpha=0.3):
    """Exponential moving average: smooths per-step noise while preserving the trend.

    `alpha` in (0, 1] is the weight of the newest point; lower = smoother.
    """
    out, acc = [], data[0]
    for v in data:
        acc = alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def _downsample_mean(data, width):
    """Reduce `data` to `width` columns by averaging each bucket.

    Averaging (rather than picking every Nth point) keeps the summary faithful and
    removes most of the visual noise. Returns `data` unchanged if it already fits.
    """
    if len(data) <= width:
        return list(data)
    out = []
    for i in range(width):
        a = int(i * len(data) / width)
        b = max(a + 1, int((i + 1) * len(data) / width))
        bucket = data[a:b]
        out.append(sum(bucket) / len(bucket))
    return out


def sparkline(loss_history, height=1, max_width=60):
    """Smoothed Unicode sparkline of a loss history, `height` character-rows tall.

    Args:
        loss_history: list of per-step loss floats.
        height:       number of text rows tall (1 = single inline line, 3 = a chart).
        max_width:    maximum number of columns (the series is bucket-mean downsampled
                      to this width).

    Returns:
        A list of `height` strings, top row first. (For height=1 take element [0].)

    The series is EMA-smoothed and bucket-mean downsampled, then normalized over the
    whole history so the chart's top stays near the early high loss and the bars fall
    as training improves. Vertical resolution is height*8 via the eighth-block glyphs.
    """
    if not loss_history:
        return [""] * height

    series = _ema(loss_history)                            # smooth the full history
    lo, hi = min(series), max(series)                      # stable scale over the run so far
    rng = (hi - lo) or 1.0
    data = _downsample_mean(series, max_width)             # reduce to display width
    levels = [(v - lo) / rng * (height * 8) for v in data]  # each value -> 0 .. height*8

    rows = []
    for r in range(height - 1, -1, -1):  # build top row down to bottom row
        base = r * 8                      # this row covers vertical band [base, base+8)
        row = ""
        for lv in levels:
            if lv >= base + 8:   row += _EIGHTHS[8]              # full cell
            elif lv <= base:     row += " "                     # empty cell
            else:                row += _EIGHTHS[int(lv - base)]  # partial cell
        rows.append(row)
    return rows


def embedding_similarity(embeddings, labels, name="token embedding cosine similarity"):
    """Heat map of cosine similarity between learned token embeddings.

    Answers "what did the model learn about tokens?" -- characters that appear in
    similar contexts drift toward similar embedding directions. Only the first
    len(labels) rows are shown (skips special tokens like BOS/SEP).

    Contrast is accentuated: the off-diagonal similarities are z-scored and clipped
    at +/-2 sigma, so even a narrow band of cosine values fills the glyph ramp and
    structure (pairs more similar than average) stands out instead of washing out.
    """
    V = len(labels)
    vecs = [[(e.data if hasattr(e, 'data') else e) for e in embeddings[i]] for i in range(V)]
    norms = [sum(x * x for x in v) ** 0.5 or 1.0 for v in vecs]
    sim = [[sum(vecs[i][k] * vecs[j][k] for k in range(len(vecs[i]))) / (norms[i] * norms[j])
            for j in range(V)] for i in range(V)]

    off = [sim[i][j] for i in range(V) for j in range(V) if i != j]  # exclude self-similarity (=1)
    mean = sum(off) / len(off)
    std = (sum((x - mean) ** 2 for x in off) / len(off)) ** 0.5 or 1.0

    ramp = " \u2591\u2592\u2593\u2588"  # ' ' ░ ▒ ▓ █  : least -> most similar (relative to average)
    def _glyph(z):
        t = max(-2.0, min(2.0, z))                      # clip to +/-2 sigma
        return ramp[int((t + 2) / 4 * (len(ramp) - 1) + 0.5)]

    print(f"\n--- {name} [{V}x{V}] (z-scored, \u00b12\u03c3) ---")
    print("     " + " ".join(labels))
    for i in range(V):
        cells = " ".join("\u2588" if i == j else _glyph((sim[i][j] - mean) / std) for j in range(V))
        print(f"  {labels[i]}  {cells}")
    print("  legend: ' ' least  \u2591\u2592\u2593 below\u2192above avg  \u2588 most (diagonal = self)")


def build_causal_matrix(attn_record):
    """Pad ragged attention rows (query t attends to keys 0..t) into a square T x T grid.

    Rows may contain Value objects or floats. The future (causal-masked) positions are
    padded with None -- a sentinel meaning "never computed" -- so visualize_matrix can
    render them distinctly from genuinely small attention weights.
    """
    if not attn_record:
        return []
    T = len(attn_record)
    return [list(row) + [None] * (T - len(row)) for row in attn_record]  # None = masked / not computed


def _cell_glyph(cell):
    """Glyph for one heat-map cell: hollow box for masked (None), else density by value."""
    if cell is None:
        return "\u25a1"                       # □ masked / not computed
    v = cell.data if hasattr(cell, 'data') else cell
    if v > 0.5:   return "\u2588"             # █ high
    if v > 0.1:   return "\u2593"             # ▓ medium
    if v > -0.1:  return "\u2591"             # ░ low / near-zero
    return "."                                # negative


def visualize_heads(heads, name="Attention (causal)"):
    """Render every head's causal attention matrix side by side (horizontally).

    `heads` maps head-index -> ragged attention rows (as collected by attn()). Each is
    padded to a square T x T grid and the heads are laid out left-to-right so you can
    compare how they specialize (diagonal / previous-token / first-token / diffuse).
    """
    if not heads:
        return
    order = sorted(heads)
    mats = [build_causal_matrix(heads[h]) for h in order]
    if not mats[0]:
        return
    T = len(mats[0])
    block = 2 * T          # each row renders as T cells of "glyph + space" = 2*T chars
    gap = "   "
    print(f"\n--- {name} | {len(order)} heads, each [{T}x{T}] ---")
    print(gap.join(f"head {h}".ljust(block) for h in order))
    for r in range(T):
        print(gap.join("".join(_cell_glyph(m[r][c]) + " " for c in range(T)) for m in mats))


def visualize_matrix(matrix_2d, name="Matrix"):
    """Render a scannable, text-based heat map of a 2D list of Value objects (or floats).

    Density glyphs by cell value: > 0.5 = full, > 0.1 = medium, > -0.1 = light, else dot.
    A cell of None renders as a hollow box (□) meaning "masked / not computed" -- distinct
    from a genuinely small weight (░), e.g. the causal upper-triangle of an attention map.
    """
    if not matrix_2d or not isinstance(matrix_2d[0], list):
        return
    rows, cols = len(matrix_2d), len(matrix_2d[0])
    print(f"\n--- {name} [{rows}x{cols}] ---")
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            cell = matrix_2d[r][c]
            if cell is None:                         # masked / never computed (causal future)
                row_str += "\u25a1 "                 # □ hollow box, distinct from weak ░
                continue
            val = cell.data if hasattr(cell, 'data') else cell  # unwrap raw scalar from Value node
            if val > 0.5:     row_str += "\u2588 "  # highly active
            elif val > 0.1:   row_str += "\u2593 "  # moderately active
            elif val > -0.1:  row_str += "\u2591 "  # neutral / near-zero
            else:             row_str += ". "        # negative / inactive
        print(row_str)


# --------------------------------------------------------------------------
# Stateful hook layer: all visualization state lives here in module globals,
# so the core script only sprinkles in a handful of one-line calls, each guarded
# by `viz.enabled` -- when --viz is absent, none of this logic runs.
# --------------------------------------------------------------------------
enabled = False          # was --viz passed? (the core guards every hook on this)
_n = 0                   # snapshot interval (0 = in-situ sparkline + end dump)
_inference_only = False  # was -i passed? (no training loop runs)
_loss = []               # per-step training loss, for the sparkline
_heads = {}              # head index -> list of attention rows for the most recent sequence
_summary_loss = None     # loss at the last summary (snapshot); baseline for the "(delta)" readout


def configure(argv):
    """Parse `--viz [N]` (and `-i`) from argv. Sets the module `enabled` flag.

    Absent       -> visualization off.
    --viz / 0    -> running in-situ sparkline + one end-of-run dump.
    --viz N (>0) -> sparkline + attention matrix dump every N steps.
    """
    global enabled, _n, _inference_only
    _inference_only = '-i' in argv
    if '--viz' not in argv:
        return
    enabled = True
    i = argv.index('--viz')
    arg = argv[i + 1] if i + 1 < len(argv) else ''
    _n = int(arg) if arg.isdigit() else 0


def attn(pos_id, head, weights):
    """gpt() hook: cache every head's attention for the current sequence.

    Called once per head per position. `pos_id == 0 and head == 0` marks a new sequence,
    so the buffer always holds the most recent sequence's per-head causal matrices.
    """
    if pos_id == 0 and head == 0:
        _heads.clear()                       # new sequence -> fresh per-head matrices
    _heads.setdefault(head, []).append(list(weights))


def step(step_idx, num_steps, loss_val):
    """Per-training-step hook: a clean live progress line + periodic multi-row snapshot.

    Non-snapshot steps print a single carriage-return line (`loss V (D)`, D = change since
    the last summary) that overwrites in place -- and is kept out of train.log by the tee.
    On every N-th step the live line is erased (so no stale frame is left behind), then a
    permanent snapshot is printed: a one-line header + a height-3 loss sparkline + the
    attention matrix.
    """
    global _summary_loss
    _loss.append(loss_val)
    if _summary_loss is None:
        _summary_loss = loss_val                     # baseline = first step's loss
    delta = loss_val - _summary_loss                 # loss change since the last summary
    progress = f"step {step_idx+1:4d} / {num_steps:4d} | loss {loss_val:.4f} ({delta:+.4f})"
    if _n > 0 and (step_idx + 1) % _n == 0:
        # snapshot: erase the transient live line, then print a permanent block on clean lines
        print('\r' + ' ' * len(progress) + '\r', end='')   # wipe the carriage-return frame
        print(f"{progress} | loss trend:")
        for row in sparkline(_loss, height=3, max_width=40):
            print(f"  {row}")
        visualize_heads(_heads, name=f"Attention (causal) @ step {step_idx+1}")
        print()  # blank separator so the next live \r progress line can't overwrite the matrix
        _summary_loss = loss_val                     # reset baseline for the next window
    else:
        print(progress, end='\r')                    # live, overwriting progress line


def finish():
    """End-of-run hook (called only when --viz): tall loss sparkline + attention matrix dump."""
    if _loss:  # only after training (inference-only has no loss history)
        print("\n--- loss trajectory (full run) ---")
        for row in sparkline(_loss, height=3, max_width=60):
            print(row)
    # attention matrix: N=0 dumps here; N>0 already dumped periodically (so only inference-only falls back)
    if (_n == 0 or _inference_only) and _heads:
        visualize_heads(_heads, name="Attention (causal)")


# --------------------------------------------------------------------------
# stdout tee: mirror everything printed to the terminal into an appended log file
# --------------------------------------------------------------------------
class _Tee:
    """Mirror stdout to the terminal verbatim, but log only newline-committed lines.

    Carriage-return progress frames (`...\\r`) overwrite in place on the terminal and are
    transient, so they are dropped from the log -- only text ending in a newline is
    written to the file. This keeps the live `\\r` UI on screen while keeping per-step
    counters out of train.log. Other attributes (isatty/encoding/...) defer to the stream.
    """
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh
        self._line = ''  # current terminal line being assembled (may be \r-overwritten)
    def write(self, s):
        self._stream.write(s)                  # terminal: verbatim (keeps live \r updates)
        for ch in s:                            # log: commit only on newline; drop \r frames
            if ch == '\r':
                self._line = ''                 # overwritten on screen -> never logged
            elif ch == '\n':
                self._fh.write(self._line + '\n')
                self._fh.flush()
                self._line = ''
            else:
                self._line += ch
        return len(s)
    def flush(self):
        self._stream.flush()
        self._fh.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)


def tee_stdout(path="train.log"):
    """Append everything written to stdout into `path` (also still printed to the terminal)."""
    import sys, datetime
    fh = open(path, "a")
    fh.write(f"\n===== run {datetime.datetime.now().isoformat(timespec='seconds')} | {' '.join(sys.argv)} =====\n")
    sys.stdout = _Tee(sys.stdout, fh)

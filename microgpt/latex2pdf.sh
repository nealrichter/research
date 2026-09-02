#!/bin/bash
# Render LaTeX docs to PDF with terminal-rendered training run appendices
# Requires: xelatex (texlive/mactex), pango-view (brew install pango)
set -e
cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:$PATH"

# Render training logs to PNG via pango-view (terminal-style dark background)
# Split into page-sized halves for PDF inclusion
for log in train_microgpt.log train_microgpt_sft.log train_microgpt_dpo.log; do
    if [ -f "$log" ]; then
        base="${log%.log}"
        echo "rendering $log -> ${base}_p1.png, ${base}_p2.png"
        pango-view --font="Menlo 9" --background="#1e1e2e" --foreground="#cdd6f4" --output="${base}_full.png" -q "$log"
        HEIGHT=$(sips -g pixelHeight "${base}_full.png" | awk '/pixelHeight/{print $2}')
        WIDTH=$(sips -g pixelWidth "${base}_full.png" | awk '/pixelWidth/{print $2}')
        HALF=$((HEIGHT / 2))
        sips -c $HALF $WIDTH "${base}_full.png" --out "${base}_p1.png" > /dev/null 2>&1
        sips -c $HALF $WIDTH --cropOffset $HALF 0 "${base}_full.png" --out "${base}_p2.png" > /dev/null 2>&1
        rm -f "${base}_full.png"
    fi
done

# Build PDFs
echo -n "$(date +%H:%M:%S)" > buildtime.tex
for tex in microgpt_py.tex microgpt_sft_py.tex microgpt_dpo_py.tex; do
    echo "building $tex..."
    xelatex -shell-escape -interaction=nonstopmode "$tex" > /dev/null
    xelatex -shell-escape -interaction=nonstopmode "$tex" > /dev/null  # second pass for TOC
done

# Clean up LaTeX temp files (NEVER delete .log or .png)
rm -f microgpt_py.aux microgpt_py.toc microgpt_py.out
rm -f microgpt_sft_py.aux microgpt_sft_py.toc microgpt_sft_py.out
rm -f microgpt_dpo_py.aux microgpt_dpo_py.toc microgpt_dpo_py.out

echo "done: microgpt_py.pdf microgpt_sft_py.pdf microgpt_dpo_py.pdf"

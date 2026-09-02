# Annotation Guide for microgpt Teaching Documents

## Purpose

The `.tex` and `.md` annotated guides exist to **teach**. They are not code — they are
educational documents. The "minimal" instruction applies to Python implementations, NOT
to explanatory writing.

## Requirements for the teaching docs

1. **Rich explanatory prose** between every code block. Explain:
   - WHY each piece exists (motivation)
   - WHAT it does (functionality)
   - HOW it connects to the math (equations inline with the code they implement)
   - WHERE it fits in the bigger picture (flow/pipeline context)

2. **Never strip explanations to be "minimal."** A teaching doc with just code and equations
   is useless — it's just the source file with LaTeX formatting. The value is the narrative
   that connects concepts for the reader.

3. **Line numbers must match the actual source.** Verify with `grep -n` before writing.

4. **Equations in context.** Show the math formula immediately before or after the code that
   implements it, with a sentence connecting them ("This implements..." or "The code above
   computes...").

5. **References at point of use.** Cite papers where the concept is introduced, not just in
   a bibliography at the end.

6. **Vim color scheme** for code blocks (dark background, colored syntax). Keep the teaching
   prose in normal black-on-white.

7. **Appendix with example run** (`train_*.log`) showing actual output with sparklines,
   attention matrices, and generated samples. Conditional on file existence.

8. **DO NOT DELETE train_*.log files.** Ever. Under any circumstances. The latex2pdf.sh
   cleanup must only remove LaTeX temp files by explicit name, never `*.log`.

## Structure

Each doc follows this flow:
1. Abstract + source link
2. Introduction (goals, context, mention of microgpt_viz.py)
3. Sections following the code's logical order
4. Usage
5. References
6. Appendix (example run)

## What NOT to do

- Do not produce a "minimal" teaching doc
- Do not skip prose between code blocks
- Do not guess line numbers — verify them
- Do not delete any log files

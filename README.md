# Tutorial: Bayesian vs. LLM-Driven Hyperparameter Search

This walks through three 10-generation search experiments on the same 1%
Criteo sample (`data/criteo_01.csv.gz`, already included in this repo and
ready to go — no download or preprocessing needed), comparing a classic
non-LLM Bayesian optimizer against an LLM acting as the hyperparameter
proposer — with and without giving the LLM room to reason about *why* a
change might help.

Before starting, follow the **Quickstart** in `AGENTS.md` (install deps,
optionally start the dashboard).

A0 is fully mechanical — one script call, no LLM involved, deterministic
given its seed. A1 and A2+A3 are **not** meant to be run from a fixed list
of commands: they need an LLM in the loop deciding each generation's
hyperparameters, one generation at a time, informed by the previous
generation's outcome. For those, this file gives the natural-language prompt
to hand to an LLM agent that has shell access to this repo — not the
per-generation commands, since those are exactly what the LLM is supposed
to produce itself.

Every generation is evaluated by `windowed_auc_avg` — AUC averaged over
sliding 20,000-row windows of the single prequential pass (predict, then
train), per `src/analyze.py`'s protocol.

---

## Experiment A0 — Classic Bayesian Optimization (DCNv2)

Baseline: unmodified `model_dcnv2.json`. Proposer: Gaussian Process +
Expected Improvement (`src/bayes_search.py`), no domain knowledge, no LLM
involved at all.

**The prompt that started this experiment:**

> Run a classic Bayesian optimization test for `model_dcnv2.json`. Just use
> the search script as-is — don't tune the model yourself. Run it for 10
> iterations on the 1% data sample, and call the experiment `A0`.

Which, since `src/bayes_search.py` needs no LLM in the loop, resolves to one
command:

```bash
uv run python src/bayes_search.py search A0 \
    --base model_dcnv2.json --data data/criteo_01.csv.gz --n-iterations 10
```

That single command runs all 10 generations (baseline + 9 GP-EI-proposed
generations) end to end.

---

## Experiment A1 — LLM Surrogate, No Explanations (DCNv2)

Same baseline, same data. Here the hyperparameter proposer is an LLM agent
instructed to pick new values each generation *without* explaining its
reasoning — just numbers in, AUC out, repeat.

**The prompt that started this experiment:**

> Run a new experiment. Take `model_dcnv2.json` again. Run 10 generations.
> This time, instead of using the Bayesian process, you are the surrogate
> model — you decide which hyperparameters to try. Don't explain your
> choices, just suggest new hyperparameters after each iteration. Call this
> experiment `A1`.

---

## Experiment A2+A3 — Implement DCN² and Run an LLM Surrogate *With* Reasoning

This experiment has two parts: first implementing a new model architecture
(DCN²) that doesn't exist in the codebase yet, then running the same kind of
LLM-surrogate search as A1 — but this time requiring the LLM to explain why
a change might help and what it will try next.

### Step 1 — Implement DCN² support

**The prompt that started this step:**

> There's a paper `dcn_paper.pdf` for implementing DCN² in this repo. Please
> implement support for it — pick a new algorithm name if one is needed,
> rather than just extending the existing DCNv2 algorithm. Write me down the
> features you have added to it.

### Step 2 — Run a 10-generation LLM surrogate search, with reasoning

Same protocol as A1, but at each generation the LLM must state why it
expects the change to help and what it intends to try next, and must decide
each generation's config itself rather than following a fixed list.

**The prompt that started this experiment:**

> Take `model_dcn2.json`. Be the surrogate model again — 10 iterations —
> but this time explain why you think each change will improve the AUC,
> and explicitly say what you'll do next, before deciding how to improve
> the model. Name the experiment A2+A3

---

## Notes

- A0's Bayesian proposer has no domain knowledge: its random-init phase can
  sample combinations that collapse training (e.g. very high dropout paired
  with very few embedding dimensions). That's an expected failure mode of
  blind search worth watching for when comparing runs, not a bug.
- A1 vs. A2+A3 isn't a clean reasoning-vs-no-reasoning comparison on its
  own, since they also differ in architecture (DCNv2 vs. DCN²) and DCN²'s
  untuned baseline is structurally weaker (no residual anchor in
  `onlydense`). For an apples-to-apples reasoning ablation, run both A1's
  and A3's prompts against the *same* model config.

## After the tutorial

All of the above runs on the bundled 1% sample so each generation trains in
seconds. Once you're done, `AGENTS.md`'s **Beyond the tutorial** section
covers downloading the full Criteo dataset and building larger samples (10%,
full) to keep experimenting on your own — `src/experiment.py` and
`src/bayes_search.py` work exactly the same way, just point `--data` at the
bigger file.

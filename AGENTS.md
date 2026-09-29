# AGENTS.md — Repo Guide for Agents

This repo is a minimal JAX framework for **single-pass ("prequential") CTR
model training on Criteo**, plus two hyperparameter-search harnesses (Bayesian
and LLM-driven) built on top of it. This file gets a new agent from zero to
running its own search experiment.

## Quickstart (do this first, in order)

```bash
# 1. Install dependencies
uv sync --project config

# 2. Start the dashboard (reads everything under experiments/)
uv run --project config streamlit run src/dashboard.py
```

That's it — the 1% Criteo sample used throughout this tutorial
(`data/criteo_01.csv.gz`, built with `--sample-mod 100`: every 100th row,
preserving temporal order) already ships in this repo. `src/train.py` reads it
directly, gzip or not, and it works with any `hash_size_bits` config, so
there's nothing to download or preprocess before you start running
experiments.

## Beyond the tutorial: scaling up with the full dataset

Once you've worked through the tutorial's experiments on the bundled 1%
sample, you can keep going on your own with more data:

```bash
scripts/download_data.sh full   # -> data/train.txt (~4.6 GB compressed download)
uv run --project config python src/preprocess_criteo.py data/train.txt data/criteo_10.csv.gz --sample-mod 10   # 10% sample
uv run --project config python src/preprocess_criteo.py data/train.txt data/criteo_full.csv.gz                 # full dataset
```

`--sample-mod k` keeps every k-th row (`row_index % k == 0`), preserving the
original temporal order — the whole single-pass protocol depends on that
order, so never shuffle these files. Sampling is applied on the *raw* file,
not on an already-hashed one, but all resulting CSVs (including the bundled
1% sample) share the same hash space (`HASH_SEED = 42`), so any of them work
with any `hash_size_bits` config — no need to regenerate a file just to
change that setting. Point `--data` at the bigger file and every workflow
below (`src/experiment.py`, `src/bayes_search.py`, the dashboard) works unchanged.

## Repository layout

The root contains this guide, the tutorial README, and project folders only:

```text
.github/   CI workflows
config/    pyproject.toml, uv.lock, and project-local ignore rules
data/      bundled and generated Criteo datasets
experiments/ generated experiment outputs (gitignored)
models/    model seed configs
papers/    reference papers
scripts/   utility scripts
src/       training, model, and search code
tasks/     one file per tutorial exercise
tests/     pytest suite
```

## Repo map

| File | Role |
|---|---|
| `src/preprocess_criteo.py` | Raw `train.txt` (TSV) → hashed CSV. Continuous features log-bucketed, categoricals murmur3-hashed. One preprocessed file serves every `hash_size_bits` sweep. |
| `src/models.py` | Parameter init (`init_params`) + forward pass (`forward`) for every `algorithm`: `lr`, `nn`, `dcnv2`, `dcn2`. Pure JAX, two pytrees: `params['emb']` (hashed table, sparse updates) and `params['dense']` (everything else, dense updates). |
| `src/optim.py` | Hand-rolled optimizers: dense Adam/SGD/SOAP for `params['dense']`, **row-sparse** Adam/SGD for `params['emb']` (only touches rows present in the batch — required to make a single pass over a `2^23`-row table tractable). |
| `src/train.py` | Streaming trainer: for every batch, predict *then* train (prequential eval — every example is scored before the model has seen it). |
| `src/analyze.py` | Scores a predictions CSV under the leaderboard protocol: overall AUC/log-loss + AUC averaged over sliding 20,000-row windows. |
| `src/experiment.py` | Generation-based experiment runner — see below. |
| `src/bayes_search.py` | Non-LLM Gaussian-Process + Expected-Improvement hyperparameter search, writing into the same experiment folder structure. |
| `src/dashboard.py` | Streamlit dashboard over `experiments/*/results.csv` + `best.json`. |
| `models/` | Model seed configs; generated `model_dcn2.json` is gitignored. |
| `config/pyproject.toml`, `config/uv.lock` | Python project metadata, dependencies, and lockfile. |
| `README.md` | Tutorial setup guide (prerequisites, install, run the dashboard). |
| `tasks/main.md`, `tasks/A0.md`, `tasks/A1.md`, `tasks/A2_A3.md` | The tutorial exercises, one file per experiment. |
| `papers/dcn_paper.pdf` | DCN² reference paper. |
| `scripts/download_data.sh` | Dataset downloader. |
| `tests/test_models.py` | Shape/gradient/sparsity tests per algorithm — run with `uv run --project config pytest -c config/pyproject.toml -q`. |

## The research loop (`src/experiment.py`)

Every experiment lives in its own folder, `experiments/<name>/`:

```
start.json              seed config (generation 0, copied verbatim from --base)
best.json               config of the best generation seen so far
best_metrics.json       metrics for best.json
results.csv             one row per generation, in run order
research_log.md         hypothesis / config diff / result per generation
generations/
    end_training_<ts>.json   the config actually trained in that generation
    preds_<ts>.csv            its predictions (gitignored; for re-checking a number by hand)
```

"Best" = highest `windowed_auc_avg`, matching `src/analyze.py`'s protocol. A
generation's config is written to disk *before* it's known whether it beat
the best, so every attempt is inspectable and reproducible, not just the
winner.

```bash
# generation 0: trains --base as-is, seeds start.json/best.json
uv run --project config python src/experiment.py init my_experiment --base models/model_dcnv2.json --data data/criteo_01.csv.gz

# one more generation, config given as overrides on top of the current best.json
uv run --project config python src/experiment.py run my_experiment --data data/criteo_01.csv.gz \
    --plan '[{"hypothesis": "smaller batches -> more gradient steps", "overrides": {"batch_size": 500}}]'

# or N generations in one call — a plan is a JSON list of
#   {"hypothesis": str, "overrides": {...}}   merged onto the *current* best
#   {"hypothesis": str, "config": {...}}      full config, used as-is
# run in order; each updates best.json before the next entry runs
uv run --project config python src/experiment.py run my_experiment --data data/criteo_01.csv.gz --plan plan.json
```

There is no built-in hyperparameter *proposer* for this path — something
(you, another script, or an LLM) has to decide what goes in `overrides` each
generation. That's what makes it comparable to `src/bayes_search.py`, which
proposes automatically.

## Non-LLM search (`src/bayes_search.py`)

Same experiment-folder output format as `src/experiment.py`, but hyperparameters
are proposed by a Gaussian Process + Expected Improvement loop
(`scikit-learn`'s `GaussianProcessRegressor`, Matérn kernel), not by a human
or an LLM. One command runs the whole thing:

```bash
uv run --project config python src/bayes_search.py search my_experiment \
    --base models/model_dcnv2.json --data data/criteo_01.csv.gz --n-iterations 10

# add more generations to an existing search later
uv run --project config python src/bayes_search.py resume my_experiment --data data/criteo_01.csv.gz --n-iterations 5
```

`--n-iterations` counts generation 0 (the unmodified baseline). Search spaces
are built in per algorithm (`lr`, `dcnv2`) in `DEFAULT_SPACES`; pass `--space`
with a JSON dict to override. Given a fixed `--seed` (default is stable), the
sequence of proposed generations is deterministic, so re-running the same
command reproduces the same results — useful for truncating a longer run
into a shorter, still-exact one (e.g. the first 10 generations of a 20-gen
run are identical to a fresh 10-gen run).

The GP has no domain knowledge: its random-init phase can (and does) sample
combinations that collapse training (e.g. high dropout + very low embedding
dimension → AUC near 0.50). That's a real, expected failure mode of blind
search, not a bug — useful context if you're comparing it against a
reasoning-driven proposer.

## Model configs (`models/model_*.json` → `algorithm` field)

All four algorithms share the same config schema (`Config` in `src/models.py`,
extra fields are just ignored by algorithms that don't need them):

- `lr` — logistic regression: logit = sum of 1-dim looked-up embedding weights.
- `nn` — plain MLP over flattened embeddings.
- `dcnv2` — DCNv2: low-rank–projected Cross layers (residual, `x0`-anchored) + MLP.
- `dcn2` — DCN² (Škrlj et al. 2025, arXiv:2506.21624): three components —
  1. **Collision-weighted lookups**: the embedding table has one extra
     learnable per-row column (init = 1.0), multiplied elementwise into the
     looked-up embedding.
  2. **`onlydense` stack**: full-rank square `W` per layer, `x = (relu(Wx+b) * x) * phi`
     — no residual, no `x0` anchor (unlike DCNv2's Cross layer), so it's more
     sensitive to batch size / init scale than DCNv2.
  3. **SimLayer**: an explicit pairwise dot-product over all field-embedding
     pairs, weighted and summed into one extra scalar, added directly to the
     final logit (not passed through the MLP).

`hash_size_bits` controls the embedding table size (`2^bits + 1` rows,
+1 sentinel); any preprocessed CSV works with any value.

## Tests

```bash
uv run --project config pytest -c config/pyproject.toml -q
```

Covers: forward-pass shapes per algorithm, the collision-weight init
invariant, gradient flow, and — the load-bearing one — that a sparse
optimizer step only moves rows that appeared in the batch (works for `adam`,
`sgd`, and `soap`).

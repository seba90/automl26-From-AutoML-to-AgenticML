# Tutorial: Bayesian vs. LLM-Driven Hyperparameter Search

Hands-on companion for the AutoML 2026 tutorial [*From AutoML to AgenticML:
LLMs as the New SearchOperator*](https://2026.automl.cc/from-automl-to-agenticml-llms-as-the-new-searchoperator/),
which explores LLMs as search operators in AutoML — proposing candidates,
acting as mutation operators, and orchestrating search loops — in place of
classical Bayesian and evolutionary methods. This repo puts that idea into
practice: an LLM acting as the hyperparameter proposer, compared against a
classic Bayesian optimizer, on a single-pass CTR model trained on a 1%
Criteo sample.

## Prerequisites

- [Claude Code](https://claude.com/claude-code) — Claude Sonnet 5 is enough.

## Setup

```bash
uv sync --project config
uv run --project config streamlit run src/dashboard.py
```

Let Claude read the [`AGENTS.md`](AGENTS.md).
The 1% Criteo sample (`data/criteo_01.csv.gz`) already ships in this repo.

## Tasks
Once you setup your environment you can start with the task which live in [`tasks/`](tasks/main.md).

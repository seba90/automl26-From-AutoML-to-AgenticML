# Tasks

Three 10-generation search experiments on the same 1% Criteo sample,
comparing a classic non-LLM Bayesian optimizer against an LLM acting as the
hyperparameter proposer — with and without giving the LLM room to reason
about *why* a change might help. Run them in order:

1. [A0](A0.md) — Classic Bayesian Optimization (DCNv2)
2. [A1](A1.md) — LLM Surrogate, No Explanations (DCNv2)
3. [A2+A3](A2_A3.md) — Implement DCN² and run an LLM surrogate *with* reasoning

Every generation is evaluated by `windowed_auc_avg` — AUC averaged over
sliding 20,000-row windows of the single prequential pass (predict, then
train), per `src/analyze.py`'s protocol.

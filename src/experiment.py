"""Generation-based experiment runner for the model-config search.

Each experiment lives in its own folder under experiments/<name>/:

    start.json                     seed config (generation 0, copied verbatim
                                    from the --base config passed to `init`)
    best.json                      config of the best generation seen so far
    best_metrics.json              metrics for best.json
    results.csv                    one row per generation, in run order
    research_log.md                hypothesis / config diff / result per generation
    generations/
        end_training_<ts>.json     the config trained in that generation
        preds_<ts>.csv              its predictions (gitignored, kept for
                                    re-running / verifying a number by hand)

"Best" is decided by windowed-AUC average, matching the leaderboard protocol
in analyze.py. A generation's config is written to disk *before* it's known
whether it beat the best, so every attempt is inspectable and reproducible,
not just the winner.

Usage:
    # generation 0: trains and evaluates --base as-is, seeds start.json/best.json
    uv run --project config python src/experiment.py init my_experiment --base models/model_lr.json --data data_example.csv

    # one generation, config given inline as a JSON file
    uv run --project config python src/experiment.py run my_experiment --data data_example.csv \\
        --config candidate.json --hypothesis "higher LR converges faster in one pass"

    # N generations in one call: a plan is a JSON list of
    #   {"hypothesis": str, "overrides": {...}}   -- merged onto the current best
    #   {"hypothesis": str, "config": {...}}      -- full config, used as-is
    # entries, run in order; each updates best.json before the next one runs.
    uv run --project config python src/experiment.py run my_experiment --data data_example.csv --plan plan.json
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from analyze import windowed_auc

EXPERIMENTS_DIR = Path('experiments')
BEST_METRIC = 'windowed_auc_avg'  # the field in the metrics dict that decides "best"

METRIC_FIELDS = [
    'n_examples', 'overall_auc', 'overall_log_loss',
    'windowed_auc_avg', 'windowed_auc_median', 'windowed_auc_min',
    'windowed_auc_max', 'windowed_auc_std', 'n_windows',
]
RESULTS_FIELDS = ['timestamp', 'generation_file', 'hypothesis', 'is_new_best'] + METRIC_FIELDS


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%dT%H%M%S')


def exp_dir(name: str) -> Path:
    return EXPERIMENTS_DIR / name


def train_and_evaluate(config: dict, data: Path, window: int, stride: int) -> dict:
    """Run train.py on `config` against `data`, then score the predictions."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / 'config.json'
        preds_path = Path(tmp) / 'preds.csv'
        config_path.write_text(json.dumps(config, indent=2))

        subprocess.run(
            ['uv', 'run', '--project', 'config', 'python', 'src/train.py', str(config_path), str(data),
             '--output_file', str(preds_path)],
            check=True,
        )

        df = pd.read_csv(preds_path, header=None, dtype=np.float32,
                         names=['label', 'prediction'])
        y, p = df.label.values, df.prediction.values
        preds_bytes = preds_path.read_bytes()

    aucs, _skipped = windowed_auc(y, p, window, stride)
    metrics = {
        'n_examples': int(len(y)),
        'overall_auc': float(roc_auc_score(y, p)),
        'overall_log_loss': float(log_loss(y, p)),
        'n_windows': int(len(aucs)),
    }
    if len(aucs):
        metrics.update({
            'windowed_auc_avg': float(aucs.mean()),
            'windowed_auc_median': float(np.median(aucs)),
            'windowed_auc_min': float(aucs.min()),
            'windowed_auc_max': float(aucs.max()),
            'windowed_auc_std': float(aucs.std()),
        })
    else:
        metrics.update({k: None for k in
                        ['windowed_auc_avg', 'windowed_auc_median',
                         'windowed_auc_min', 'windowed_auc_max', 'windowed_auc_std']})
    return metrics, preds_bytes


def config_diff(old: dict, new: dict) -> dict:
    keys = sorted(set(old) | set(new))
    return {k: (old.get(k), new.get(k)) for k in keys if old.get(k) != new.get(k)}


def append_csv_row(path: Path, row: dict, fieldnames: list):
    is_new = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def append_log_entry(path: Path, *, generation_file: str, hypothesis: str,
                     diff: dict, metrics: dict, is_new_best: bool):
    diff_lines = '\n'.join(f'  - `{k}`: `{old}` -> `{new}`' for k, (old, new) in diff.items()) \
        or '  (no change from previous best)'
    entry = f"""
## {generation_file}

**Hypothesis:** {hypothesis}

**Config diff (vs. best going in):**
{diff_lines}

**Result:** windowed AUC avg = {metrics.get('windowed_auc_avg')}, \
overall AUC = {metrics.get('overall_auc')}, log loss = {metrics.get('overall_log_loss')}
**New best:** {'yes' if is_new_best else 'no'}

**Verdict (fill in after verifying):** TBD
"""
    with open(path, 'a') as f:
        f.write(entry)


def next_generation_num(results_path: Path) -> int:
    """Generation 0 is the seed trained by `init`; each `run` call after that
    increments by one. Counted from results.csv row count so it survives
    across separate invocations of this script.

    Uses csv.reader (not a line count) because a multi-line hypothesis is
    correctly quoted by the CSV writer and would otherwise inflate the count."""
    if not results_path.exists():
        return 0
    with open(results_path, newline='') as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def run_generation(name: str, config: dict, data: Path, hypothesis: str,
                   window: int, stride: int) -> dict:
    """Train+evaluate one config, record it as a generation, update best.json
    if it wins. Returns the metrics dict."""
    edir = exp_dir(name)
    gens_dir = edir / 'generations'
    gens_dir.mkdir(parents=True, exist_ok=True)

    best_path = edir / 'best.json'
    best_metrics_path = edir / 'best_metrics.json'
    prev_best_config = json.loads(best_path.read_text()) if best_path.exists() else {}
    prev_best_metrics = json.loads(best_metrics_path.read_text()) if best_metrics_path.exists() else {}

    print(f'[{name}] training generation with hypothesis: {hypothesis}', file=sys.stderr)
    metrics, preds_bytes = train_and_evaluate(config, data, window, stride)

    ts = timestamp()
    gen_num = next_generation_num(edir / 'results.csv')
    generation_file = f'{ts}_gen_{gen_num}.json'
    (gens_dir / generation_file).write_text(json.dumps(config, indent=2))
    (gens_dir / f'preds_{ts}_gen_{gen_num}.csv').write_bytes(preds_bytes)

    prev_score = prev_best_metrics.get(BEST_METRIC)
    this_score = metrics.get(BEST_METRIC)
    is_new_best = prev_score is None or (this_score is not None and this_score > prev_score)

    if is_new_best:
        best_path.write_text(json.dumps(config, indent=2))
        best_metrics_path.write_text(json.dumps(metrics, indent=2))

    diff = config_diff(prev_best_config, config)
    append_csv_row(edir / 'results.csv',
                   {'timestamp': ts, 'generation_file': generation_file,
                    'hypothesis': hypothesis, 'is_new_best': is_new_best, **metrics},
                   RESULTS_FIELDS)
    append_log_entry(edir / 'research_log.md', generation_file=generation_file,
                     hypothesis=hypothesis, diff=diff, metrics=metrics,
                     is_new_best=is_new_best)

    print(f'[{name}] {generation_file}: {BEST_METRIC}={this_score} '
          f'(new best: {is_new_best})', file=sys.stderr)
    return metrics


def cmd_init(args):
    edir = exp_dir(args.name)
    if edir.exists():
        print(f'error: experiment {args.name!r} already exists at {edir}', file=sys.stderr)
        sys.exit(1)
    edir.mkdir(parents=True)
    (edir / 'generations').mkdir()

    base_config = json.loads(Path(args.base).read_text())
    (edir / 'start.json').write_text(json.dumps(base_config, indent=2))

    run_generation(args.name, base_config, Path(args.data),
                   hypothesis=f'baseline: unmodified {args.base}',
                   window=args.window, stride=args.stride or args.window)
    print(f'initialized experiment {args.name!r} at {edir}', file=sys.stderr)


def cmd_run(args):
    edir = exp_dir(args.name)
    if not edir.exists():
        print(f'error: experiment {args.name!r} not found at {edir}; run `init` first',
              file=sys.stderr)
        sys.exit(1)

    stride = args.stride or args.window

    if args.plan:
        try:
            plan = json.loads(args.plan)
        except json.JSONDecodeError:
            plan = json.loads(Path(args.plan).read_text())
        for i, entry in enumerate(plan):
            best_path = edir / 'best.json'
            current_best = json.loads(best_path.read_text()) if best_path.exists() else {}
            if 'config' in entry:
                config = entry['config']
            else:
                config = {**current_best, **entry.get('overrides', {})}
            print(f'[{args.name}] plan entry {i + 1}/{len(plan)}', file=sys.stderr)
            run_generation(args.name, config, Path(args.data),
                           hypothesis=entry.get('hypothesis', ''),
                           window=args.window, stride=stride)
    else:
        config = json.loads(Path(args.config).read_text())
        run_generation(args.name, config, Path(args.data),
                       hypothesis=args.hypothesis or '', window=args.window, stride=stride)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p_init = sub.add_parser('init', help='seed a new experiment (trains generation 0)')
    p_init.add_argument('name', help='experiment name -> experiments/<name>/')
    p_init.add_argument('--base', required=True, help='seed model config JSON')
    p_init.add_argument('--data', required=True, help='training CSV')
    p_init.add_argument('--window', type=int, default=20000)
    p_init.add_argument('--stride', type=int, default=0)
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser('run', help='run one or more generations')
    p_run.add_argument('name', help='existing experiment name')
    p_run.add_argument('--data', required=True, help='training CSV')
    group = p_run.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', help='single candidate config JSON (one generation)')
    group.add_argument('--plan', help='JSON list of {hypothesis, config|overrides} '
                                      '(N generations, one call)')
    p_run.add_argument('--hypothesis', default='', help='only used with --config')
    p_run.add_argument('--window', type=int, default=20000)
    p_run.add_argument('--stride', type=int, default=0)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

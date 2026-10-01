"""Non-LLM hyperparameter search: Gaussian-process Bayesian optimization.

Candidates are chosen purely by a GP surrogate fit on windowed_auc_avg from
previous generations (Expected Improvement acquisition) -- no LLM involved.
Writes into the *same* experiments/<name>/ folder layout as experiment.py
(start.json, best.json, results.csv, research_log.md, generations/), so its
search progress is directly comparable, in dashboard.py, to an LLM-driven
experiment.py run on the same base config and data.

Usage:
    # one command: seeds the experiment (baseline = generation 0) and runs
    # N-1 further GP-EI-driven generations against it
    uv run --project config python src/bayes_search.py search bo_lr_01 --base models/model_lr.json \\
        --data data/criteo_01.csv.gz --n-iterations 8 --n-initial 3

    # optionally override the default search space for the algorithm
    uv run --project config python src/bayes_search.py search bo_lr_01 --base models/model_lr.json \\
        --data data/criteo_01.csv.gz --space my_space.json --n-iterations 8

    # add more GP-EI generations to a search that already exists
    uv run --project config python src/bayes_search.py resume bo_lr_01 --data data/criteo_01.csv.gz --n-iterations 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from experiment import BEST_METRIC, cmd_init as experiment_cmd_init, exp_dir, run_generation

# Each entry: {"type": "log"|"float"|"int", "low": ..., "high": ...}
DEFAULT_SPACES = {
    'lr': {
        'learning_rate': {'type': 'log', 'low': 1e-4, 'high': 1e-1},
        'adam_beta1': {'type': 'float', 'low': 0.0, 'high': 0.99},
        'regularization': {'type': 'log', 'low': 1e-8, 'high': 1e-3},
        'init_weight_std': {'type': 'log', 'low': 1e-4, 'high': 1e-1},
    },
    'dcnv2': {
        'learning_rate': {'type': 'log', 'low': 1e-4, 'high': 1e-1},
        'adam_beta1': {'type': 'float', 'low': 0.0, 'high': 0.99},
        'regularization': {'type': 'log', 'low': 1e-8, 'high': 1e-3},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
        'dimensions': {'type': 'int', 'low': 4, 'high': 32},
        'hidden_layer_size': {'type': 'int', 'low': 32, 'high': 256},
        'n_hidden_layers': {'type': 'int', 'low': 1, 'high': 4},
        'cross_n_hidden': {'type': 'int', 'low': 1, 'high': 4},
        'cross_projection_dim': {'type': 'int', 'low': 16, 'high': 128},
    },
}


def load_space(space_arg: str | None, algorithm: str) -> dict:
    if space_arg:
        return json.loads(Path(space_arg).read_text())
    if algorithm not in DEFAULT_SPACES:
        raise SystemExit(f'no default search space for algorithm={algorithm!r}; pass --space')
    return DEFAULT_SPACES[algorithm]


def encode(config: dict, space: dict) -> np.ndarray:
    """config -> normalized [0, 1] vector, in space's key order."""
    x = []
    for name, spec in space.items():
        v = config[name]
        if spec['type'] == 'log':
            v, lo, hi = np.log(v), np.log(spec['low']), np.log(spec['high'])
        else:
            lo, hi = spec['low'], spec['high']
        x.append((v - lo) / (hi - lo))
    return np.array(x)


def decode(x: np.ndarray, space: dict) -> dict:
    """normalized [0, 1] vector -> config overrides."""
    overrides = {}
    for xi, (name, spec) in zip(x, space.items()):
        xi = float(np.clip(xi, 0.0, 1.0))
        if spec['type'] == 'log':
            lo, hi = np.log(spec['low']), np.log(spec['high'])
            v = float(np.exp(lo + xi * (hi - lo)))
        else:
            lo, hi = spec['low'], spec['high']
            v = lo + xi * (hi - lo)
            if spec['type'] == 'int':
                v = int(round(v))
        overrides[name] = v
    return overrides


def load_history(edir: Path, space: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read every past generation's config + BEST_METRIC to build GP training
    data. Generations that predate a tuned parameter, or that have no
    windowed_auc_avg (e.g. too little data for a full window), are skipped."""
    results_path = edir / 'results.csv'
    if not results_path.exists():
        return np.empty((0, len(space))), np.empty((0,))
    df = pd.read_csv(results_path)
    xs, ys = [], []
    for _, row in df.iterrows():
        if pd.isna(row.get(BEST_METRIC)):
            continue
        gen_file = edir / 'generations' / row['generation_file']
        if not gen_file.exists():
            continue
        config = json.loads(gen_file.read_text())
        try:
            xs.append(encode(config, space))
        except KeyError:
            continue
        ys.append(row[BEST_METRIC])
    if not xs:
        return np.empty((0, len(space))), np.empty((0,))
    return np.array(xs), np.array(ys)


def expected_improvement(x: np.ndarray, gp: GaussianProcessRegressor, y_best: float,
                         xi: float = 0.01) -> float:
    mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
    sigma = max(float(sigma[0]), 1e-9)
    improvement = float(mu[0]) - y_best - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def propose_next(space: dict, X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                 n_restarts: int = 10) -> np.ndarray:
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-4)
    gp_seed = int(rng.integers(0, 2 ** 32 - 1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                  random_state=gp_seed)
    gp.fit(X, y)
    y_best = float(y.max())

    best_x, best_ei = None, -np.inf
    bounds = [(0.0, 1.0)] * len(space)
    for _ in range(n_restarts):
        x0 = rng.uniform(0.0, 1.0, size=len(space))
        res = minimize(lambda x: -expected_improvement(x, gp, y_best), x0,
                       bounds=bounds, method='L-BFGS-B')
        if -res.fun > best_ei:
            best_ei, best_x = -res.fun, res.x
    return best_x


def bo_loop(name: str, base_config: dict, data: Path, space: dict, n_generations: int,
           n_initial: int, rng: np.random.Generator, window: int, stride: int):
    """Run `n_generations` more GP-EI-driven generations against an existing
    (already seeded) experiment."""
    edir = exp_dir(name)
    for i in range(n_generations):
        X, y = load_history(edir, space)
        n_done = len(X)
        if n_done < n_initial:
            x = rng.uniform(0.0, 1.0, size=len(space))
            method = 'random-init'
        else:
            x = propose_next(space, X, y, rng)
            method = 'GP-EI'
        overrides = decode(x, space)
        config = {**base_config, **overrides}
        hypothesis = (f'bayes-opt ({method}, {n_done} prior points): ' +
                     ', '.join(f'{k}={v:.6g}' for k, v in overrides.items()))
        print(f'[{name}] {i + 1}/{n_generations} {hypothesis}', file=sys.stderr)
        run_generation(name, config, data, hypothesis=hypothesis, window=window, stride=stride)


def cmd_search(args):
    edir = exp_dir(args.name)
    if edir.exists():
        print(f'error: experiment {args.name!r} already exists at {edir}; '
              f'use `resume` to add more generations to it', file=sys.stderr)
        sys.exit(1)

    experiment_cmd_init(argparse.Namespace(name=args.name, base=args.base, data=args.data,
                                           window=args.window, stride=args.stride))

    base_config = json.loads((edir / 'start.json').read_text())
    space = load_space(args.space, base_config.get('algorithm'))
    stride = args.stride or args.window
    bo_loop(args.name, base_config, Path(args.data), space,
           n_generations=max(args.n_iterations - 1, 0),  # gen 0 already trained by init
           n_initial=args.n_initial, rng=np.random.default_rng(args.seed),
           window=args.window, stride=stride)


def cmd_resume(args):
    edir = exp_dir(args.name)
    if not edir.exists():
        print(f'error: experiment {args.name!r} not found at {edir}; run `search` first',
              file=sys.stderr)
        sys.exit(1)

    base_config = json.loads((edir / 'start.json').read_text())
    space = load_space(args.space, base_config.get('algorithm'))
    stride = args.stride or args.window
    bo_loop(args.name, base_config, Path(args.data), space,
           n_generations=args.n_iterations, n_initial=args.n_initial,
           rng=np.random.default_rng(args.seed), window=args.window, stride=stride)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def add_common_bo_args(p):
        p.add_argument('--space', default=None,
                       help='JSON search space; defaults to a built-in space for the '
                            'base config\'s algorithm (lr or dcnv2)')
        p.add_argument('--n-initial', type=int, default=3,
                       help='number of random-sampled generations before GP-EI kicks in '
                            '(counts the baseline generation 0)')
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--window', type=int, default=20000)
        p.add_argument('--stride', type=int, default=0)

    p_search = sub.add_parser('search', help='seed a new experiment and run a full GP-EI search')
    p_search.add_argument('name', help='experiment name -> experiments/<name>/')
    p_search.add_argument('--base', required=True, help='seed model config JSON')
    p_search.add_argument('--data', required=True, help='training CSV')
    p_search.add_argument('--n-iterations', type=int, default=8,
                          help='total generations, including the baseline (generation 0)')
    add_common_bo_args(p_search)
    p_search.set_defaults(func=cmd_search)

    p_resume = sub.add_parser('resume', help='add more GP-EI generations to an existing search')
    p_resume.add_argument('name', help='existing experiment name')
    p_resume.add_argument('--data', required=True, help='training CSV')
    p_resume.add_argument('--n-iterations', type=int, default=5,
                          help='number of additional generations to run')
    add_common_bo_args(p_resume)
    p_resume.set_defaults(func=cmd_resume)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

"""Single-pass streaming trainer (prequential evaluation: predict, then train).

Usage:
    python src/train.py model.json data.csv --output_file preds.csv

The data CSV has a header; the first column is the label, all other columns are
hashed int32 feature indices (see preprocess_criteo.py). For every batch the
model first predicts (those predictions are the evaluation output), then does
one gradient step — so every example is scored before the model has seen it.
"""

import argparse
import gzip
import json
import pickle
import sys
import time
import jax
import numpy as np
import pandas as pd

from models import Config, init_params, loss_fn, forward
from optim import make_optimizer


def _open_text(path):
    """Open a CSV for a single header line -- transparently, whether or not
    it's gzip-compressed (pandas.read_csv infers that from the extension
    itself; this plain read doesn't, so it needs its own check)."""
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path)


def set_features(args, cfg):
    """Read the CSV header; resolve '_all' and validate feature names."""
    with _open_text(args.input_file) as f:
        header = f.readline().strip().split(',')

    if len(cfg.features) == 1 and cfg.features[0] == '_all':
        header.remove(cfg.label_feature)
        cfg.features = header
    else:
        for ftr in cfg.features:
            if ftr not in header:
                raise ValueError(f'Feature {ftr} not found in input CSV file')

    cfg.n_features = len(cfg.features)


def train_model(params, opt, args, cfg):
    out = open(args.output_file, 'w') if args.output_file else sys.stdout

    # cfg and the train flag are static (they select the model graph);
    # everything else is traced. One compilation per batch shape.
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)),
                      static_argnums=(3, 5))
    predict_fn = jax.jit(forward, static_argnums=(2, 4))

    mask = np.int32(2 ** cfg.hash_size_bits - 1)
    key = jax.random.PRNGKey(cfg.seed)
    n, t0 = 0, time.time()

    for chunk in pd.read_csv(args.input_file, dtype=np.int32,
                             chunksize=cfg.batch_size):
        x = chunk[cfg.features].values & mask
        y = chunk[cfg.label_feature].values.astype(np.float32)

        emb_rows = params['emb'][x]  # one gather, reused for predict + train

        # 1) predict on unseen examples (prequential evaluation output)
        logits = predict_fn(params['dense'], emb_rows, cfg, None, False)
        p = np.asarray(jax.nn.sigmoid(logits))
        np.savetxt(out, np.column_stack([y, p]), fmt='%g', delimiter=',')

        # 2) one gradient step on the same batch
        key, dk = jax.random.split(key)
        _, (g_dense, g_emb) = grad_fn(params['dense'], emb_rows, y, cfg, dk, True)
        params = opt.step(params, g_dense, x, g_emb)

        n += len(y)
        if (n // cfg.batch_size) % 100 == 0:
            print(f'{n} examples, {n / (time.time() - t0):.0f} ex/s',
                  file=sys.stderr)

    if args.output_file:
        out.close()
    print(f'done: {n} examples in {time.time() - t0:.1f}s', file=sys.stderr)
    return params


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str, help='JSON model specification')
    parser.add_argument('input_file', type=str, help='CSV training data')
    parser.add_argument('--output_file', type=str, default='',
                        help='label,prediction CSV (default: stdout)')
    parser.add_argument('--save', type=str, default='',
                        help='pickle trained parameters to this file')
    parser.add_argument('--load', type=str, default='',
                        help='initialize parameters from this pickle file')
    args = parser.parse_args()

    with open(args.model_file) as f:
        cfg = Config(**json.load(f))
    cfg.seed = getattr(cfg, 'seed', 1024)  # fixed default seed, overridable in JSON

    set_features(args, cfg)

    if args.load:
        with open(args.load, 'rb') as f:
            params = pickle.load(f)
    else:
        params = init_params(cfg, jax.random.PRNGKey(cfg.seed))

    opt = make_optimizer(cfg)
    opt.init(params)

    params = train_model(params, opt, args, cfg)

    if args.save:
        with open(args.save, 'wb') as f:
            pickle.dump(jax.device_get(params), f)


if __name__ == '__main__':
    run()

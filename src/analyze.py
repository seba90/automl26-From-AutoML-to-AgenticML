"""Evaluation under the hackathon protocol: AUC over sliding 20,000-instance
windows of the prequential prediction stream, reported as avg / median / min /
max / std, plus overall AUC and log loss.

Usage:
    python src/analyze.py preds.csv [--window 20000] [--stride 20000]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def windowed_auc(y, p, window, stride):
    aucs, skipped = [], 0
    for start in range(0, len(y) - window + 1, stride):
        yw = y[start:start + window]
        if yw.min() == yw.max():  # single-class window: AUC undefined
            skipped += 1
            continue
        aucs.append(roc_auc_score(yw, p[start:start + window]))
    return np.array(aucs), skipped


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_file', type=str,
                        help='CSV with two columns: label,prediction (no header)')
    parser.add_argument('--window', type=int, default=20000)
    parser.add_argument('--stride', type=int, default=0,
                        help='window step; 0 (default) = same as --window')
    args = parser.parse_args()
    args.stride = args.stride or args.window

    df = pd.read_csv(args.predictions_file, header=None, dtype=np.float32,
                     names=['label', 'prediction'])
    y, p = df.label.values, df.prediction.values

    print(f'examples: {len(y)}')
    print(f'overall AUC: {roc_auc_score(y, p):.6f}')
    print(f'overall log loss: {log_loss(y, p):.6f}')

    aucs, skipped = windowed_auc(y, p, args.window, args.stride)
    if len(aucs) == 0:
        print(f'windowed AUC: not enough data for a {args.window}-instance window')
        return
    print(f'windowed AUC ({args.window} x {len(aucs)} windows'
          + (f', {skipped} skipped' if skipped else '') + '):')
    print(f'  avg:    {aucs.mean():.6f}')
    print(f'  median: {np.median(aucs):.6f}')
    print(f'  min:    {aucs.min():.6f}')
    print(f'  max:    {aucs.max():.6f}')
    print(f'  std:    {aucs.std():.6f}')


if __name__ == '__main__':
    run()

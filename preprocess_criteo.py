"""Convert raw Criteo (train.txt, tab-separated) into the framework's hashed CSV.

Raw format, one example per line, no header:
    label \\t I1..I13 (ints, may be empty/negative) \\t C1..C26 (hex strings, may be empty)

Preprocessing (follows the DCN2 paper):
  * continuous features: log-transform, discretized as floor(ln(v)^2) for v > 2,
    kept as-is for v <= 2, 'na' when missing -- then hashed like a categorical
  * categorical features: hashed directly (murmur3), NO pruning of rare values
  * every token is hashed as "<column>|<value>" into 31 bits; train.py masks
    down to the configured hash_size_bits at load time, so ONE preprocessed
    file serves every hash-space size in a sweep

Deterministic splits (everyone must train on byte-identical files):
    full stream : python preprocess_criteo.py train.txt criteo_full.csv
    5% sample   : python preprocess_criteo.py train.txt criteo_05.csv --sample-mod 20
    1% sample   : python preprocess_criteo.py train.txt criteo_01.csv --sample-mod 100
Sampling keeps every k-th row (row_index % k == 0), preserving temporal order.
"""

import argparse
import math
import sys
import time
from functools import lru_cache

import mmh3

N_NUM, N_CAT = 13, 26
HASH_SEED = 42          # fixed so every team produces identical files
INT31 = 0x7FFFFFFF      # keep hashes positive int32; mask further at train time


@lru_cache(maxsize=2 ** 20)
def hash_token(token: str) -> int:
    return mmh3.hash(token, HASH_SEED, signed=False) & INT31


def numeric_bucket(raw: str) -> str:
    if raw == '':
        return 'na'
    v = int(raw)
    if v > 2:
        return str(int(math.floor(math.log(v) ** 2)))
    return str(v)  # small and negative values kept as their own buckets


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='raw Criteo train.txt (TSV)')
    parser.add_argument('output_file', type=str, help='hashed CSV output')
    parser.add_argument('--sample-mod', type=int, default=1,
                        help='keep rows where row_index %% k == 0 (default 1 = all)')
    parser.add_argument('--max-rows', type=int, default=0,
                        help='stop after writing this many rows (0 = no limit)')
    args = parser.parse_args()

    header = (['label']
              + [f'num_{i:02d}' for i in range(1, N_NUM + 1)]
              + [f'cat_{i:02d}' for i in range(1, N_CAT + 1)])

    n_in = n_out = 0
    t0 = time.time()
    with open(args.input_file) as fin, open(args.output_file, 'w') as fout:
        fout.write(','.join(header) + '\n')
        for line in fin:
            row_idx, n_in = n_in, n_in + 1
            if row_idx % args.sample_mod != 0:
                continue

            fields = line.rstrip('\n').split('\t')
            if len(fields) != 1 + N_NUM + N_CAT:
                print(f'skipping malformed line {row_idx} '
                      f'({len(fields)} fields)', file=sys.stderr)
                continue

            cols = [fields[0]]
            for i in range(N_NUM):
                cols.append(str(hash_token(f'num_{i}|{numeric_bucket(fields[1 + i])}')))
            for i in range(N_CAT):
                v = fields[1 + N_NUM + i] or 'na'
                cols.append(str(hash_token(f'cat_{i}|{v}')))
            fout.write(','.join(cols) + '\n')

            n_out += 1
            if n_out % 1_000_000 == 0:
                print(f'{n_out} rows written ({n_in / (time.time() - t0):.0f} '
                      f'lines/s)', file=sys.stderr)
            if args.max_rows and n_out >= args.max_rows:
                break

    print(f'done: read {n_in} lines, wrote {n_out} rows '
          f'in {time.time() - t0:.1f}s', file=sys.stderr)


if __name__ == '__main__':
    run()

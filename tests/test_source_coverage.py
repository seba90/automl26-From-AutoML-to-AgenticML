import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import analyze
import bayes_search
import experiment
import preprocess_criteo
import train


def test_windowed_auc_skips_single_class_windows():
    labels = np.array([0, 1, 0, 0, 1, 0])
    predictions = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])

    aucs, skipped = analyze.windowed_auc(labels, predictions, window=2, stride=2)

    np.testing.assert_allclose(aucs, [1.0, 1.0])
    assert skipped == 1


def test_preprocessor_buckets_and_hashes_tokens():
    assert preprocess_criteo.numeric_bucket('') == 'na'
    assert preprocess_criteo.numeric_bucket('-1') == '-1'
    assert preprocess_criteo.numeric_bucket('2') == '2'
    assert preprocess_criteo.numeric_bucket('3') == '1'
    assert preprocess_criteo.hash_token('cat_0|value') == preprocess_criteo.hash_token(
        'cat_0|value')


def test_preprocessor_writes_sampled_gzip_csv(tmp_path, monkeypatch):
    raw_path = tmp_path / 'raw.tsv'
    output_path = tmp_path / 'sample.csv.gz'
    rows = []
    for label in ('0', '1', '0'):
        rows.append('\t'.join([label] + ['1'] * preprocess_criteo.N_NUM
                              + ['category'] * preprocess_criteo.N_CAT))
    raw_path.write_text('\n'.join(rows) + '\n')
    monkeypatch.setattr(
        'sys.argv',
        ['preprocess_criteo.py', str(raw_path), str(output_path), '--sample-mod', '2'],
    )

    preprocess_criteo.run()

    with gzip.open(output_path, 'rt', newline='') as stream:
        written = list(csv.reader(stream))
    assert len(written) == 3
    assert written[0][0] == 'label'
    assert [row[0] for row in written[1:]] == ['0', '0']
    assert len(written[0]) == 1 + preprocess_criteo.N_NUM + preprocess_criteo.N_CAT


@pytest.mark.parametrize('compressed', [False, True])
def test_train_set_features_reads_csv_header(tmp_path, compressed):
    csv_path = tmp_path / ('features.csv.gz' if compressed else 'features.csv')
    opener = gzip.open if compressed else open
    with opener(csv_path, 'wt') as stream:
        stream.write('label,num_01,cat_01\n')
    cfg = SimpleNamespace(features=['_all'], label_feature='label')

    train.set_features(SimpleNamespace(input_file=str(csv_path)), cfg)

    assert cfg.features == ['num_01', 'cat_01']
    assert cfg.n_features == 2


def test_train_set_features_rejects_missing_feature(tmp_path):
    csv_path = tmp_path / 'features.csv'
    csv_path.write_text('label,num_01\n')
    cfg = SimpleNamespace(features=['missing'], label_feature='label')

    with pytest.raises(ValueError, match='Feature missing not found'):
        train.set_features(SimpleNamespace(input_file=str(csv_path)), cfg)


def test_run_generation_records_results_and_keeps_best_config(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'EXPERIMENTS_DIR', tmp_path)
    metrics = {
        'n_examples': 10,
        'overall_auc': 0.7,
        'overall_log_loss': 0.5,
        'windowed_auc_avg': 0.7,
        'windowed_auc_median': 0.7,
        'windowed_auc_min': 0.7,
        'windowed_auc_max': 0.7,
        'windowed_auc_std': 0.0,
        'n_windows': 1,
    }
    results = iter([
        (metrics, b'first predictions'),
        ({**metrics, 'windowed_auc_avg': 0.6}, b'second predictions'),
    ])
    monkeypatch.setattr(experiment, 'train_and_evaluate', lambda *args: next(results))

    experiment.run_generation('sample', {'learning_rate': 0.1}, Path('data.csv'),
                              'baseline', 10, 10)
    experiment.run_generation('sample', {'learning_rate': 0.2}, Path('data.csv'),
                              'candidate', 10, 10)

    experiment_dir = tmp_path / 'sample'
    assert json.loads((experiment_dir / 'best.json').read_text()) == {
        'learning_rate': 0.1
    }
    with (experiment_dir / 'results.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['is_new_best'] for row in rows] == ['True', 'False']
    assert [row['generation_file'].split('_gen_')[-1] for row in rows] == ['0.json', '1.json']
    assert experiment.next_generation_num(experiment_dir / 'results.csv') == 2


def test_bayes_search_encodes_and_decodes_search_space():
    space = {
        'learning_rate': {'type': 'log', 'low': 1e-4, 'high': 1e-1},
        'batch_size': {'type': 'int', 'low': 100, 'high': 300},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
    }
    config = {'learning_rate': 1e-2, 'batch_size': 200, 'dropout': 0.25}

    encoded = bayes_search.encode(config, space)
    decoded = bayes_search.decode(encoded, space)

    np.testing.assert_allclose(encoded, [2 / 3, 0.5, 0.5])
    assert decoded['learning_rate'] == pytest.approx(config['learning_rate'])
    assert decoded['batch_size'] == config['batch_size']
    assert decoded['dropout'] == config['dropout']
    assert bayes_search.decode(np.array([-1.0, 2.0, 0.5]), space) == {
        'learning_rate': pytest.approx(1e-4),
        'batch_size': 300,
        'dropout': pytest.approx(0.25),
    }


def test_bayes_search_requires_custom_space_for_unknown_algorithm():
    with pytest.raises(SystemExit, match='no default search space'):
        bayes_search.load_space(None, 'unknown')

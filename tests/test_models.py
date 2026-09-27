"""Tests for the shipped models, and the pattern to copy for DCN2 layers.

Run with:  uv run pytest

Phase 2 asks for a shape test per component plus an initialization-invariant
test for collision-weighted lookups. `test_dcnv2_forward_shape` and
`test_sparse_step_touches_only_used_rows` below are the two shapes those
tests should take: assert on the contract, not on a golden number.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models import Config, emb_dim, forward, init_params, loss_fn
from optim import make_optimizer

BATCH, N_FEATURES = 8, 20


def make_cfg(algorithm, **overrides):
    """A small, fast config. hash_size_bits is tiny so tables stay cheap."""
    cfg = Config(
        algorithm=algorithm,
        batch_size=BATCH,
        hash_size_bits=8,
        dimensions=4,
        n_features=N_FEATURES,
        learning_rate=0.01,
        optimizer='adam',
        adam_beta1=0.9,
        init_weight_std=0.01,
        regularization=1e-6,
        n_hidden_layers=2,
        hidden_layer_size=16,
        dropout=0.0,
        cross_n_hidden=2,
        cross_projection_dim=6,
        seed=0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def gather(params, cfg, key):
    """Random feature indices -> gathered embedding rows, as train.py does."""
    n_rows = 2 ** cfg.hash_size_bits  # excludes the sentinel row
    idx = jax.random.randint(key, (BATCH, cfg.n_features), 0, n_rows)
    return np.asarray(idx), params['emb'][idx]


@pytest.mark.parametrize('algorithm', ['lr', 'nn', 'dcnv2'])
def test_forward_returns_one_logit_per_example(algorithm):
    cfg = make_cfg(algorithm)
    params = init_params(cfg, jax.random.PRNGKey(0))
    _, emb_rows = gather(params, cfg, jax.random.PRNGKey(1))

    logits = forward(params['dense'], emb_rows, cfg)

    assert logits.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(logits))


@pytest.mark.parametrize('algorithm', ['lr', 'nn', 'dcnv2'])
def test_table_has_sentinel_row_and_correct_dim(algorithm):
    cfg = make_cfg(algorithm)
    params = init_params(cfg, jax.random.PRNGKey(0))

    # +1 row is the sparse-update padding sentinel; see optim.py
    assert params['emb'].shape == (2 ** cfg.hash_size_bits + 1, emb_dim(cfg))


def test_dcnv2_forward_shape():
    """Cross stack must preserve the flattened width F*d it is handed."""
    cfg = make_cfg('dcnv2')
    params = init_params(cfg, jax.random.PRNGKey(0))
    flat_dim = cfg.n_features * cfg.dimensions

    for layer in params['dense']['cross']:
        assert layer['U'].shape == (flat_dim, cfg.cross_projection_dim)
        assert layer['V'].shape == (cfg.cross_projection_dim, flat_dim)


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match='Unknown algorithm'):
        init_params(make_cfg('does_not_exist'), jax.random.PRNGKey(0))


def test_loss_is_finite_and_gradients_flow():
    cfg = make_cfg('dcnv2')
    params = init_params(cfg, jax.random.PRNGKey(0))
    _, emb_rows = gather(params, cfg, jax.random.PRNGKey(1))
    y = jnp.float32(jax.random.bernoulli(jax.random.PRNGKey(2), 0.3, (BATCH,)))

    loss, (g_dense, g_emb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        params['dense'], emb_rows, y, cfg, None, True)

    assert jnp.isfinite(loss)
    assert g_emb.shape == emb_rows.shape  # batch-sized, not table-sized
    assert any(jnp.any(g != 0) for g in jax.tree.leaves(g_dense))


def test_sparse_step_touches_only_used_rows():
    """The core invariant of optim.py: rows absent from the batch must not move.

    If you add a layer that differentiates through a gather instead of through
    the gathered rows, this test is what catches it.
    """
    cfg = make_cfg('nn')
    params = init_params(cfg, jax.random.PRNGKey(0))
    opt = make_optimizer(cfg)
    opt.init(params)

    idx, emb_rows = gather(params, cfg, jax.random.PRNGKey(1))
    y = jnp.float32(jax.random.bernoulli(jax.random.PRNGKey(2), 0.3, (BATCH,)))
    before = np.asarray(params['emb']).copy()

    _, (g_dense, g_emb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        params['dense'], emb_rows, y, cfg, None, True)
    params = opt.step(params, g_dense, idx, g_emb)

    after = np.asarray(params['emb'])
    moved = np.where(np.any(before != after, axis=1))[0]
    assert set(moved.tolist()) <= set(np.unique(idx).tolist())


def test_lr_logit_is_the_sum_of_looked_up_weights():
    """Sanity anchor for the simplest model: logit == sum(rows) + global_bias."""
    cfg = make_cfg('lr')
    params = init_params(cfg, jax.random.PRNGKey(0))
    _, emb_rows = gather(params, cfg, jax.random.PRNGKey(1))

    logits = forward(params['dense'], emb_rows, cfg)

    expected = emb_rows.sum(axis=(1, 2)) + params['dense']['global_bias'][0]
    np.testing.assert_allclose(logits, expected, rtol=1e-6)

"""Model definitions: parameter initialization + forward pass, in pure JAX.

Every model is a dict of two pytrees:
  params['emb']   -- one big hashed embedding table, shape (2^hash_size_bits + 1, dim).
                     The extra last row is a sentinel used to pad sparse updates to a
                     fixed shape (see optim.py); no real feature ever maps to it.
                     Updated SPARSELY (only rows touched by a batch).
  params['dense'] -- everything else (MLP / cross weights, global bias).
                     Updated densely.

The forward pass takes the already-gathered embedding rows (batch, n_features, dim)
instead of raw indices. This is deliberate: differentiating through a gather gives a
dense table-sized gradient, while differentiating w.r.t. the gathered rows gives a
(batch, n_features, dim) gradient we can scatter back to only the touched rows
(see optim.py for the sparse scatter step).
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


class Config(SimpleNamespace):
    """Model config. Hashable by identity so it can be a static jit argument
    (the config selects the model graph; it never changes during training)."""
    __hash__ = object.__hash__
    __eq__ = object.__eq__


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _glorot(key, shape):
    std = np.sqrt(2.0 / (shape[0] + shape[1]))
    return std * jax.random.normal(key, shape, dtype=jnp.float32)


def _truncated_normal(key, shape, std=0.05):
    return std * jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype=jnp.float32)


def emb_dim(cfg) -> int:
    """Embedding dimension of the single table used by the model."""
    return 1 if cfg.algorithm == 'lr' else cfg.dimensions + (1 if cfg.algorithm == 'dcn2' else 0)


def init_params(cfg, key):
    """Build the parameter pytree for cfg.algorithm ('lr', 'nn', 'dcnv2', 'dcn2')."""
    n_rows = 2 ** cfg.hash_size_bits + 1  # +1: sentinel padding row (optim.py)
    d = emb_dim(cfg)
    keys = jax.random.split(key, 64)
    ki = iter(range(64))

    emb = cfg.init_weight_std * jax.random.normal(
        keys[next(ki)], (n_rows, d), dtype=jnp.float32)
    if cfg.algorithm == 'dcn2':
        emb = emb.at[:, -1].set(1.0)

    dense = {'global_bias': jnp.zeros((1,), dtype=jnp.float32)}
    # Width the dense layers see. If you widen the table with extra per-row
    # columns that forward() consumes rather than feeds through (e.g. a
    # collision-weight column), set this from the surviving width instead --
    # otherwise the mismatch surfaces as a dot_general shape error inside
    # _cross_forward, far from its cause.
    flat_dim = cfg.n_features * (d - 1 if cfg.algorithm == 'dcn2' else d)

    if cfg.algorithm == 'lr':
        pass  # linear model: logit is just the sum of 1-dim embedding weights

    elif cfg.algorithm == 'nn':
        dense['mlp'] = _init_mlp(cfg, keys, ki, flat_dim)

    elif cfg.algorithm == 'dcnv2':
        # Stacked DCN-v2: cross layers on the flattened embeddings, then an MLP.
        cross = []
        for _ in range(cfg.cross_n_hidden):
            cross.append({
                'U': _truncated_normal(keys[next(ki)], (flat_dim, cfg.cross_projection_dim)),
                'bu': jnp.zeros((cfg.cross_projection_dim,), dtype=jnp.float32),
                'V': _truncated_normal(keys[next(ki)], (cfg.cross_projection_dim, flat_dim)),
                'bv': jnp.zeros((flat_dim,), dtype=jnp.float32),
            })
        dense['cross'] = cross
        dense['mlp'] = _init_mlp(cfg, keys, ki, flat_dim)

    elif cfg.algorithm == 'dcn2':
        dense['onlydense'] = []
        for _ in range(cfg.cross_n_hidden):
            dense['onlydense'].append({
                'W': _glorot(keys[next(ki)], (flat_dim, flat_dim)),
                'b': jnp.zeros((flat_dim,), dtype=jnp.float32),
                'phi': jnp.ones((1,), dtype=jnp.float32),
            })
        n_pairs = cfg.n_features * (cfg.n_features - 1) // 2
        dense['simlayer'] = {'weight': jnp.zeros((n_pairs,), dtype=jnp.float32)}
        dense['mlp'] = _init_mlp(cfg, keys, ki, flat_dim)

    else:
        raise ValueError(f'Unknown algorithm: {cfg.algorithm}')

    return {'emb': emb, 'dense': dense}


def _init_mlp(cfg, keys, ki, in_dim):
    layers = []
    for _ in range(cfg.n_hidden_layers):
        layers.append({'W': _glorot(keys[next(ki)], (in_dim, cfg.hidden_layer_size)),
                       'b': jnp.zeros((cfg.hidden_layer_size,), dtype=jnp.float32)})
        in_dim = cfg.hidden_layer_size
    # final projection to a single logit; no bias here, global_bias covers it
    layers.append({'W': _glorot(keys[next(ki)], (in_dim, 1))})
    return layers


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _mlp_forward(layers, x, cfg, dropout_key, train):
    *hidden, final = layers
    for i, layer in enumerate(hidden):
        x = jax.nn.relu(x @ layer['W'] + layer['b'])
        if train and cfg.dropout > 0.0:
            keep = 1.0 - cfg.dropout
            mask = jax.random.bernoulli(jax.random.fold_in(dropout_key, i), keep, x.shape)
            x = jnp.where(mask, x / keep, 0.0)
    return (x @ final['W']).reshape(-1)


def _cross_forward(cross_layers, x, diag_scale=0.9):
    # x_{l+1} = x0 * (V @ relu(U @ x + bu) + bv + diag_scale * x) + x
    # Low-rank projected Cross layer: U/V factor the (F*d, F*d) weight through
    # cross_projection_dim, which is what makes depth affordable at F*d ~ 300+.
    x0 = x
    for layer in cross_layers:
        proj = jax.nn.relu(x @ layer['U'] + layer['bu'])
        out = proj @ layer['V'] + layer['bv'] + diag_scale * x
        x = x0 * out + x
    return x


def _onlydense_forward(layers, x):
    for layer in layers:
        x = jax.nn.relu(x @ layer['W'] + layer['b']) * x * layer['phi']
    return x


def forward(dense, emb_rows, cfg, dropout_key=None, train=False):
    """Compute logits from gathered embedding rows (batch, n_features, dim)."""
    if cfg.algorithm == 'lr':
        logit = emb_rows.sum(axis=(1, 2))
    else:
        if cfg.algorithm == 'dcn2':
            values = emb_rows[..., :-1] * emb_rows[..., -1:]
            x = values.reshape(values.shape[0], -1)
            pairwise = []
            for i in range(values.shape[1]):
                for j in range(i + 1, values.shape[1]):
                    pairwise.append(jnp.sum(values[:, i] * values[:, j], axis=1))
            sim = jnp.stack(pairwise, axis=1) @ dense['simlayer']['weight']
            x = _onlydense_forward(dense['onlydense'], x)
            logit = _mlp_forward(dense['mlp'], x, cfg, dropout_key, train) + sim
            return logit + dense['global_bias'][0]

        x = emb_rows.reshape(emb_rows.shape[0], -1)  # (batch, F*d)
        if cfg.algorithm == 'dcnv2':
            x = _cross_forward(dense['cross'], x)
        logit = _mlp_forward(dense['mlp'], x, cfg, dropout_key, train)
    return logit + dense['global_bias'][0]


def loss_fn(dense, emb_rows, y, cfg, dropout_key=None, train=True):
    """Mean binary cross-entropy from logits + L2 on the *used* embedding rows.

    Penalizing only the gathered rows (not the whole table) keeps the
    embedding gradient sparse; an L2 term over all 2^bits rows would make
    every step dense and defeat the sparse optimizer.
    """
    logits = forward(dense, emb_rows, cfg, dropout_key, train)
    bce = -(y * jax.nn.log_sigmoid(logits) + (1.0 - y) * jax.nn.log_sigmoid(-logits))
    reg = cfg.regularization * jnp.sum(jnp.square(emb_rows)) / emb_rows.shape[0]
    return jnp.mean(bce) + reg

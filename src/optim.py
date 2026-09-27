"""Hand-rolled optimizers: dense Adam/SGD for the small parameter pytree,
row-sparse Adam/SGD for the embedding table.

Why sparse: a 2^23 x 16 table is ~134M floats; a dense Adam step reads/writes
param + m + v (~1.6 GB of memory traffic) every batch, even though a batch of
2500 x 39 features touches at most ~97k rows. Updating only the touched rows
is what makes a single-pass Criteo run tractable on a laptop.

Fixed-shape trick: the number of *unique* rows per batch varies, which would
force a new XLA compilation every step. We therefore pad the unique-index
array to a constant length (batch * n_features) using a sentinel row appended
to the table (index = 2^hash_size_bits, never produced by real features).
The sentinel receives zero gradient, so with Adam its parameter update is
exactly 0; only its (unused) momentum decays.

Lazy semantics (deliberate trade-off): bias correction uses the global step
count, and the momentum of untouched rows is frozen rather than decayed. A
rarely-seen row therefore resumes from the moment it last had gradient, which
is not what dense Adam would do -- the cost of never touching cold rows.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def make_optimizer(cfg):
    if cfg.optimizer == 'adam':
        return Adam(cfg.learning_rate, b1=getattr(cfg, 'adam_beta1', 0.9))
    elif cfg.optimizer == 'sgd':
        return SGD(cfg.learning_rate)
    raise ValueError(f'Unknown optimizer: {cfg.optimizer}')


def _pad_unique(idx, sentinel):
    """Host-side np.unique, padded to fixed length len(idx) with the sentinel.

    Returns (uniq_padded (P,), inverse (P,)) where P = idx.size. Padding with a
    dedicated sentinel (instead of repeating a real row) keeps scatter targets
    duplicate-free, so .at[...].set/.add stay deterministic.
    """
    flat = np.asarray(idx).reshape(-1)
    uniq, inverse = np.unique(flat, return_inverse=True)
    padded = np.full(flat.shape[0], sentinel, dtype=flat.dtype)
    padded[:uniq.shape[0]] = uniq
    return jnp.asarray(padded), jnp.asarray(inverse)


@jax.jit
def _adam_dense(params, m, v, grads, t, hyper):
    lr, b1, b2, eps = hyper

    def upd(p, m_, v_, g):
        m_ = b1 * m_ + (1 - b1) * g
        v_ = b2 * v_ + (1 - b2) * jnp.square(g)
        mhat = m_ / (1 - b1 ** t)
        vhat = v_ / (1 - b2 ** t)
        return p - lr * mhat / (jnp.sqrt(vhat) + eps), m_, v_

    out = jax.tree.map(upd, params, m, v, grads)
    unzip = lambda i: jax.tree.map(lambda tup: tup[i], out,
                                   is_leaf=lambda x: isinstance(x, tuple))
    return unzip(0), unzip(1), unzip(2)


@jax.jit
def _adam_sparse(table, m, v, uniq, inverse, grad_rows, t, hyper):
    lr, b1, b2, eps = hyper
    # Sum duplicate-row gradients within the batch; segments beyond the true
    # unique count stay zero, so sentinel-padded rows get zero gradient.
    g = jax.ops.segment_sum(grad_rows.reshape(-1, grad_rows.shape[-1]),
                            inverse, num_segments=uniq.shape[0])
    m_u = b1 * m[uniq] + (1 - b1) * g
    v_u = b2 * v[uniq] + (1 - b2) * jnp.square(g)
    mhat = m_u / (1 - b1 ** t)
    vhat = v_u / (1 - b2 ** t)
    table = table.at[uniq].add(-lr * mhat / (jnp.sqrt(vhat) + eps))
    return table, m.at[uniq].set(m_u), v.at[uniq].set(v_u)


@jax.jit
def _sgd_sparse(table, uniq, inverse, grad_rows, lr):
    g = jax.ops.segment_sum(grad_rows.reshape(-1, grad_rows.shape[-1]),
                            inverse, num_segments=uniq.shape[0])
    return table.at[uniq].add(-lr * g)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.hyper = (lr, b1, b2, eps)
        self.t = 0

    def init(self, params):
        self.sentinel = params['emb'].shape[0] - 1
        self.dense_m = jax.tree.map(jnp.zeros_like, params['dense'])
        self.dense_v = jax.tree.map(jnp.zeros_like, params['dense'])
        self.emb_m = jnp.zeros_like(params['emb'])
        self.emb_v = jnp.zeros_like(params['emb'])

    def step(self, params, g_dense, idx, g_emb_rows):
        """idx: (batch, F) int rows into the table; g_emb_rows: matching grads."""
        self.t += 1
        t = jnp.float32(self.t)
        params['dense'], self.dense_m, self.dense_v = _adam_dense(
            params['dense'], self.dense_m, self.dense_v, g_dense, t, self.hyper)
        uniq, inverse = _pad_unique(idx, self.sentinel)
        params['emb'], self.emb_m, self.emb_v = _adam_sparse(
            params['emb'], self.emb_m, self.emb_v, uniq, inverse,
            g_emb_rows, t, self.hyper)
        return params


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def init(self, params):
        self.sentinel = params['emb'].shape[0] - 1

    def step(self, params, g_dense, idx, g_emb_rows):
        params['dense'] = jax.tree.map(lambda p, g: p - self.lr * g,
                                       params['dense'], g_dense)
        uniq, inverse = _pad_unique(idx, self.sentinel)
        params['emb'] = _sgd_sparse(params['emb'], uniq, inverse,
                                    g_emb_rows, self.lr)
        return params

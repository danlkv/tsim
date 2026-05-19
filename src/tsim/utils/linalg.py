"""Linear algebra utilities for GF(2) operations."""

import os

import jax.numpy as jnp
import numpy as np
from jax import Array

try:
    from cust_jax import (
        matmul_gf2_csr_ffi as _cust_matmul_gf2_csr_spsp,
        matmul_gf2_csr_spdn_ffi as _cust_matmul_gf2_csr_spdn,
        pack_params_transposed as _cust_pack_params_transposed,
        pack_params_dense_packed as _cust_pack_params_dense_packed,
        upload_static_csr as _cust_upload_static_csr,
        upload_static_packed as _cust_upload_static_packed,
    )
    _CUST_AVAILABLE = True
except Exception:
    _CUST_AVAILABLE = False

_CUST_BACKENDS = ("cust", "cust_spdn")
_CUST_MIN_GT = int(os.environ.get("TSIM_GF2MM_MIN_GT", "0"))


def _current_backend() -> str:
    return os.environ.get("TSIM_GF2MM_BACKEND", "")


class _CustCsr:
    """Static side data uploaded to device at compile time.

    Holds whichever representation the active backend needs (SpSp: rowoff/colidx;
    SpDn: bit-packed). Hashed by identity so jit caches by instance.
    """

    __slots__ = ("backend", "rowoff_d", "colidx_d", "packed_d", "n_pad")

    def __init__(self, backend, rowoff_d, colidx_d, packed_d, n_pad):
        self.backend = backend
        self.rowoff_d = rowoff_d
        self.colidx_d = colidx_d
        self.packed_d = packed_d
        self.n_pad = n_pad

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def _use_cust_backend() -> bool:
    return _CUST_AVAILABLE and _current_backend() in _CUST_BACKENDS


def build_params_csr(params):
    """Compile-time helper: pack params^T and upload to device.

    Returns None when cust_jax isn't importable, backend isn't a cust one,
    params is empty along (G, T), or G·T is below TSIM_GF2MM_MIN_GT.
    """
    if not _CUST_AVAILABLE:
        return None
    if params.ndim != 3:
        return None
    G, T, _ = params.shape
    if G * T == 0:
        return None
    if _CUST_MIN_GT > 0 and G * T < _CUST_MIN_GT:
        return None
    backend = _current_backend()
    if backend not in _CUST_BACKENDS:
        return None
    params_u8 = np.asarray(params, dtype=np.uint8)
    if backend == "cust":
        rowoff_h, colidx_h, n_pad = _cust_pack_params_transposed(params_u8)
        rowoff_d, colidx_d = _cust_upload_static_csr(rowoff_h, colidx_h)
        return _CustCsr("cust", rowoff_d, colidx_d, None, n_pad)
    packed_h, n_pad = _cust_pack_params_dense_packed(params_u8)
    packed_d = _cust_upload_static_packed(packed_h)
    return _CustCsr("cust_spdn", None, None, packed_d, n_pad)


def find_basis(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a set of binary vectors into a basis subset and a transformation matrix over GF(2).

    Given a set of vectors V, this function finds a maximal linearly independent subset B
    (the basis) and computes a transformation matrix T such that the original vectors can be
    reconstructed from the basis via matrix multiplication over GF(2):

    V = T @ B (mod 2)

    Args:
        vectors: Input binary vectors of shape `(N, D)`. Can be a list of lists or a numpy array.
                 Elements should be 0 or 1 (or convertible to them).

    Returns:
        A tuple `(basis, transform)` where:
            basis: The subset of independent vectors, shape `(K, D)`, where `K` is the rank.
            transform: The transformation matrix, shape `(N, K)`.

    """
    vecs = np.array(vectors, dtype=np.uint8)
    num_vectors, _ = vecs.shape

    basis_indices = []
    reduced_basis = []
    pivots = []
    basis_expansion = []
    t_rows = []

    for i in range(num_vectors):
        v = vecs[i].copy()
        coeffs = []

        for j, b in enumerate(reduced_basis):
            if v[pivots[j]]:
                v ^= b
                coeffs.append(j)

        is_independent = np.any(v)
        current_rank = len(basis_indices)
        new_size = current_rank + 1 if is_independent else current_rank

        # Compute dependency on existing basis vectors
        dep_sum = np.zeros(new_size, dtype=np.uint8)
        for idx in coeffs:
            e = basis_expansion[idx]
            dep_sum[: len(e)] ^= e

        if is_independent:
            basis_indices.append(i)
            reduced_basis.append(v)
            pivots.append(np.argmax(v))

            # Update basis expansion for the new reduced vector
            # reduced_v = v_original + sum(reduced_basis[c])
            # => reduced_v_expansion = e_new + sum(basis_expansion[c])
            dep_sum[current_rank] = 1
            basis_expansion.append(dep_sum)

            t_row = np.zeros(new_size, dtype=np.uint8)
            t_row[current_rank] = 1
            t_rows.append(t_row)
        else:
            # Dependent vector is the sum of basis expansions of reducing vectors
            t_rows.append(dep_sum)

    rank = len(basis_indices)
    transform = np.zeros((num_vectors, rank), dtype=np.uint8)
    for i, row in enumerate(t_rows):
        transform[i, : len(row)] = row

    return vecs[basis_indices], transform


def matmul_gf2(a: Array, b: Array) -> Array:
    """Compute binary dot products mod 2 as ``a_GTP x b_BP -> b_BGT``.

    Uses float32 matmul (integer matmul does not have BLAS support on CPU)
    then casts back to uint8.

    Args:
        a: Parameter bit-masks, shape ``(G, T, P)`` — G graphs, T terms, P parameters.
        b: Binary parameter values, shape ``(B, P)`` — B batch elements.

    Returns:
        Binary row-sums mod 2, shape ``(B, G, T)``.

    """
    G, T, _ = a.shape
    if G * T == 0:
        return jnp.zeros((b.shape[0], G, T), dtype=jnp.uint8)
    # NOTE: ``% 2`` must run on float32 — JAX's float→uint8 cast saturates at
    # 255 (it does not wrap mod 256), which would corrupt parity for inner
    # products with more than 255 set bits.
    sum_f32 = b.astype(jnp.float32) @ a.astype(jnp.float32).reshape(G * T, -1).T
    return (sum_f32.reshape(-1, G, T) % 2).astype(jnp.uint8)

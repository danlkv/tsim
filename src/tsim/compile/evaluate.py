"""Evaluation of compiled scalar graphs using exact arithmetic."""

import functools
import os

import jax
import jax.numpy as jnp
from jax import Array, lax

from tsim.compile.compile import CompiledScalarGraphs
from tsim.core.exact_scalar import ComplexScalarArray, ExactScalarArray, _scalar_mul_with_power_split
from tsim.utils.linalg import _use_cust_backend

_USE_COMPLEX128 = os.environ.get("TSIM_SCALAR_BACKEND", "") == "complex128"

# H2: c128 register-tiled prod kernels (counterpart of C1 for complex backend).
# Defaults to enabled when _USE_COMPLEX128 and the FFI module is importable;
# can be force-off via TSIM_SCALAR_BACKEND_CUST_KERNEL=0.
_HC_CUST_KERNEL = os.environ.get(
    "TSIM_SCALAR_BACKEND_CUST_KERNEL", "1" if _USE_COMPLEX128 else "0"
) in ("1", "true", "True")
_HC_CUST_MIN_GT = int(os.environ.get("TSIM_SCALAR_BACKEND_CUST_MIN_GT", "0"))
if _USE_COMPLEX128 and _HC_CUST_KERNEL:
    try:
        from cust_jax import (
            node_phases_scalar_prod_c128_ffi as _cust_node_phases_c128,
            phase_pairs_scalar_prod_c128_ffi as _cust_phase_pairs_c128,
        )
        _HC_CUST_AVAILABLE = True
    except Exception:
        _HC_CUST_AVAILABLE = False
else:
    _HC_CUST_AVAILABLE = False

# Feature flags ported from session-2026-05-13 work onto PR #78's restructured
# evaluate.py. terms.py was deleted in PR #78 and the term-family logic moved
# inline here, so the lazy/split/cust prod dispatch lives in evaluate.py too
# (was originally in terms.py).
_LAZY_TERM_VALS = os.environ.get("TSIM_TERM_VALS_LAZY", "") in ("1", "scan", "true", "True")
_LAZY_UNROLL = int(os.environ.get("TSIM_TERM_VALS_UNROLL", "4"))
_LAZY_SPLIT_COEFFS = os.environ.get(
    "TSIM_TERM_VALS_SPLIT_COEFFS", "") in ("1", "true", "True")
_LAZY_CUST_KERNEL = os.environ.get(
    "TSIM_TERM_VALS_CUST_KERNEL", "") in ("1", "true", "True")
_LAZY_CUST_MIN_GT = int(os.environ.get("TSIM_TERM_VALS_CUST_MIN_GT", "0"))
if _LAZY_CUST_KERNEL:
    try:
        from cust_jax import (
            node_phases_scalar_prod_ffi as _cust_node_phases_scalar_prod,
            phase_pairs_scalar_prod_ffi as _cust_phase_pairs_scalar_prod,
        )
        _LAZY_CUST_KERNEL_AVAILABLE = True
    except Exception:
        _LAZY_CUST_KERNEL_AVAILABLE = False
else:
    _LAZY_CUST_KERNEL_AVAILABLE = False

# Pre-computed exact scalars for phase values, for powers of omega = e^(i*pi/4)
_UNIT_PHASES = jnp.array(
    [
        [1, 0, 0, 0],  # omega^0 = 1
        [0, 1, 0, 0],  # omega^1
        [0, 0, 1, 0],  # omega^2 = i
        [0, 0, 0, -1],  # omega^3
        [-1, 0, 0, 0],  # omega^4 = -1
        [0, -1, 0, 0],  # omega^5
        [0, 0, -1, 0],  # omega^6 = -i
        [0, 0, 0, 1],  # omega^7
    ],
    dtype=jnp.int32,
)

# Lookup table for exact scalars (1 + omega^k)
_ONE_PLUS_PHASES = _UNIT_PHASES.at[:, 0].add(1)

_IDENTITY = jnp.array([1, 0, 0, 0], dtype=jnp.int32)

# Complex backend (H): native complex128 lookup tables. ω = e^(iπ/4).
_OMEGA_K_C128 = jnp.exp(1j * jnp.pi / 4 * jnp.arange(8)).astype(jnp.complex128)
_ONE_PLUS_OMEGA_K_C128 = (1.0 + _OMEGA_K_C128).astype(jnp.complex128)


def _make_summands(coeffs: Array, power: Array | None = None):
    """Construct an ExactScalarArray or ComplexScalarArray based on the active scalar backend."""
    if _USE_COMPLEX128:
        return ComplexScalarArray.from_coeffs(coeffs, power)
    return ExactScalarArray(coeffs) if power is None else ExactScalarArray(coeffs, power)


def _lazy_type_a_prod_complex(phase_idx: Array, counts: Array) -> ComplexScalarArray:
    """Complex-backend lazy Type-A product. (B,G) carry, scan over T."""
    B, G, T = phase_idx.shape
    if T == 0:
        return ComplexScalarArray(jnp.ones((B, G), dtype=jnp.complex128))

    if _HC_CUST_AVAILABLE and G * T >= _HC_CUST_MIN_GT:
        phase_gtb = jnp.transpose(phase_idx.astype(jnp.uint8), (1, 2, 0))
        counts_i32 = counts.astype(jnp.int32)
        out_gb = _cust_node_phases_c128(phase_gtb, counts_i32, B=B, G=G, T=T)
        return ComplexScalarArray(jnp.transpose(out_gb, (1, 0)))

    def step(carry, t):
        pi_t = phase_idx[:, :, t]
        tv_t = _ONE_PLUS_OMEGA_K_C128[pi_t]
        mask_g = counts > t
        tv_t = jnp.where(mask_g[None, :], tv_t, jnp.complex128(1.0))
        return carry * tv_t, None

    init = jnp.ones((B, G), dtype=jnp.complex128)
    # unroll=True: fully unroll the T-axis scan so XLA emits one fused kernel
    # with the (B, G) c128 carry living in registers, avoiding the inter-step
    # HBM round-trip that unroll=4 was paying. T is static (compile-time), so
    # this only blows up the trace per circuit, not per shot.
    final, _ = lax.scan(step, init, jnp.arange(T), unroll=True)
    return ComplexScalarArray(final)


def _lazy_type_d_prod_complex(alpha: Array, beta: Array, counts: Array) -> ComplexScalarArray:
    """Complex-backend lazy Type-D product. (B,G) carry, scan over T."""
    B, G, T = alpha.shape
    if T == 0:
        return ComplexScalarArray(jnp.ones((B, G), dtype=jnp.complex128))

    if _HC_CUST_AVAILABLE and G * T >= _HC_CUST_MIN_GT:
        alpha_gtb = jnp.transpose(alpha.astype(jnp.uint8), (1, 2, 0))
        beta_gtb = jnp.transpose(beta.astype(jnp.uint8), (1, 2, 0))
        counts_i32 = counts.astype(jnp.int32)
        out_gb = _cust_phase_pairs_c128(alpha_gtb, beta_gtb, counts_i32, B=B, G=G, T=T)
        return ComplexScalarArray(jnp.transpose(out_gb, (1, 0)))

    def step(carry, t):
        a_t = alpha[:, :, t]
        b_t = beta[:, :, t]
        g_t = (a_t + b_t) % 8
        tv_t = 1.0 + _OMEGA_K_C128[a_t] + _OMEGA_K_C128[b_t] - _OMEGA_K_C128[g_t]
        mask_g = counts > t
        tv_t = jnp.where(mask_g[None, :], tv_t, jnp.complex128(1.0))
        return carry * tv_t, None

    init = jnp.ones((B, G), dtype=jnp.complex128)
    # unroll=True: fully unroll the T-axis scan so XLA emits one fused kernel
    # with the (B, G) c128 carry living in registers, avoiding the inter-step
    # HBM round-trip that unroll=4 was paying. T is static (compile-time), so
    # this only blows up the trace per circuit, not per shot.
    final, _ = lax.scan(step, init, jnp.arange(T), unroll=True)
    return ComplexScalarArray(final)


# Per-coordinate views of the lookup tables. Used by the split-coeffs variants
# of the lazy TYPE A / TYPE D paths (TSIM_TERM_VALS_SPLIT_COEFFS=1).
_ONE_PLUS_PHASES_C = tuple(_ONE_PLUS_PHASES[:, i] for i in range(4))
_UNIT_PHASES_C = tuple(_UNIT_PHASES[:, i] for i in range(4))
_ID_C = (jnp.int32(1), jnp.int32(0), jnp.int32(0), jnp.int32(0))


def _lazy_type_a_prod(phase_idx: Array, counts: Array) -> ExactScalarArray:
    """TYPE A (NodePhases) lazy product over T.

    Replaces the materialised (B, G, T, 4) term_vals_a_exact tensor with a
    lax.scan whose live state is the (B, G, 4) carry — never materialises the
    T axis. Counterpart of terms.py::_lazy_node_phases_prod from session-main.
    """
    B, G, T = phase_idx.shape
    if T == 0:
        return ExactScalarArray(
            jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
            jnp.zeros((B, G), dtype=jnp.int32),
        )
    if _LAZY_CUST_KERNEL_AVAILABLE and G * T >= _LAZY_CUST_MIN_GT:
        return _cust_node_phases_prod(phase_idx, counts)
    if _LAZY_SPLIT_COEFFS:
        return _lazy_type_a_prod_split(phase_idx, counts)

    def step(carry, t):
        carry_coeffs, carry_power = carry
        pi_t = phase_idx[:, :, t]
        tv_t = _ONE_PLUS_PHASES[pi_t]
        mask_g = counts > t
        tv_t = jnp.where(mask_g[None, :, None], tv_t, _IDENTITY)
        from tsim.core.exact_scalar import _scalar_mul_with_power
        new_power, new_coeffs = _scalar_mul_with_power(
            (carry_power, carry_coeffs),
            (jnp.zeros((), dtype=jnp.int32), tv_t),
        )
        return (new_coeffs, new_power), None

    init = (
        jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
        jnp.zeros((B, G), dtype=jnp.int32),
    )
    (final_coeffs, final_power), _ = lax.scan(step, init, jnp.arange(T), unroll=_LAZY_UNROLL)
    return ExactScalarArray(final_coeffs, final_power)


def _lazy_type_a_prod_split(phase_idx: Array, counts: Array) -> ExactScalarArray:
    """Split-coeffs variant: thread the 4 Z[ω] coefs as 5 separate (B,G) ints
    through the scan, eliminating the body/.../concatenate XLA fusion."""
    B, G, T = phase_idx.shape

    def step(carry, t):
        p, c0, c1, c2, c3 = carry
        pi_t = phase_idx[:, :, t]
        d0 = _ONE_PLUS_PHASES_C[0][pi_t]
        d1 = _ONE_PLUS_PHASES_C[1][pi_t]
        d2 = _ONE_PLUS_PHASES_C[2][pi_t]
        d3 = _ONE_PLUS_PHASES_C[3][pi_t]
        mask_g = counts > t
        d0 = jnp.where(mask_g[None, :], d0, _ID_C[0])
        d1 = jnp.where(mask_g[None, :], d1, _ID_C[1])
        d2 = jnp.where(mask_g[None, :], d2, _ID_C[2])
        d3 = jnp.where(mask_g[None, :], d3, _ID_C[3])
        new_p, n0, n1, n2, n3 = _scalar_mul_with_power_split(
            p, c0, c1, c2, c3,
            jnp.int32(0), d0, d1, d2, d3,
        )
        return (new_p, n0, n1, n2, n3), None

    init_p = jnp.zeros((B, G), dtype=jnp.int32)
    init_c0 = jnp.broadcast_to(_ID_C[0], (B, G)).astype(jnp.int32)
    init_c1 = jnp.broadcast_to(_ID_C[1], (B, G)).astype(jnp.int32)
    init_c2 = jnp.broadcast_to(_ID_C[2], (B, G)).astype(jnp.int32)
    init_c3 = jnp.broadcast_to(_ID_C[3], (B, G)).astype(jnp.int32)
    (final_p, c0, c1, c2, c3), _ = lax.scan(
        step, (init_p, init_c0, init_c1, init_c2, init_c3),
        jnp.arange(T), unroll=_LAZY_UNROLL,
    )
    final_coeffs = jnp.stack([c0, c1, c2, c3], axis=-1)
    return ExactScalarArray(final_coeffs, final_p)


def _lazy_type_d_prod(alpha: Array, beta: Array, counts: Array) -> ExactScalarArray:
    """TYPE D (PhasePairs) lazy product over T."""
    B, G, T = alpha.shape
    if T == 0:
        return ExactScalarArray(
            jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
            jnp.zeros((B, G), dtype=jnp.int32),
        )
    if _LAZY_CUST_KERNEL_AVAILABLE and G * T >= _LAZY_CUST_MIN_GT:
        return _cust_phase_pairs_prod(alpha, beta, counts)
    if _LAZY_SPLIT_COEFFS:
        return _lazy_type_d_prod_split(alpha, beta, counts)

    def step(carry, t):
        carry_coeffs, carry_power = carry
        a_t = alpha[:, :, t]
        b_t = beta[:, :, t]
        g_t = (a_t + b_t) % 8
        tv_t = (
            _IDENTITY + _UNIT_PHASES[a_t] + _UNIT_PHASES[b_t] - _UNIT_PHASES[g_t]
        )
        mask_g = counts > t
        tv_t = jnp.where(mask_g[None, :, None], tv_t, _IDENTITY)
        from tsim.core.exact_scalar import _scalar_mul_with_power
        new_power, new_coeffs = _scalar_mul_with_power(
            (carry_power, carry_coeffs),
            (jnp.zeros((), dtype=jnp.int32), tv_t),
        )
        return (new_coeffs, new_power), None

    init = (
        jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
        jnp.zeros((B, G), dtype=jnp.int32),
    )
    (final_coeffs, final_power), _ = lax.scan(step, init, jnp.arange(T), unroll=_LAZY_UNROLL)
    return ExactScalarArray(final_coeffs, final_power)


def _lazy_type_d_prod_split(alpha: Array, beta: Array, counts: Array) -> ExactScalarArray:
    B, G, T = alpha.shape

    def step(carry, t):
        p, c0, c1, c2, c3 = carry
        a_t = alpha[:, :, t]
        b_t = beta[:, :, t]
        g_t = (a_t + b_t) % 8
        d0 = _ID_C[0] + _UNIT_PHASES_C[0][a_t] + _UNIT_PHASES_C[0][b_t] - _UNIT_PHASES_C[0][g_t]
        d1 = _ID_C[1] + _UNIT_PHASES_C[1][a_t] + _UNIT_PHASES_C[1][b_t] - _UNIT_PHASES_C[1][g_t]
        d2 = _ID_C[2] + _UNIT_PHASES_C[2][a_t] + _UNIT_PHASES_C[2][b_t] - _UNIT_PHASES_C[2][g_t]
        d3 = _ID_C[3] + _UNIT_PHASES_C[3][a_t] + _UNIT_PHASES_C[3][b_t] - _UNIT_PHASES_C[3][g_t]
        mask_g = counts > t
        d0 = jnp.where(mask_g[None, :], d0, _ID_C[0])
        d1 = jnp.where(mask_g[None, :], d1, _ID_C[1])
        d2 = jnp.where(mask_g[None, :], d2, _ID_C[2])
        d3 = jnp.where(mask_g[None, :], d3, _ID_C[3])
        new_p, n0, n1, n2, n3 = _scalar_mul_with_power_split(
            p, c0, c1, c2, c3,
            jnp.int32(0), d0, d1, d2, d3,
        )
        return (new_p, n0, n1, n2, n3), None

    init_p = jnp.zeros((B, G), dtype=jnp.int32)
    init_c0 = jnp.broadcast_to(_ID_C[0], (B, G)).astype(jnp.int32)
    init_c1 = jnp.broadcast_to(_ID_C[1], (B, G)).astype(jnp.int32)
    init_c2 = jnp.broadcast_to(_ID_C[2], (B, G)).astype(jnp.int32)
    init_c3 = jnp.broadcast_to(_ID_C[3], (B, G)).astype(jnp.int32)
    (final_p, c0, c1, c2, c3), _ = lax.scan(
        step, (init_p, init_c0, init_c1, init_c2, init_c3),
        jnp.arange(T), unroll=_LAZY_UNROLL,
    )
    final_coeffs = jnp.stack([c0, c1, c2, c3], axis=-1)
    return ExactScalarArray(final_coeffs, final_p)


def _cust_node_phases_prod(phase_idx: Array, counts: Array) -> ExactScalarArray:
    """Cust kernel path: register-tiled scalar_prod_over_T."""
    B, G, T = phase_idx.shape
    if T == 0:
        return ExactScalarArray(
            jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
            jnp.zeros((B, G), dtype=jnp.int32),
        )
    phase_gtb = jnp.transpose(phase_idx.astype(jnp.uint8), (1, 2, 0))
    counts_i32 = counts.astype(jnp.int32)
    power_gb, coeffs_gcb = _cust_node_phases_scalar_prod(
        phase_gtb, counts_i32, B=B, G=G, T=T,
    )
    return ExactScalarArray(
        jnp.transpose(coeffs_gcb, (2, 0, 1)),
        jnp.transpose(power_gb, (1, 0)),
    )


def _cust_phase_pairs_prod(alpha: Array, beta: Array, counts: Array) -> ExactScalarArray:
    B, G, T = alpha.shape
    if T == 0:
        return ExactScalarArray(
            jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
            jnp.zeros((B, G), dtype=jnp.int32),
        )
    alpha_gtb = jnp.transpose(alpha.astype(jnp.uint8), (1, 2, 0))
    beta_gtb = jnp.transpose(beta.astype(jnp.uint8), (1, 2, 0))
    counts_i32 = counts.astype(jnp.int32)
    power_gb, coeffs_gcb = _cust_phase_pairs_scalar_prod(
        alpha_gtb, beta_gtb, counts_i32, B=B, G=G, T=T,
    )
    return ExactScalarArray(
        jnp.transpose(coeffs_gcb, (2, 0, 1)),
        jnp.transpose(power_gb, (1, 0)),
    )


def _matmul_gf2(a: Array, b: Array, csr=None) -> Array:
    """Compute binary dot products mod 2 as ``a_GTP x b_BP -> b_BGT``.

    Default path is float32 matmul cast to uint8 then mod 2. When ``csr`` is a
    precomputed ``_CustCsr`` and ``TSIM_GF2MM_BACKEND`` selects a cust backend,
    routes through cuStabilizer's SpSp/SpDn kernel via cust_jax. Output is
    bit-exact in either case.

    Args:
        a: Parameter bit-masks, shape ``(G, T, P)`` — G graphs, T terms, P parameters.
        b: Binary parameter values, shape ``(B, P)`` — B batch elements.
        csr: Optional precomputed CSR/packed view of ``a^T : (P, G·T_pad)``.

    Returns:
        Binary row-sums mod 2, shape ``(B, G, T)``.
    """
    G, T, _ = a.shape
    if G * T == 0:
        return jnp.zeros((b.shape[0], G, T), dtype=b.dtype)
    if csr is not None and _use_cust_backend():
        from cust_jax import matmul_gf2_csr_ffi, matmul_gf2_csr_spdn_ffi
        P = int(a.shape[-1])
        if csr.backend == "cust":
            return matmul_gf2_csr_ffi(
                b, B_rowoff_d=csr.rowoff_d, B_colidx_d=csr.colidx_d,
                G=G, T=T, P=P, n_pad=csr.n_pad,
            )
        if csr.backend == "cust_spdn":
            return matmul_gf2_csr_spdn_ffi(
                b, B_packed_d=csr.packed_d,
                G=G, T=T, P=P, n_pad=csr.n_pad,
            )
    return (b.astype(jnp.float32) @ a.astype(jnp.float32).reshape(G * T, -1).T).reshape(
        -1, G, T
    ).astype(jnp.uint8) % 2


@jax.jit
def evaluate(circuit: CompiledScalarGraphs, param_vals: Array) -> Array:
    """Evaluate compiled circuit with batched parameter values.

    Args:
        circuit: Compiled circuit representation
        param_vals: Binary parameter values (error bits + measurement/detector outcomes),
            shape (batch_size, n_params)

    Returns:
        A complex array of shape (batch_size,) containing the amplitudes of the provided
        circuit evaluated with the given binary parameter values.

    """
    # ====================================================================
    # TYPE A: Node Terms (1 + e^(i*alpha))
    # Padded values are masked to multiplicative identity.
    # ====================================================================
    # a_param_bits: (num_graphs, max_a, n_params), param_vals: (batch_size, n_params,)
    rowsum_a = _matmul_gf2(circuit.a_param_bits, param_vals, csr=circuit.a_param_csr)
    phase_idx_a = (4 * rowsum_a + circuit.a_const_phases) % 8

    if _USE_COMPLEX128:
        summands_a = _lazy_type_a_prod_complex(phase_idx_a, circuit.a_num_terms)
    elif _LAZY_TERM_VALS:
        summands_a = _lazy_type_a_prod(phase_idx_a, circuit.a_num_terms)
    else:
        term_vals_a_exact = _ONE_PLUS_PHASES[phase_idx_a]
        a_mask = (
            jnp.arange(circuit.a_const_phases.shape[1])[None, :]
            < circuit.a_num_terms[:, None]
        )
        term_vals_a_exact = jnp.where(a_mask[..., None], term_vals_a_exact, _IDENTITY)
        term_vals_a = ExactScalarArray(term_vals_a_exact)
        summands_a = term_vals_a.prod(axis=-1)

    # ====================================================================
    # TYPE B: Half-Pi Terms (e^(i*beta))
    # Padded values are 0, so they don't affect the sum.
    # ====================================================================
    rowsum_b = _matmul_gf2(circuit.b_param_bits, param_vals, csr=circuit.b_param_csr)
    phase_idx_b = (rowsum_b * circuit.b_term_types) % 8

    sum_phases_b = jnp.sum(phase_idx_b, axis=-1) % 8

    summands_b_exact = _UNIT_PHASES[sum_phases_b]
    summands_b = _make_summands(summands_b_exact)

    # ====================================================================
    # TYPE C: Pi-Pair Terms, (-1)^(Psi*Phi)
    # ====================================================================
    rowsum_a_c = (
        circuit.c_const_bits_a
        + _matmul_gf2(circuit.c_param_bits_a, param_vals, csr=circuit.c_param_csr_a)
    ) % 2
    rowsum_b_c = (
        circuit.c_const_bits_b
        + _matmul_gf2(circuit.c_param_bits_b, param_vals, csr=circuit.c_param_csr_b)
    ) % 2

    exponent_c = (rowsum_a_c * rowsum_b_c) % 2
    sum_exponents_c = jnp.sum(exponent_c, axis=-1) % 2

    summands_c_exact = (1 - 2 * sum_exponents_c)[..., None] * jnp.array(
        [1, 0, 0, 0], dtype=jnp.int32
    )
    summands_c = _make_summands(summands_c_exact)

    # ====================================================================
    # TYPE D: Phase Pairs (1 + e^a + e^b - e^g)
    # Padded values are masked to multiplicative identity.
    # ====================================================================
    rowsum_a_d = _matmul_gf2(circuit.d_param_bits_a, param_vals, csr=circuit.d_param_csr_a)
    rowsum_b_d = _matmul_gf2(circuit.d_param_bits_b, param_vals, csr=circuit.d_param_csr_b)

    alpha = (circuit.d_const_alpha + rowsum_a_d * 4) % 8
    beta = (circuit.d_const_beta + rowsum_b_d * 4) % 8

    if _USE_COMPLEX128:
        summands_d = _lazy_type_d_prod_complex(alpha, beta, circuit.d_num_terms)
    elif _LAZY_TERM_VALS:
        summands_d = _lazy_type_d_prod(alpha, beta, circuit.d_num_terms)
    else:
        gamma = (alpha + beta) % 8
        term_vals_d_exact = (
            _IDENTITY + _UNIT_PHASES[alpha] + _UNIT_PHASES[beta] - _UNIT_PHASES[gamma]
        )
        d_mask = (
            jnp.arange(circuit.d_const_alpha.shape[1])[None, :]
            < circuit.d_num_terms[:, None]
        )
        term_vals_d_exact = jnp.where(d_mask[..., None], term_vals_d_exact, _IDENTITY)
        term_vals_d = ExactScalarArray(term_vals_d_exact)
        summands_d = term_vals_d.prod(axis=-1)

    # ====================================================================
    # FINAL COMBINATION
    # ====================================================================
    static_phases = _make_summands(_UNIT_PHASES[circuit.phase_indices])
    float_factor = _make_summands(circuit.floatfactor)

    total_summands = functools.reduce(
        lambda a, b: a * b,
        [summands_a, summands_b, summands_c, summands_d, static_phases, float_factor],
    )

    if not circuit.has_approximate_floatfactors:
        if _USE_COMPLEX128:
            return jnp.sum(
                total_summands.to_complex() * (2.0 ** circuit.power2),
                axis=-1,
            )
        total_summands = ExactScalarArray(
            total_summands.coeffs, total_summands.power + circuit.power2
        )
        return total_summands.sum().to_complex()
    else:
        return jnp.sum(
            total_summands.to_complex()
            * circuit.approximate_floatfactors
            * 2.0**circuit.power2,
            axis=-1,
        )

"""Term-family modules for compiled scalar graphs.

Each compiled ZX scalar is the product of four term families plus a global
phase and a floatfactor. This module defines the four families as
``equinox.Module`` records, bundles the shared phase tables, and gives each
family an ``evaluate`` method that turns a batch of binary parameter values
into an ``ExactScalarArray`` (or ``ComplexScalarArray`` when
``TSIM_SCALAR_BACKEND=complex128``).

Downstream, ``compile.py`` builds instances of these classes from
``pyzx_param`` scalars, and ``evaluate.py`` orchestrates the products.
"""

import os

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax

from tsim.core.exact_scalar import (
    ComplexScalarArray,
    ExactScalarArray,
    _scalar_mul_with_power,
    _scalar_mul_with_power_split,
    _USE_COMPLEX128,
    make_summands,
)
from tsim.utils.linalg import _CustCsr, matmul_gf2

# Powers of ω = e^(iπ/4). UNIT_PHASES[k] is the exact 4-coefficient
# representation of ω^k.
UNIT_PHASES = jnp.array(
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

# Lookup table for exact scalars (1 + ω^k).
_ONE_PLUS_PHASES = UNIT_PHASES.at[:, 0].add(1)

_IDENTITY = jnp.array([1, 0, 0, 0], dtype=jnp.int32)

# Per-coordinate views of the lookup tables (split-coeffs path).
_ONE_PLUS_PHASES_C = tuple(_ONE_PLUS_PHASES[:, i] for i in range(4))
_UNIT_PHASES_C = tuple(UNIT_PHASES[:, i] for i in range(4))
_ID_C = (jnp.int32(1), jnp.int32(0), jnp.int32(0), jnp.int32(0))

# Complex-backend lookup tables (H).
_OMEGA_K_C128 = jnp.exp(1j * jnp.pi / 4 * jnp.arange(8)).astype(jnp.complex128)
_ONE_PLUS_OMEGA_K_C128 = (1.0 + _OMEGA_K_C128).astype(jnp.complex128)

# B: lazy prod-over-T flags. _LAZY_TERM_VALS gates the lax.scan replacement of
# the materialized (B,G,T,4) tensor; _LAZY_SPLIT_COEFFS threads four int32
# carries instead of one (...,4) tensor to skip an XLA concat fusion.
_LAZY_TERM_VALS = os.environ.get(
    "TSIM_TERM_VALS_LAZY", "") in ("1", "scan", "true", "True")
_LAZY_UNROLL = int(os.environ.get("TSIM_TERM_VALS_UNROLL", "4"))
_LAZY_SPLIT_COEFFS = os.environ.get(
    "TSIM_TERM_VALS_SPLIT_COEFFS", "") in ("1", "true", "True")

# C-prod: register-tiled cust kernels for prod-over-T.
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

# H2: register-tiled cust c128 prod kernels. Defaults on when H is on and the
# FFI is importable; force-off with TSIM_SCALAR_BACKEND_CUST_KERNEL=0.
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


# --------------------------------------------------------------------------
# Lazy / cust / complex helpers for NodePhases prod-over-T.
# --------------------------------------------------------------------------

def _lazy_node_phases_prod(phase_idx: Array, counts: Array) -> ExactScalarArray:
    """B: lax.scan over T with a (B,G,4) carry — no materialised T axis."""
    B, G, T = phase_idx.shape
    if T == 0:
        return ExactScalarArray(
            jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
            jnp.zeros((B, G), dtype=jnp.int32),
        )

    def step(carry, t):
        carry_coeffs, carry_power = carry
        pi_t = phase_idx[:, :, t]
        tv_t = _ONE_PLUS_PHASES[pi_t]
        mask_g = counts > t
        tv_t = jnp.where(mask_g[None, :, None], tv_t, _IDENTITY)
        new_power, new_coeffs = _scalar_mul_with_power(
            (carry_power, carry_coeffs),
            (jnp.zeros((), dtype=jnp.int32), tv_t),
        )
        return (new_coeffs, new_power), None

    init = (
        jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
        jnp.zeros((B, G), dtype=jnp.int32),
    )
    (final_coeffs, final_power), _ = lax.scan(
        step, init, jnp.arange(T), unroll=_LAZY_UNROLL,
    )
    return ExactScalarArray(final_coeffs, final_power)


def _lazy_node_phases_prod_split(phase_idx: Array, counts: Array) -> ExactScalarArray:
    """B + split-coeffs: thread (p, c0..c3) as five int32 carries through scan."""
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


def _cust_node_phases_prod(phase_idx: Array, counts: Array) -> ExactScalarArray:
    """C-prod: register-tiled cust kernel for NodePhases scalar_prod_over_T."""
    B, G, T = phase_idx.shape
    phase_gtb = jnp.transpose(phase_idx.astype(jnp.uint8), (1, 2, 0))
    counts_i32 = counts.astype(jnp.int32)
    power_gb, coeffs_gcb = _cust_node_phases_scalar_prod(
        phase_gtb, counts_i32, B=B, G=G, T=T,
    )
    return ExactScalarArray(
        jnp.transpose(coeffs_gcb, (2, 0, 1)),
        jnp.transpose(power_gb, (1, 0)),
    )


def _lazy_node_phases_prod_complex(phase_idx: Array, counts: Array) -> ComplexScalarArray:
    """H: c128 lazy prod with fully-unrolled scan ((B,G) carry in registers)."""
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
    # with the (B,G) c128 carry living in registers, avoiding the inter-step
    # HBM round-trip that unroll=4 was paying. T is static at compile time, so
    # this only blows up the trace per circuit, not per shot.
    final, _ = lax.scan(step, init, jnp.arange(T), unroll=True)
    return ComplexScalarArray(final)


# --------------------------------------------------------------------------
# Lazy / cust / complex helpers for PhasePairs prod-over-T.
# --------------------------------------------------------------------------

def _lazy_phase_pairs_prod(alpha: Array, beta: Array, counts: Array) -> ExactScalarArray:
    """B: lax.scan over T for PhasePairs."""
    B, G, T = alpha.shape
    if T == 0:
        return ExactScalarArray(
            jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
            jnp.zeros((B, G), dtype=jnp.int32),
        )

    def step(carry, t):
        carry_coeffs, carry_power = carry
        a_t = alpha[:, :, t]
        b_t = beta[:, :, t]
        g_t = (a_t + b_t) % 8
        tv_t = (
            _IDENTITY + UNIT_PHASES[a_t] + UNIT_PHASES[b_t] - UNIT_PHASES[g_t]
        )
        mask_g = counts > t
        tv_t = jnp.where(mask_g[None, :, None], tv_t, _IDENTITY)
        new_power, new_coeffs = _scalar_mul_with_power(
            (carry_power, carry_coeffs),
            (jnp.zeros((), dtype=jnp.int32), tv_t),
        )
        return (new_coeffs, new_power), None

    init = (
        jnp.broadcast_to(_IDENTITY, (B, G, 4)).astype(jnp.int32),
        jnp.zeros((B, G), dtype=jnp.int32),
    )
    (final_coeffs, final_power), _ = lax.scan(
        step, init, jnp.arange(T), unroll=_LAZY_UNROLL,
    )
    return ExactScalarArray(final_coeffs, final_power)


def _lazy_phase_pairs_prod_split(alpha: Array, beta: Array, counts: Array) -> ExactScalarArray:
    """B + split-coeffs for PhasePairs."""
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


def _cust_phase_pairs_prod(alpha: Array, beta: Array, counts: Array) -> ExactScalarArray:
    """C-prod: register-tiled cust kernel for PhasePairs scalar_prod_over_T."""
    B, G, T = alpha.shape
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


def _lazy_phase_pairs_prod_complex(
    alpha: Array, beta: Array, counts: Array,
) -> ComplexScalarArray:
    """H: c128 lazy prod for PhasePairs."""
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
    final, _ = lax.scan(step, init, jnp.arange(T), unroll=True)
    return ComplexScalarArray(final)


def _node_phases_prod_dispatch(phase_idx: Array, counts: Array):
    """Pick the active prod-over-T backend for NodePhases."""
    B, G, T = phase_idx.shape
    if _USE_COMPLEX128:
        return _lazy_node_phases_prod_complex(phase_idx, counts)
    if _LAZY_CUST_KERNEL_AVAILABLE and T > 0 and G * T >= _LAZY_CUST_MIN_GT:
        return _cust_node_phases_prod(phase_idx, counts)
    if _LAZY_SPLIT_COEFFS and T > 0:
        return _lazy_node_phases_prod_split(phase_idx, counts)
    if _LAZY_TERM_VALS:
        return _lazy_node_phases_prod(phase_idx, counts)
    return None


def _phase_pairs_prod_dispatch(alpha: Array, beta: Array, counts: Array):
    """Pick the active prod-over-T backend for PhasePairs."""
    B, G, T = alpha.shape
    if _USE_COMPLEX128:
        return _lazy_phase_pairs_prod_complex(alpha, beta, counts)
    if _LAZY_CUST_KERNEL_AVAILABLE and T > 0 and G * T >= _LAZY_CUST_MIN_GT:
        return _cust_phase_pairs_prod(alpha, beta, counts)
    if _LAZY_SPLIT_COEFFS and T > 0:
        return _lazy_phase_pairs_prod_split(alpha, beta, counts)
    if _LAZY_TERM_VALS:
        return _lazy_phase_pairs_prod(alpha, beta, counts)
    return None


# --------------------------------------------------------------------------
# Term-family modules.
# --------------------------------------------------------------------------


class NodePhases(eqx.Module):
    """Product of ``1 + exp(i·(α + ⊕params)·π)`` terms, one factor per stored term.

    Padded slots use ``0`` for both ``phases`` and ``params``; the evaluator masks
    padded slots to the multiplicative identity using ``counts``.

    Shapes are ``(num_graphs, max_terms)`` except ``params`` which is
    ``(num_graphs, max_terms, n_params)``.
    """

    phases: Array  # uint8, values 0-7 (the constant offset α, as ``4·α``)
    params: Array  # uint8, parameter parity bitmasks
    counts: Array  # int32, number of real (non-padded) terms per graph
    # D: precomputed CSR/packed view of params^T for cust SpSp/SpDn MM. None
    # when the cust backend isn't selected or the term is too small; static so
    # jit caches by instance identity (device pointers).
    params_csr: "_CustCsr | None" = eqx.field(static=True, default=None)

    def evaluate(self, param_vals: Array):
        """Evaluate Π (1 + ω^(4·parity + phase)) per graph, batched over param_vals.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.

        Returns:
            ``ExactScalarArray`` (default) or ``ComplexScalarArray`` (when
            ``TSIM_SCALAR_BACKEND=complex128``) of shape ``(batch, num_graphs)``.

        """
        rowsum = matmul_gf2(self.params, param_vals, csr=self.params_csr)
        phase_idx = (4 * rowsum + self.phases) % 8

        alt = _node_phases_prod_dispatch(phase_idx, self.counts)
        if alt is not None:
            return alt

        term_vals = _ONE_PLUS_PHASES[phase_idx]
        mask = jnp.arange(self.phases.shape[1])[None, :] < self.counts[:, None]
        term_vals = jnp.where(mask[..., None], term_vals, _IDENTITY)
        return ExactScalarArray(term_vals).prod(axis=-1)


class HalfPiPhases(eqx.Module):
    """Sum of ``exp(i·j·π·⊕params / 2)`` terms with ``j ∈ {1, 3}``.

    Terms sharing a parameter bitstring have been combined to a single stored
    coefficient ``j' ∈ {1, 2, 3}`` (see ``_compile_halfpi_phases``). Coefficients
    are stored in eighth-turn units — i.e. as ``2·j'`` — so the evaluator can
    reuse the ``ω = e^(iπ/4)`` phase table.

    Padded slots use ``0`` (the additive identity for phase sums), so padded
    entries contribute nothing to the summed exponent.

    Shapes are ``(num_graphs, max_terms)`` except ``params`` which is
    ``(num_graphs, max_terms, n_params)``.
    """

    coeffs: Array  # uint8, values in {0, 2, 4, 6}  (= 2·j', with 0 = padding)
    params: Array  # uint8, parameter parity bitmasks
    params_csr: "_CustCsr | None" = eqx.field(static=True, default=None)

    def evaluate(self, param_vals: Array):
        """Evaluate ω^(Σ coeffs · parity) per graph, batched over param_vals."""
        rowsum = matmul_gf2(self.params, param_vals, csr=self.params_csr)
        phase_idx = (rowsum * self.coeffs) % 8
        total_phase = jnp.sum(phase_idx, axis=-1) % 8
        return make_summands(UNIT_PHASES[total_phase])


class PiProducts(eqx.Module):
    """Product of ``(-1)^(ψ · φ)`` terms, with ψ and φ each a parity expression.

    Each side is encoded as a constant bit plus a parameter bitmask. Padded slots
    use ``0`` everywhere; a padded term contributes ``(-1)^0 = 1`` to the product.

    Shapes are ``(num_graphs, max_terms)`` except ``*_params`` which are
    ``(num_graphs, max_terms, n_params)``.
    """

    psi_const: Array  # uint8, values {0, 1}
    psi_params: Array  # uint8, parameter parity bitmask for ψ
    phi_const: Array  # uint8, values {0, 1}
    phi_params: Array  # uint8, parameter parity bitmask for φ
    psi_csr: "_CustCsr | None" = eqx.field(static=True, default=None)
    phi_csr: "_CustCsr | None" = eqx.field(static=True, default=None)

    def evaluate(self, param_vals: Array):
        """Evaluate Π (-1)^(ψ·φ) per graph as a real ±1 exact scalar."""
        psi = (
            self.psi_const + matmul_gf2(self.psi_params, param_vals, csr=self.psi_csr)
        ) % 2
        phi = (
            self.phi_const + matmul_gf2(self.phi_params, param_vals, csr=self.phi_csr)
        ) % 2

        exponent = (psi * phi) % 2
        sum_exponents = jnp.sum(exponent, axis=-1) % 2

        # (1 - 2·bit) ∈ {+1, -1}; promote to the 4-coefficient ExactScalar basis.
        # Cast to int32 first — without it the subtraction wraps in uint
        # (255 in uint8, 2^64-1 in uint64 under JAX_ENABLE_X64=1).
        signed = (1 - 2 * sum_exponents.astype(jnp.int32))[..., None]
        summands_exact = signed * _IDENTITY
        return make_summands(summands_exact)


class PhasePairs(eqx.Module):
    """Product of ``1 + e^(iα) + e^(iβ) − e^(i(α+β))`` terms.

    Each of ``α`` and ``β`` combines a constant phase with a parameter parity.
    Padded slots use ``0`` and are masked to the multiplicative identity using
    ``counts``.

    Shapes are ``(num_graphs, max_terms)`` except ``*_params`` which are
    ``(num_graphs, max_terms, n_params)``.
    """

    alpha: Array  # uint8, values 0-7 (constant offset of α, as ``4·α``)
    alpha_params: Array  # uint8, parameter parity bitmask for α
    beta: Array  # uint8, values 0-7 (constant offset of β, as ``4·β``)
    beta_params: Array  # uint8, parameter parity bitmask for β
    counts: Array  # int32, number of real (non-padded) terms per graph
    alpha_csr: "_CustCsr | None" = eqx.field(static=True, default=None)
    beta_csr: "_CustCsr | None" = eqx.field(static=True, default=None)

    def evaluate(self, param_vals: Array):
        """Evaluate Π (1 + ω^α + ω^β - ω^(α+β)) per graph, batched."""
        rowsum_a = matmul_gf2(self.alpha_params, param_vals, csr=self.alpha_csr)
        rowsum_b = matmul_gf2(self.beta_params, param_vals, csr=self.beta_csr)

        alpha = (self.alpha + rowsum_a * 4) % 8
        beta = (self.beta + rowsum_b * 4) % 8

        alt = _phase_pairs_prod_dispatch(alpha, beta, self.counts)
        if alt is not None:
            return alt

        gamma = (alpha + beta) % 8
        term_vals = (
            _IDENTITY + UNIT_PHASES[alpha] + UNIT_PHASES[beta] - UNIT_PHASES[gamma]
        )
        mask = jnp.arange(self.alpha.shape[1])[None, :] < self.counts[:, None]
        term_vals = jnp.where(mask[..., None], term_vals, _IDENTITY)
        return ExactScalarArray(term_vals).prod(axis=-1)


class ScalarPrefactor(eqx.Module):
    """Per-graph static scalar prefactor — independent of parameter values.

    Each graph's amplitude is the product of the four term families times the
    prefactor ``ω^phase_index · floatfactor · 2^power2``, with an additional
    complex ``approximate_floatfactor`` multiplied in when any graph's phase
    had a denominator outside ``{1, 2, 4}`` and was folded into float form.

    This class is pure data; the final fold-in with the term-family product
    (which requires branching on ``has_approximate_floatfactors``) lives in
    ``evaluate.py``.
    """

    phase_indices: Array  # uint8, shape (num_graphs,), values 0-7
    floatfactor: Array  # int32, shape (num_graphs, 4) — exact dyadic (a,b,c,d)
    power2: Array  # int32, shape (num_graphs,) — exponent on the 2^(·) scaling
    approximate_floatfactors: Array  # complex64, shape (num_graphs,)
    has_approximate_floatfactors: bool = eqx.field(static=True)

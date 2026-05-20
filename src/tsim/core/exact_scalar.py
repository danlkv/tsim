"""Exact scalar arithmetic for ZX-calculus phase computations.

Implements exact arithmetic for complex numbers of the form:
    (a + b*e^(i*pi/4) + c*i + d*e^(-i*pi/4)) * 2^power

This representation enables exact computation of phases in ZX-calculus graphs
without floating-point errors.
"""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

# Feature flags ported from session-2026-05-13 work onto PR #78. Only flags
# that fit cleanly within exact_scalar.py are kept; terms.py-based flags
# (lazy term_vals, split-coeffs prod, cust prod kernel) and the SpDn matmul
# flag (which now lives inside evaluate.py on PR #78) are dropped here.
_SCAN_BACKEND = os.environ.get("TSIM_EXACTSCALAR_SCAN", "")
_SCAN_UNROLL = int(os.environ.get("TSIM_EXACTSCALAR_UNROLL", "1"))

# H: complex128 scalar backend. When set, term modules return
# ComplexScalarArray (single c128 array) instead of ExactScalarArray
# (int32 dyadic coefficients + int32 power). Faster but precision-bounded;
# breaks above ~G=32 for cutting / high-stab-rank circuits.
_USE_COMPLEX128 = os.environ.get("TSIM_SCALAR_BACKEND", "") == "complex128"
_EXACTSCALAR_CUST_KERNEL = os.environ.get(
    "TSIM_EXACTSCALAR_CUST_KERNEL", "") in ("1", "true", "True")
_EXACTSCALAR_CUST_MIN_G = int(os.environ.get("TSIM_EXACTSCALAR_CUST_MIN_G", "0"))
if _EXACTSCALAR_CUST_KERNEL:
    try:
        from cust_jax import scalar_sum_along_axis_ffi as _cust_scalar_sum_ffi
        _EXACTSCALAR_CUST_KERNEL_AVAILABLE = True
    except Exception:
        _EXACTSCALAR_CUST_KERNEL_AVAILABLE = False
else:
    _EXACTSCALAR_CUST_KERNEL_AVAILABLE = False

# H2: c128 register-tiled sum kernel for ComplexScalarArray.sum.
_C128_SUM_CUST_KERNEL = os.environ.get(
    "TSIM_SCALAR_BACKEND_CUST_KERNEL", "1"
) in ("1", "true", "True")
_C128_SUM_CUST_MIN_G = int(os.environ.get("TSIM_SCALAR_BACKEND_CUST_MIN_G", "0"))
if _C128_SUM_CUST_KERNEL:
    try:
        from cust_jax import scalar_sum_along_axis_c128_ffi as _cust_sum_c128_ffi
        _C128_SUM_CUST_AVAILABLE = True
    except Exception:
        _C128_SUM_CUST_AVAILABLE = False
else:
    _C128_SUM_CUST_AVAILABLE = False


def _cust_c128_sum_along_axis(value, axis):
    """Dispatch ComplexScalarArray.sum to the c128 register-tiled CUDA kernel.

    Kernel signature is (G, BATCH) c128 → (BATCH,) c128. We flatten the
    non-G axes into BATCH at the FFI boundary and unflatten the result.
    """
    if axis < 0:
        axis += value.ndim
    val_t = jnp.moveaxis(value, axis, 0)  # (G, ...)
    G = val_t.shape[0]
    rest_shape = val_t.shape[1:]
    batch = 1
    for s in rest_shape:
        batch *= int(s)
    flat = val_t.reshape(G, batch).astype(jnp.complex128)
    out = _cust_sum_c128_ffi(flat, BATCH=batch, G=G)
    return out.reshape(rest_shape)

_E4 = jnp.exp(1j * jnp.pi / 4)
_E4D = jnp.exp(-1j * jnp.pi / 4)


@jax.jit
def _scalar_mul(d1: jax.Array, d2: jax.Array) -> jax.Array:
    """Multiply two exact scalar coefficient arrays.

    Args:
        d1: Shape (..., 4) array of coefficients.
        d2: Shape (..., 4) array of coefficients.

    Returns:
        Shape (..., 4) array of product coefficients.

    """
    a1, b1, c1, d1_coeff = d1[..., 0], d1[..., 1], d1[..., 2], d1[..., 3]
    a2, b2, c2, d2_coeff = d2[..., 0], d2[..., 1], d2[..., 2], d2[..., 3]

    A = a1 * a2 + b1 * d2_coeff - c1 * c2 + d1_coeff * b2
    B = a1 * b2 + b1 * a2 + c1 * d2_coeff + d1_coeff * c2
    C = a1 * c2 + b1 * b2 + c1 * a2 - d1_coeff * d2_coeff
    D = a1 * d2_coeff - b1 * c2 - c1 * b2 + d1_coeff * a2

    return jnp.stack([A, B, C, D], axis=-1).astype(d1.dtype)


def _reduce_power_coeffs_step(
    power: jax.Array, coeffs: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Reduce one common factor of 2 from coefficients into the power."""
    reducible = jnp.all(coeffs % 2 == 0, axis=-1) & jnp.any(coeffs != 0, axis=-1)
    coeffs = jnp.where(reducible[..., None], coeffs // 2, coeffs)
    power = jnp.where(reducible, power + 1, power)
    return power, coeffs


def _scalar_mul_with_power(x: tuple, y: tuple) -> tuple:
    """Multiply two exact scalars represented as (power, coeffs) tuples.

    Delegates coefficient multiplication to `_scalar_mul` and applies a single
    reduction step (divides by 2) when all resulting coefficients are even.

    Args:
        x: Tuple of (power, coeffs) where coeffs has shape (..., 4).
        y: Tuple of (power, coeffs) where coeffs has shape (..., 4).

    Returns:
        Tuple of (new_power, new_coeffs).

    """
    p1, c1 = x
    p2, c2 = y

    new_coeffs = _scalar_mul(c1, c2)
    p = p1 + p2
    return _reduce_power_coeffs_step(p, new_coeffs)


def _scalar_add_with_power(x: tuple, y: tuple) -> tuple:
    """Add two exact scalars represented as (power, coeffs) tuples."""
    p1, c1 = x
    p2, c2 = y

    c1_scale = jnp.left_shift(jnp.ones_like(p1), jnp.maximum(p1 - p2, 0))[..., None]
    c2_scale = jnp.left_shift(jnp.ones_like(p2), jnp.maximum(p2 - p1, 0))[..., None]

    p = jnp.minimum(p1, p2)
    new_coeffs = c1 * c1_scale + c2 * c2_scale
    return _reduce_power_coeffs_step(p, new_coeffs)


def _scalar_to_complex(data: jax.Array) -> jax.Array:
    """Convert a (N, 4) array of coefficients to a (N,) array of complex numbers."""
    return data[..., 0] + data[..., 1] * _E4 + data[..., 2] * 1j + data[..., 3] * _E4D


def _scalar_mul_with_power_split(
    p1, a1, b1, c1, d1,
    p2, a2, b2, c2, d2,
):
    """_scalar_mul + _reduce_power_coeffs_step on split (per-coordinate) coeffs.

    Returns (p, a, b, c, d) — 5 scalars instead of (power, (..., 4)) tuple.
    """
    A = a1 * a2 + b1 * d2 - c1 * c2 + d1 * b2
    B = a1 * b2 + b1 * a2 + c1 * d2 + d1 * c2
    C = a1 * c2 + b1 * b2 + c1 * a2 - d1 * d2
    D = a1 * d2 - b1 * c2 - c1 * b2 + d1 * a2
    p = p1 + p2
    all_even = ((A | B | C | D) & 1) == 0
    any_nonzero = (A != 0) | (B != 0) | (C != 0) | (D != 0)
    reducible = all_even & any_nonzero
    A = jnp.where(reducible, A // 2, A)
    B = jnp.where(reducible, B // 2, B)
    C = jnp.where(reducible, C // 2, C)
    D = jnp.where(reducible, D // 2, D)
    p = jnp.where(reducible, p + 1, p)
    return p, A, B, C, D


def _reduce_along_scan(power, coeffs, op, axis):
    """lax.scan-based reduction along `axis`. O(N) depth, O(1) extra memory.

    Used when TSIM_EXACTSCALAR_SCAN=scan. Default path uses lax.associative_scan.
    """
    if axis < 0:
        axis += power.ndim
    power_t = jnp.moveaxis(power, axis, 0)
    coeffs_t = jnp.moveaxis(coeffs, axis, 0)
    init = (power_t[0], coeffs_t[0])
    rest = (power_t[1:], coeffs_t[1:])

    def step(carry, x):
        return op(carry, x), None

    (final_power, final_coeffs), _ = lax.scan(step, init, rest, unroll=_SCAN_UNROLL)
    # Final fixpoint pass. ``_scalar_add_with_power`` and
    # ``_scalar_mul_with_power`` each apply only one
    # ``_reduce_power_coeffs_step`` per call, so a sequential scan over N
    # elements can lag canonical form (gcd of coeffs is odd) by up to
    # ``log2(N)`` reductions. Iterate with ``lax.while_loop`` so we only
    # pay for the iters that actually reduce.
    def _fixpoint_cond(state):
        _, _, did_change = state
        return did_change

    def _fixpoint_body(state):
        p, c, _ = state
        new_p, new_c = _reduce_power_coeffs_step(p, c)
        return new_p, new_c, jnp.any(new_p != p)

    init_state = (final_power, final_coeffs, jnp.bool_(True))
    final_power, final_coeffs, _ = lax.while_loop(
        _fixpoint_cond, _fixpoint_body, init_state,
    )
    return final_power, final_coeffs



def _cust_sum_along_axis(arr, axis):
    """Dispatch sum to cust_jax's register-tiled CUDA kernel.

    The kernel signature is (G, BATCH) + (G, BATCH, 4) → (BATCH,) + (BATCH, 4).
    """
    if axis < 0:
        axis += arr.power.ndim
    power_t = jnp.moveaxis(arr.power, axis, 0)
    coeffs_t = jnp.moveaxis(arr.coeffs, axis, 0)
    G = power_t.shape[0]
    rest_shape = power_t.shape[1:]
    batch = 1
    for s in rest_shape:
        batch *= int(s)
    p_in = power_t.reshape(G, batch).astype(jnp.int32)
    c_in = coeffs_t.reshape(G, batch, 4).astype(jnp.int32)
    out_p, out_c = _cust_scalar_sum_ffi(p_in, c_in, BATCH=batch, G=G)
    out_p = out_p.reshape(rest_shape)
    out_c = out_c.reshape(rest_shape + (4,))
    return ExactScalarArray(out_c, out_p)


class ExactScalarArray(eqx.Module):
    """Exact scalar array for ZX-calculus phase arithmetic using dyadic representation.

    Represents values of the form (c_0 + c_1·ω + c_2·ω² + c_3·ω³) × 2^power
    where ω = e^(iπ/4). This enables exact computation without floating-point errors.

    Attributes:
        coeffs: Array of shape (..., 4) containing dyadic coefficients.
        power: Array of powers of 2 for scaling.

    """

    coeffs: Array
    power: Array

    def __init__(self, coeffs: Array, power: Array | None = None):
        """Initialize from coefficients and optional power.

        The value represented is (c_0 + c_1*omega + c_2*omega^2 + c_3*omega^3) * 2^power
        where omega = e^{i*pi/4}.
        """
        self.coeffs = coeffs
        if power is None:
            self.power = jnp.zeros(coeffs.shape[:-1], dtype=jnp.int32)
        else:
            self.power = power

    def __mul__(self, other: "ExactScalarArray") -> "ExactScalarArray":
        """Element-wise multiplication."""
        new_coeffs = _scalar_mul(self.coeffs, other.coeffs)
        new_power = self.power + other.power
        return ExactScalarArray(new_coeffs, new_power)

    def sum(self, axis: int = -1) -> "ExactScalarArray":
        """Sum elements along the specified axis using normalized pairwise adds."""
        if _EXACTSCALAR_CUST_KERNEL_AVAILABLE and _SCAN_BACKEND == "scan":
            ax = axis if axis >= 0 else axis + self.power.ndim
            G_along = self.power.shape[ax]
            if G_along >= _EXACTSCALAR_CUST_MIN_G:
                return _cust_sum_along_axis(self, axis)
        if axis < 0:
            axis += self.power.ndim
        if _SCAN_BACKEND == "scan":
            result_power, result_coeffs = _reduce_along_scan(
                self.power, self.coeffs, _scalar_add_with_power, axis,
            )
            return ExactScalarArray(result_coeffs, result_power)
        scanned_power, scanned_coeffs = lax.associative_scan(
            _scalar_add_with_power, (self.power, self.coeffs), axis=axis
        )
        result_power = jnp.take(scanned_power, indices=-1, axis=axis)
        result_coeffs = jnp.take(scanned_coeffs, indices=-1, axis=axis)
        return ExactScalarArray(result_coeffs, result_power)

    def prod(self, axis: int = -1) -> "ExactScalarArray":
        """Compute product along the specified axis using associative scan.

        Returns identity (1+0i with power 0) for empty reductions.

        Args:
            axis: The axis along which to compute the product.

        Returns:
            ExactScalarArray with the product computed along the axis.

        """
        if axis < 0:
            axis += self.power.ndim

        if self.coeffs.shape[axis] == 0:
            # Product of empty sequence is identity: [1, 0, 0, 0] * 2^0
            coeffs_shape = self.coeffs.shape[:axis] + self.coeffs.shape[axis + 1 :]
            result_coeffs = jnp.zeros(coeffs_shape, dtype=self.coeffs.dtype)
            result_coeffs = result_coeffs.at[..., 0].set(1)
            return ExactScalarArray(result_coeffs)

        if _SCAN_BACKEND == "scan":
            result_power, result_coeffs = _reduce_along_scan(
                self.power, self.coeffs, _scalar_mul_with_power, axis,
            )
            return ExactScalarArray(result_coeffs, result_power)

        scanned_power, scanned_coeffs = lax.associative_scan(
            _scalar_mul_with_power, (self.power, self.coeffs), axis=axis
        )
        result_power = jnp.take(scanned_power, indices=-1, axis=axis)
        result_coeffs = jnp.take(scanned_coeffs, indices=-1, axis=axis)
        return ExactScalarArray(result_coeffs, result_power)

    def to_complex(self) -> jax.Array:
        """Convert to complex number."""
        c_val = _scalar_to_complex(self.coeffs)
        scale = jnp.pow(2.0, self.power)
        return c_val * scale


class ComplexScalarArray(eqx.Module):
    """complex128 scalar carrier with the same API surface as ExactScalarArray.

    Trades exact dyadic-rational arithmetic for native complex128 ops. Skips
    the (..., 4) coefficient layout entirely — values are stored as a single
    complex128 array, so `prod` / `sum` reduce to `jnp.prod` / `jnp.sum`. No
    power tracking; the 2^power factor is folded into the complex magnitude.

    Intended drop-in for ExactScalarArray when TSIM_SCALAR_BACKEND=complex128.
    Loses ~16 digits of precision per op (vs exact ESA), but eliminates the
    split-coeffs path and the cust prod/sum kernels.
    """

    value: Array  # complex128, shape (...)

    def __init__(self, value: Array):
        self.value = value.astype(jnp.complex128)

    @classmethod
    def from_coeffs(cls, coeffs: Array, power: Array | None = None) -> "ComplexScalarArray":
        val = _scalar_to_complex(coeffs.astype(jnp.float64))
        if power is not None:
            val = val * jnp.pow(2.0, power.astype(jnp.float64))
        return cls(val)

    def __mul__(self, other: "ComplexScalarArray") -> "ComplexScalarArray":
        return ComplexScalarArray(self.value * other.value)

    def sum(self, axis: int = -1) -> "ComplexScalarArray":
        if _C128_SUM_CUST_AVAILABLE:
            ax = axis if axis >= 0 else axis + self.value.ndim
            G_along = self.value.shape[ax]
            if G_along >= _C128_SUM_CUST_MIN_G:
                return ComplexScalarArray(_cust_c128_sum_along_axis(self.value, axis))
        return ComplexScalarArray(self.value.sum(axis=axis))

    def prod(self, axis: int = -1) -> "ComplexScalarArray":
        if self.value.shape[axis] == 0:
            shp = self.value.shape[:axis] + self.value.shape[axis + 1:]
            return ComplexScalarArray(jnp.ones(shp, dtype=jnp.complex128))
        return ComplexScalarArray(self.value.prod(axis=axis))

    def to_complex(self) -> jax.Array:
        return self.value


def make_summands(coeffs: Array, power: Array | None = None):
    """Wrap dyadic coefficients in the active scalar backend's array type.

    Term modules call this to produce their `.evaluate(...)` return value so
    the choice of ExactScalarArray vs ComplexScalarArray is centralized here.
    """
    if _USE_COMPLEX128:
        return ComplexScalarArray.from_coeffs(coeffs, power)
    return ExactScalarArray(coeffs) if power is None else ExactScalarArray(coeffs, power)

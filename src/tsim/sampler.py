"""Compiled samplers for measurements and detectors."""

from __future__ import annotations

import contextlib
import os
import time
import warnings
from math import ceil
from typing import TYPE_CHECKING, Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from pyzx_param.simulate import DecompositionStrategy

from tsim.compile.evaluate import evaluate
from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.core.types import CompiledComponent, CompiledProgram
from tsim.noise.channels import ChannelSampler
from tsim.utils.cuda_helpers import copy_d2h

_CHANNEL_BACKEND = os.environ.get("TSIM_CHANNEL_SAMPLER", "")
# G d2h variants (only consulted on the direct path when G is active):
#   ""        — default: bit_packed=False, slice, cp.asnumpy with implicit contig
#   "contig"  — bit_packed=False, slice, ascontiguousarray on GPU before d2h
#   "packed"  — bit_packed=True, d2h the byte-packed buffer, unpack on CPU
_G_DIRECT_PACK = os.environ.get("TSIM_G_DIRECT_PACK", "")

try:
    import cupy.cuda.nvtx as _nvtx
    _have_nvtx = True
except Exception:
    _have_nvtx = False


@contextlib.contextmanager
def _nvtx_range(name: str):
    """NVTX RangePush/Pop wrapper, no-op when cupy isn't available.

    Use as ``with _nvtx_range("name"): ...`` to keep call sites flat.
    """
    if _have_nvtx:
        _nvtx.RangePush(name)
        try:
            yield
        finally:
            _nvtx.RangePop()
    else:
        yield

if TYPE_CHECKING:
    from jax import Array as PRNGKey

    from tsim.circuit import Circuit


def _sample_component(
    component: CompiledComponent,
    f_params: jax.Array,
    key: PRNGKey,
) -> tuple[jax.Array, PRNGKey, jax.Array]:
    """Sample from component using autoregressive sampling.

    Args:
        component: The compiled component to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.

    Returns:
        Tuple of (samples, next_key, max_norm_deviation) where samples has
        shape (batch_size, num_outputs_for_component).

    """
    batch_size = f_params.shape[0]
    num_outputs = len(component.compiled_scalar_graphs) - 1

    f_selected = f_params[:, component.f_selection].astype(jnp.bool_)

    # Pre-allocate output array with final shape to avoid dynamic hstack
    m_accumulated = jnp.zeros((batch_size, num_outputs), dtype=jnp.bool_)

    # First circuit is normalization (only f-params)
    prev = jnp.abs(evaluate(component.compiled_scalar_graphs[0], f_selected))

    ones = jnp.ones((batch_size, 1), dtype=jnp.bool_)
    zero = jnp.zeros((1, 1), dtype=jnp.bool_)

    max_norm_deviation = jnp.array(0.0)

    # Autoregressive sampling for remaining circuits
    for i, circuit in enumerate(component.compiled_scalar_graphs[1:]):
        # Evaluate the real batch with trying_bit=1, plus one extra row for the
        # first sample's prefix with trying_bit=0 used by the norm check.
        params = jnp.hstack([f_selected, m_accumulated[:, :i], ones])
        check_row = jnp.hstack([f_selected[:1], m_accumulated[:1, :i], zero])
        probs = jnp.abs(evaluate(circuit, jnp.vstack([params, check_row])))
        p1 = probs[:batch_size]
        p0_single = probs[-1]

        norm = (p0_single + p1[0]) / prev[0]
        max_norm_deviation = jnp.maximum(max_norm_deviation, jnp.abs(norm - 1.0))

        key, subkey = jax.random.split(key)
        bits = jax.random.bernoulli(subkey, p=p1 / prev)
        m_accumulated = m_accumulated.at[:, i].set(bits)

        # Update prev using chain rule
        prev = jnp.where(bits, p1, prev - p1)

    return m_accumulated, key, max_norm_deviation


@jax.jit
def _sample_component_jit(
    component: CompiledComponent,
    f_params: jax.Array,
    key: PRNGKey,
) -> tuple[jax.Array, PRNGKey, jax.Array]:
    """JIT-compiled version of _sample_component."""
    return _sample_component(component, f_params, key)


def sample_component(
    component: CompiledComponent,
    f_params: jax.Array,
    key: PRNGKey,
) -> tuple[jax.Array, PRNGKey, jax.Array]:
    """Sample outputs from a single component using autoregressive sampling.

    Args:
        component: The compiled component to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.

    Returns:
        Tuple of (samples, next_key, max_norm_deviation) where samples has shape
        (batch_size, num_outputs_for_component).

    """
    # Skip JIT for small components (overhead not worth it)
    if len(component.output_indices) <= 1:
        return _sample_component(component, f_params, key)
    return _sample_component_jit(component, f_params, key)


def sample_program(
    program: CompiledProgram,
    f_params: jax.Array,
    key: PRNGKey,
) -> jax.Array:
    """Sample all outputs from a compiled program.

    Args:
        program: The compiled program to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.

    Returns:
        Samples array of shape (batch_size, num_outputs), reordered to
        match the original output indices.

    """
    results: list[jax.Array] = []

    if program.num_outputs == 0:
        batch_size = f_params.shape[0]
        return jnp.zeros((batch_size, 0), dtype=jnp.bool_)

    if len(program.direct_f_indices) > 0:
        with _nvtx_range(f"direct_bits[n={len(program.direct_f_indices)}]"):
            direct_bits = (
                f_params[:, program.direct_f_indices].astype(jnp.bool_)
                ^ program.direct_flips
            )
            results.append(direct_bits)

    for ci, component in enumerate(program.components):
        with _nvtx_range(f"sample_component[{ci}]"):
            samples, key, max_norm_deviation = sample_component(component, f_params, key)
        if np.isclose(max_norm_deviation, 1):
            raise ValueError(
                "A vanishing marginal probability distribution was encountered (normalization 0). "
                "This is likely the result of an underflow error. Please report this "
                "as a bug at https://github.com/QuEraComputing/tsim/issues/new."
            )  # pragma: no cover
        if max_norm_deviation > 1e-5:
            warnings.warn(
                "A marginal probability was not normalized correctly "
                f"(normalization deviated from 1 by {max_norm_deviation:.1e}). "
                "This is likely a floating point precision issue.",
                stacklevel=2,
            )
        results.append(samples)

    with _nvtx_range("concat_reindex"):
        combined = jnp.concatenate(results, axis=1)
        if program.output_reindex is not None:
            combined = combined[:, program.output_reindex]
    return combined


class _CompiledSamplerBase:
    """Base class for compiled samplers with common initialization logic."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        sample_detectors: bool,
        mode: Literal["sequential", "joint"],
        strategy: DecompositionStrategy = "cat5",
        seed: int | None = None,
    ):
        """Initialize the sampler by compiling the circuit.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, sample detectors/observables instead of measurements.
            mode: Compilation mode - "sequential" for autoregressive, "joint" for probabilities.
            strategy: Stabilizer rank decomposition strategy.
                Must be one of "cat5", "bss", "cutting".
            seed: Random seed. If None, a random seed is generated. Note that
                deterministic results are only guaranteed for a fixed batch size
                and fixed reference sample settings.

        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**30))

        self._key = jax.random.key(seed)

        prepared = prepare_graph(circuit, sample_detectors=sample_detectors)
        self._program = compile_program(prepared, mode=mode, strategy=strategy)

        channel_seed = int(np.random.default_rng(seed).integers(0, 2**30))
        self._channel_sampler = ChannelSampler(
            channel_probs=prepared.channel_probs,
            error_transform=prepared.error_transform,
            seed=channel_seed,
        )

        self.circuit = circuit
        self._num_detectors = prepared.num_detectors

        prog = self._program
        self._direct_f_indices = np.asarray(prog.direct_f_indices)
        self._direct_flips = np.asarray(prog.direct_flips, dtype=np.bool_)
        self._direct_reindex = (
            np.asarray(prog.output_reindex) if prog.output_reindex is not None else None
        )
        # Zero-copy fast path: f-indices are 0..n-1, no flips, no reindex.
        # Hit by typical surface-code detector circuits at low noise.
        n_direct = len(self._direct_f_indices)
        self._direct_zero_copy = (
            n_direct > 0
            and self._direct_reindex is None
            and not self._direct_flips.any()
            and np.array_equal(self._direct_f_indices, np.arange(n_direct))
        )

    def _init_cust_channel_sampler(self, batch_size: int) -> None:
        """Build a cuStabilizer BitMatrixSparseSampler for on-device channel sampling.

        Approximates per-channel exclusive outcomes as independent Bernoulli draws
        (the `approximate_disjoint_errors=True` semantics). Difference is O(p^2)
        per channel per shot.
        """
        import cupy as cp
        from cuquantum.stabilizer._options import Options
        from cuquantum.stabilizer.dem_sampling import BitMatrixSparseSampler

        probs_h, sig_h = self._channel_sampler.build_flat_error_matrix()
        if probs_h.size == 0:
            self._cust_sampler = None
            self._cust_sampler_max_shots = 0
            return
        self._cust_sampler = BitMatrixSparseSampler(
            cp.asarray(sig_h, dtype=cp.uint8),
            cp.asarray(probs_h, dtype=cp.float64),
            max_shots=int(batch_size),
            package="cupy",
            seed=int(np.random.default_rng().integers(0, 2**31)),
            options=Options(device_id=0),
        )
        self._cust_sampler_max_shots = int(batch_size)
        self._cust_sampler_rng = np.random.default_rng()

    def _peak_bytes_per_sample(self) -> int:
        """Estimate peak device memory per sample from compiled program structure."""
        peak = 0
        for component in self._program.components:
            for circuit in component.compiled_scalar_graphs:
                G = circuit.num_graphs
                max_a = circuit.node_phases.phases.shape[1]
                max_b = circuit.halfpi_phases.coeffs.shape[1]
                max_c = circuit.pi_products.psi_const.shape[1]
                max_d = circuit.phase_pairs.alpha.shape[1]
                largest = max(max_a * 16, max_b * 4, max_c * 4, max_d * 16)
                peak = max(peak, G * largest * 3)
        return max(peak, 1)

    def _estimate_batch_size(self) -> int:
        """Estimate the largest batch size that fits in available device memory."""
        device = jax.devices()[0]
        if device.platform == "gpu":
            stats = device.memory_stats()
            available = stats.get("bytes_limit", 8 * 1024**3) - stats.get(
                "bytes_in_use", 0
            )
        else:
            available = psutil.virtual_memory().available

        half_of_available = int(available * 0.5)  # conservative estimate
        return max(1, half_of_available // self._peak_bytes_per_sample())

    @overload
    def _sample_batches(
        self,
        shots: int,
        batch_size: int | None = None,
        *,
        compute_reference: Literal[False] = False,
    ) -> np.ndarray: ...

    @overload
    def _sample_batches(
        self,
        shots: int,
        batch_size: int | None = None,
        *,
        compute_reference: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def _sample_batches(
        self,
        shots: int,
        batch_size: int | None = None,
        *,
        compute_reference: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sample in batches and concatenate results.

        Args:
            shots: Number of samples to draw.
            batch_size: Samples per batch. Auto-determined if None.
            compute_reference: If True, add one extra sample to the first
                batch for a noiseless reference (f_params=0).

        Returns:
            Samples array, or (samples, reference) tuple when compute_reference=True.

        """
        if shots < 0:
            raise ValueError(f"shots must be non-negative, got {shots}")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        if shots == 0:
            empty = np.empty((0, self._program.num_outputs), dtype=np.bool_)
            if compute_reference:
                return empty, np.zeros(self._program.num_outputs, dtype=np.bool_)
            return empty

        if not self._program.components and not compute_reference:
            return self._sample_direct(shots)

        if batch_size is None:
            max_batch_size = self._estimate_batch_size()
            num_batches = max(1, ceil(shots / max_batch_size))
            batch_size = ceil(shots / num_batches)
        else:
            num_batches = ceil(shots / batch_size)

        if compute_reference and batch_size * num_batches == shots:
            # Bump batch_size so the first batch's reference sample fits
            # within existing batches (keeps shapes uniform for JIT).
            batch_size += 1

        batches: list[jax.Array] = []
        reference: np.ndarray | None = None

        _t_start = time.perf_counter()

        use_cust_chan = _CHANNEL_BACKEND == "cust"
        if use_cust_chan:
            if (not hasattr(self, "_cust_sampler")
                    or self._cust_sampler is None
                    or self._cust_sampler_max_shots < batch_size):
                self._init_cust_channel_sampler(batch_size)
            if self._cust_sampler is None:
                use_cust_chan = False  # no errors at all → fall back

        for _batch_i in range(num_batches):
            with _nvtx_range(f"sample_iter[{_batch_i}]"):
                if use_cust_chan:
                    import cupy as cp
                    with _nvtx_range("channel_sample"):
                        seed = int(self._cust_sampler_rng.integers(0, 2**31))
                        self._cust_sampler.sample(batch_size, seed=seed)
                        f_outc_cp = self._cust_sampler.get_outcomes(bit_packed=False)
                        f_outc_cp = f_outc_cp[:batch_size]
                        if compute_reference and reference is None:
                            f_outc_cp[0] = 0
                        # cuStabilizer outcomes are bit-pack-padded along axis
                        # 1 (544 bytes for a 532-wide matrix). JAX dlpack rejects
                        # non-trivial strides — make it contiguous.
                        f_outc_cp = cp.ascontiguousarray(f_outc_cp)
                    with _nvtx_range("dlpack"):
                        # Modern cupy + JAX both support the __dlpack__ protocol;
                        # pass the cupy array directly. Older .toDlpack() gives
                        # "dltensor" capsules that current JAX rejects.
                        f_params = jnp.from_dlpack(f_outc_cp)
                else:
                    with _nvtx_range("channel_sample"):
                        f_params_np = self._channel_sampler.sample(batch_size)
                        if compute_reference and reference is None:
                            f_params_np[0] = 0
                    with _nvtx_range("h2d"):
                        f_params = jnp.asarray(f_params_np)

                self._key, subkey = jax.random.split(self._key)

                with _nvtx_range("sample_program"):
                    samples = sample_program(self._program, f_params, subkey)

                if compute_reference and reference is None:
                    reference = np.asarray(samples[0])
                    samples = samples[1:]

                batches.append(samples)

        with _nvtx_range("gpu_concat"):
            # Concat on device first, then a single d2h into one numpy buffer.
            # The prior np.concatenate(batches) triggered per-batch __array__
            # d2h + a second host-side memcpy into the concat output — for big
            # bool tensors the host concat alone was ~1 s on star7 500k. Device
            # concat folds everything into one PCIe transfer.
            combined = batches[0] if len(batches) == 1 else jnp.concatenate(batches, axis=0)
            # Compute/d2h boundary: one sync, so the perf_counter split below
            # is honest about which side any pending async work belongs to.
            jax.block_until_ready(combined)
        self._last_sample_compute_s = time.perf_counter() - _t_start

        _t_d2h = time.perf_counter()
        with _nvtx_range("d2h_concat"):
            result = copy_d2h(combined)[:shots]
        self._last_sample_d2h_s = time.perf_counter() - _t_d2h

        if compute_reference:
            assert reference is not None
            return result, reference
        return result

    def _sample_direct(self, shots: int) -> np.ndarray:
        """Fast path when all outputs are direct functions of error bits.

        Default (numpy): channel sample + index/XOR all on host, single
        allocation returned to user. `_last_sample_compute_s` covers the
        whole call; `_last_sample_d2h_s` is 0 (data was never on device).

        G (``TSIM_CHANNEL_SAMPLER=cust``): channel sample + index/XOR all
        on GPU via cupy, then one ``cp.asnumpy`` at the API boundary. The
        d2h volume is the *reduced* output (n_outputs bits per shot),
        not the full channel-outcomes matrix.
        """
        use_cust_chan = _CHANNEL_BACKEND == "cust"
        if use_cust_chan:
            if (not hasattr(self, "_cust_sampler")
                    or self._cust_sampler is None
                    or self._cust_sampler_max_shots < shots):
                self._init_cust_channel_sampler(shots)
            if self._cust_sampler is None:
                use_cust_chan = False

        if not use_cust_chan:
            _t = time.perf_counter()
            with _nvtx_range("direct_numpy"):
                f_params = self._channel_sampler.sample(shots)
                if self._direct_zero_copy:
                    result = f_params[:, : len(self._direct_f_indices)].view(np.bool_)
                else:
                    result = f_params[:, self._direct_f_indices] ^ self._direct_flips
                    if self._direct_reindex is not None:
                        result = result[:, self._direct_reindex]
                    result = result.view(np.bool_)
            self._last_sample_compute_s = time.perf_counter() - _t
            self._last_sample_d2h_s = 0.0
            return result

        # G path: keep everything on GPU through the reduction, d2h only the
        # final per-shot outputs.
        import cupy as cp
        if not hasattr(self, "_direct_f_indices_cp"):
            self._direct_f_indices_cp = cp.asarray(self._direct_f_indices)
            self._direct_flips_cp = cp.asarray(self._direct_flips)
            self._direct_reindex_cp = (
                cp.asarray(self._direct_reindex)
                if self._direct_reindex is not None else None
            )

        n_direct = len(self._direct_f_indices)

        # bit_packed=True is only legal for zero_copy circuits (we'd need
        # per-output indexing into bit-packed storage to support the general
        # case, which isn't worth the complexity here).
        use_packed = _G_DIRECT_PACK == "packed" and self._direct_zero_copy

        _t = time.perf_counter()
        with _nvtx_range(f"direct_g[{_G_DIRECT_PACK or 'default'}]"):
            with _nvtx_range("channel_sample"):
                seed = int(self._cust_sampler_rng.integers(0, 2**31))
                self._cust_sampler.sample(shots, seed=seed)
                f_outc_cp = self._cust_sampler.get_outcomes(bit_packed=use_packed)
                f_outc_cp = f_outc_cp[:shots]
            with _nvtx_range("direct_reduce"):
                if use_packed:
                    # Packed buffer: (shots, ceil(n_padded/8)) uint8. Take the
                    # first ceil(n_direct/8) bytes per shot, then unpack on
                    # host post-d2h. Avoids the 8× unpack-on-device + the
                    # corresponding 8× d2h volume.
                    n_bytes = (n_direct + 7) // 8
                    result_cp = cp.ascontiguousarray(f_outc_cp[:, :n_bytes])
                elif self._direct_zero_copy:
                    # The slice [:, :n_direct] is non-contiguous (stride =
                    # padded width); make it contig before the d2h memcpy.
                    result_cp = cp.ascontiguousarray(
                        f_outc_cp[:, :n_direct].view(cp.bool_)
                    )
                else:
                    result_cp = f_outc_cp[:, self._direct_f_indices_cp] ^ self._direct_flips_cp
                    if self._direct_reindex_cp is not None:
                        result_cp = result_cp[:, self._direct_reindex_cp]
                    result_cp = cp.ascontiguousarray(result_cp.view(cp.bool_))
                # Sync once at the compute/d2h boundary so the perf_counter
                # split is honest.
                cp.cuda.get_current_stream().synchronize()
        self._last_sample_compute_s = time.perf_counter() - _t

        # d2h via cudaMemcpy into a cached pinned destination. cp.asnumpy
        # would do the same transfer through a pageable host buffer, which
        # collapses to ~2 GB/s (H100) / ~4 GB/s (B200) at large sizes — see
        # bench-alternatives/d2h-bandwidth. With pinned: ~22 / ~51 GB/s.
        # The pinned buffer aliases across sample() calls; callers that need
        # to retain results past the next sample() should .copy() the return.
        _t_d2h = time.perf_counter()
        with _nvtx_range("d2h_direct"):
            result = self._d2h_into_pinned(result_cp)
        self._last_sample_d2h_s = time.perf_counter() - _t_d2h

        if use_packed:
            # Host-side bit-unpack: (shots, n_bytes) uint8 → (shots, n_direct)
            # bool. We keep the unpack inside the d2h time bucket so the
            # comparison is apples-to-apples (caller sees a bool array either
            # way).
            _t_unpack = time.perf_counter()
            unpacked = np.unpackbits(result, axis=1, bitorder="little")
            result = unpacked[:, :n_direct].astype(np.bool_, copy=False)
            self._last_sample_d2h_s += time.perf_counter() - _t_unpack
        return result

    def _d2h_into_pinned(self, src_cp):
        """cudaMemcpy a contiguous device cupy array into a cached pinned host buffer.

        Returns an ndarray view of the pinned buffer with src_cp's dtype +
        shape. The buffer is allocated once via cp.cuda.alloc_pinned_memory
        and reused (resized only if a later call needs more bytes); the
        returned ndarray's lifetime is tied to the cached PinnedMemoryPointer.
        """
        import cupy as cp
        from cuda.bindings import runtime as cudart

        nbytes = int(src_cp.nbytes)
        if (not hasattr(self, "_pinned_mem")
                or self._pinned_mem is None
                or self._pinned_nbytes < nbytes):
            # Grow (or initialise). PinnedMemoryPointer is cleaned up when
            # the previous reference drops.
            self._pinned_mem = cp.cuda.alloc_pinned_memory(nbytes)
            self._pinned_nbytes = nbytes
            self._pinned_buf = np.frombuffer(self._pinned_mem, dtype=np.uint8)

        # Slice the destination to the exact byte count for this transfer.
        dst_bytes = self._pinned_buf[:nbytes]
        err = cudart.cudaMemcpy(
            dst_bytes.ctypes.data,
            int(src_cp.data.ptr),
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )[0]
        if err != cudart.cudaError_t.cudaSuccess:
            msg = cudart.cudaGetErrorString(err)[1].decode()
            raise RuntimeError(f"cudaMemcpy d2h failed: {msg}")

        # Reinterpret as src_cp's dtype + shape, then return that view of
        # the pinned region.
        out = dst_bytes.view(np.dtype(src_cp.dtype)).reshape(src_cp.shape)
        return out

    def __repr__(self) -> str:
        """Return a string representation with compilation statistics."""
        n_direct = len(self._program.direct_f_indices)

        c_graphs = []
        c_params = []
        c_a_terms = []
        c_b_terms = []
        c_c_terms = []
        c_d_terms = []
        total_memory_bytes = 0
        num_outputs = []

        for component in self._program.components:
            for circuit in component.compiled_scalar_graphs:
                num_outputs.append(len(component.output_indices))
                c_graphs.append(circuit.num_graphs)
                c_params.append(circuit.n_params)
                c_a_terms.append(circuit.node_phases.phases.size)
                c_b_terms.append(circuit.halfpi_phases.coeffs.size)
                c_c_terms.append(circuit.pi_products.psi_const.size)
                c_d_terms.append(
                    circuit.phase_pairs.alpha.size + circuit.phase_pairs.beta.size
                )

                total_memory_bytes += sum(
                    v.nbytes
                    for v in jax.tree_util.tree_leaves(circuit)
                    if isinstance(v, jax.Array)
                )

        def _format_bytes(n: int) -> str:
            if n < 1024:
                return f"{n} B"
            if n < 1024**2:
                return f"{n / 1024:.1f} kB"
            return f"{n / (1024**2):.1f} MB"

        total_memory_str = _format_bytes(total_memory_bytes)
        error_channel_bits = sum(
            channel.num_bits for channel in self._channel_sampler.channels
        )

        return (
            f"{type(self).__name__}({n_direct} direct, "
            f"{np.sum(c_graphs)} graphs, "
            f"{error_channel_bits} error channel bits, "
            f"{np.max(num_outputs) if num_outputs else 0} outputs for largest cc, "
            f"≤ {np.max(c_params) if c_params else 0} parameters, {np.sum(c_a_terms)} A terms, "
            f"{np.sum(c_b_terms)} B terms, "
            f"{np.sum(c_c_terms)} C terms, {np.sum(c_d_terms)} D terms, "
            f"{total_memory_str})"
        )


class CompiledMeasurementSampler(_CompiledSamplerBase):
    """Samples measurement outcomes from a quantum circuit.

    Uses sequential decomposition [0, 1, 2, ..., n] where:
    - compiled_scalar_graphs[0]: normalization (0 outputs plugged)
    - compiled_scalar_graphs[i]: cumulative probability up to bit i
    """

    def __init__(
        self,
        circuit: Circuit,
        *,
        strategy: DecompositionStrategy = "cat5",
        seed: int | None = None,
    ):
        """Create a measurement sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: Stabilizer rank decomposition strategy.
                Must be one of "cat5", "bss", "cutting".
            seed: Random seed for the sampler. IMPORTANT: Currently, the sampler
                will only produce deterministic samples for fixed batch size. If
                deterministic samples are needed, the batch size should be set
                manually.

        """
        super().__init__(
            circuit,
            sample_detectors=False,
            mode="sequential",
            seed=seed,
            strategy=strategy,
        )

    def sample(self, shots: int, *, batch_size: int | None = None) -> np.ndarray:
        """Sample measurement outcomes from the circuit.

        Args:
            shots: The number of times to sample every measurement in the circuit.
            batch_size: The number of samples to process in each batch. Defaults to
                None, which automatically chooses a batch size based on available
                memory. When using a GPU, setting this explicitly can help fully
                utilize VRAM for maximum performance. NOTE: Changing the batch size
                will affect reproducibility even with a fixed seed.

        Returns:
            A numpy array containing the measurement samples.

        """
        return self._sample_batches(shots, batch_size)


def _maybe_bit_pack(array: np.ndarray, *, bit_packed: bool) -> np.ndarray:
    """Optionally bit-pack a boolean array."""
    if not bit_packed:
        return array
    return np.packbits(array.astype(np.bool_), axis=1, bitorder="little")


class CompiledDetectorSampler(_CompiledSamplerBase):
    """Samples detector and observable outcomes from a quantum circuit."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        strategy: DecompositionStrategy = "cat5",
        seed: int | None = None,
    ):
        """Create a detector sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: Stabilizer rank decomposition strategy.
                Must be one of "cat5", "bss", "cutting".
            seed: Random seed for the sampler. IMPORTANT: Currently, the sampler
                will only produce deterministic samples for fixed batch size and
                fixed reference sample settings. If deterministic samples are
                needed, the batch size should be set manually.

        """
        super().__init__(
            circuit,
            sample_detectors=True,
            mode="sequential",
            seed=seed,
            strategy=strategy,
        )

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[True],
        bit_packed: bool = False,
        use_detector_reference_sample: bool = False,
        use_observable_reference_sample: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[False] = False,
        bit_packed: bool = False,
        use_detector_reference_sample: bool = False,
        use_observable_reference_sample: bool = False,
    ) -> np.ndarray: ...

    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
        use_detector_reference_sample: bool = False,
        use_observable_reference_sample: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return detector samples from the circuit.

        The circuit must define the detectors using DETECTOR instructions. Observables
        defined by OBSERVABLE_INCLUDE instructions can also be included in the results
        as honorary detectors.

        Args:
            shots: The number of times to sample every detector in the circuit.
            batch_size: The number of samples to process in each batch. Defaults to
                None, which automatically chooses a batch size based on available
                memory. When using a GPU, setting this explicitly can help fully
                utilize VRAM for maximum performance. NOTE: Changing the batch size
                will affect reproducibility even with a fixed seed.
            separate_observables: Defaults to False. When set to True, the return value
                is a (detection_events, observable_flips) tuple instead of a flat
                detection_events array.
            prepend_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the start of the results.
            append_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the end of the results.
            bit_packed: Defaults to false. When set, results are bit-packed.
            use_detector_reference_sample: Defaults to False. When True, a noiseless
                reference sample is computed and XORed with detector outcomes so that
                results represent deviations from the noiseless baseline. This should
                only be used when detectors are deterministic. Otherwise, it can
                unpredictably change the results.
            use_observable_reference_sample: Defaults to False. When True, a noiseless
                reference sample is computed and XORed with observable outcomes so that
                results represent deviations from the noiseless baseline. This should
                only be used when observables are deterministic. Otherwise, it can
                unpredictably change the results.

        Returns:
            A numpy array or tuple of numpy arrays containing the samples.

        Raises:
            ValueError: If ``separate_observables`` is combined with
                ``prepend_observables`` or ``append_observables``.

        """
        if separate_observables and (prepend_observables or append_observables):
            raise ValueError(
                "Can't specify separate_observables=True with "
                "append_observables=True or prepend_observables=True"
            )

        compute_reference = (
            use_detector_reference_sample or use_observable_reference_sample
        )

        if compute_reference:
            samples, reference = self._sample_batches(
                shots, batch_size, compute_reference=True
            )
            num_detectors = self._num_detectors
            if use_detector_reference_sample:
                samples[:, :num_detectors] ^= reference[:num_detectors]
            if use_observable_reference_sample:
                samples[:, num_detectors:] ^= reference[num_detectors:]
        else:
            samples = self._sample_batches(shots, batch_size)

        num_detectors = self._num_detectors
        det_samples = samples[:, :num_detectors]
        obs_samples = samples[:, num_detectors:]

        if prepend_observables and append_observables:
            combined = np.concatenate([obs_samples, det_samples, obs_samples], axis=1)
            return _maybe_bit_pack(combined, bit_packed=bit_packed)
        if append_observables:
            return _maybe_bit_pack(samples, bit_packed=bit_packed)
        if prepend_observables:
            combined = np.concatenate([obs_samples, det_samples], axis=1)
            return _maybe_bit_pack(combined, bit_packed=bit_packed)
        if separate_observables:
            return (
                _maybe_bit_pack(det_samples, bit_packed=bit_packed),
                _maybe_bit_pack(obs_samples, bit_packed=bit_packed),
            )

        return _maybe_bit_pack(det_samples, bit_packed=bit_packed)
        # TODO: don't compute observables if they are discarded here


class CompiledStateProbs(_CompiledSamplerBase):
    """Computes measurement probabilities for a given state.

    Uses joint decomposition [0, n] where:
    - compiled_scalar_graphs[0]: normalization (0 outputs plugged)
    - compiled_scalar_graphs[1]: full joint probability (all outputs plugged)
    """

    def __init__(
        self,
        circuit: Circuit,
        *,
        sample_detectors: bool = False,
        strategy: DecompositionStrategy = "cat5",
        seed: int | None = None,
    ):
        """Create a probability estimator.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, compute detector/observable probabilities.
            strategy: Stabilizer rank decomposition strategy.
                Must be one of "cat5", "bss", "cutting".
            seed: Random seed. If None, a random seed is generated. Note that
                deterministic results are only guaranteed for a fixed batch size.

        """
        super().__init__(
            circuit,
            sample_detectors=sample_detectors,
            mode="joint",
            seed=seed,
            strategy=strategy,
        )

    def probability_of(self, state: np.ndarray, *, batch_size: int) -> np.ndarray:
        """Compute probabilities for a batch of error samples given a measurement state.

        Args:
            state: The measurement outcome state to compute probability for.
            batch_size: Number of error samples to use for estimation.

        Returns:
            Array of probabilities P(state | error_sample) for each error sample.

        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        expected_outputs = self._program.num_outputs
        if state.shape != (expected_outputs,):
            raise ValueError(
                f"state must have shape ({expected_outputs},), got {state.shape}"
            )
        f_samples = jnp.asarray(self._channel_sampler.sample(batch_size))
        p_norm = jnp.ones(batch_size)
        p_joint = jnp.ones(batch_size)

        if len(self._program.direct_f_indices) > 0:
            direct_bits = (
                f_samples[:, self._program.direct_f_indices].astype(jnp.bool_)
                ^ self._program.direct_flips
            )
            n_direct = len(self._program.direct_f_indices)
            targets = state[self._program.output_order[:n_direct]]
            p_joint = p_joint * (direct_bits == targets).all(axis=1)

        for component in self._program.components:
            assert len(component.compiled_scalar_graphs) == 2

            f_selected = f_samples[:, component.f_selection]

            norm_circuit, joint_circuit = component.compiled_scalar_graphs

            # Normalization: only f-params
            p_norm = p_norm * jnp.abs(evaluate(norm_circuit, f_selected))

            # Joint probability: f-params + state
            component_state = state[list(component.output_indices)]
            tiled_state = jnp.tile(component_state, (batch_size, 1))
            joint_params = jnp.hstack([f_selected, tiled_state])
            p_joint = p_joint * jnp.abs(evaluate(joint_circuit, joint_params))

        return np.asarray(p_joint / p_norm)

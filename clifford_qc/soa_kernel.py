"""Structure-of-Arrays (SoA) vectorized Pauli-operator algebra kernel.

This module provides a pure-NumPy, cache-free, bit-packed Structure-of-Arrays
representation (:class:`PauliSoA`) and batched algebra primitives for sparse
Pauli sums in ``Cl(2n, C)``:

* **Compact width-selected codes.** Single-word unsigned arrays (``uint16`` for
  ``n <= 8``, ``uint32`` for ``n <= 16``, ``uint64`` for ``n <= 32``) and
  little-endian multiword ``uint64`` arrays of shape ``(t, ceil(n / 32))`` for
  ``n > 32``, lifting the 31-qubit ``int64`` ceiling.
* **Batched XOR + popcount products.** Vectorized outer products using the same
  2-bit-lane phase formula as :func:`clifford_qc.pauli_kernel.word_mul_reference`,
  chunked to bound peak memory and merged via dense ``np.bincount`` (at small
  ``n``) or sorted ``np.add.reduceat`` (for general and multiword ``n``).
* **Symplectic commutator pre-filter.** Skips commuting pairs before forming
  products or phases, halving work on typical commutator workloads.
* **Sorted-array intersections.** Computes Pauli trace pairings and
  Hilbert-Schmidt inner products via vectorized binary search on sorted codes.
* **Global bank collation.** Builds ``A_i† A_j`` and ``A_i† H A_j`` directly in
  SoA (sharing ``H A_j`` across rows) and collates all pair products into one
  canonical multiword :class:`SoAWordTable` and CSR-packed coefficient buffer
  without per-pair Python dictionaries or the process-wide ``lru_cache`` memo.
* **Optional GPU backends.** Opt-in ``cupy`` (fused CUDA ``__popcll`` kernel)
  and ``torch`` (CUDA SWAR popcount + ``scatter_add_``) backends with zero
  added core dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Number
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np

from .pauli_kernel import (
    code_to_label,
    label_to_code,
    validate_n,
    validate_word_code,
)

TOL = 1e-12
_PHASE4_C = np.array([1.0 + 0j, 1j, -1.0 + 0j, -1j], dtype=np.complex128)
_BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")

DEFAULT_CPU_CHUNK_PAIRS = 2_000_000
DEFAULT_GPU_CHUNK_PAIRS = 8_000_000


# ---------------------------------------------------------------------------
# Code-width, popcount, and multiword helpers
# ---------------------------------------------------------------------------


def word_width_for_n(n: int) -> int:
    """Number of 64-bit limbs needed for ``n`` qubits (0 means single-word)."""
    validate_n(n)
    if n <= 32:
        return 0
    return (n + 31) // 32


def code_dtype_for_n(n: int) -> np.dtype[Any]:
    """Smallest unsigned integer dtype holding a single-word ``n``-qubit code."""
    validate_n(n)
    if n <= 8:
        return np.dtype(np.uint16)
    if n <= 16:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def lane_mask_for_width(qubits: int, dtype: np.dtype[Any]) -> np.unsignedinteger[Any]:
    """Bit 0 of each 2-bit Pauli lane for ``qubits <= 32`` in ``dtype``."""
    mask_int = ((1 << (2 * qubits)) - 1) // 3 if qubits > 0 else 0
    return cast(np.unsignedinteger[Any], dtype.type(mask_int))


def multiword_lane_masks(n: int) -> np.ndarray:
    """Per-limb lane masks for ``n > 32`` (shape ``(w,)``, ``uint64``)."""
    w = word_width_for_n(n)
    if w == 0:
        raise ValueError(f"multiword_lane_masks requires n > 32, got {n}")
    masks = np.empty(w, dtype=np.uint64)
    full = np.uint64(((1 << 64) - 1) // 3)
    for limb in range(w):
        rem = min(32, n - 32 * limb)
        if rem == 32:
            masks[limb] = full
        else:
            masks[limb] = np.uint64(((1 << (2 * rem)) - 1) // 3)
    return masks


def popcount_array(arr: np.ndarray) -> np.ndarray:
    """Bitwise popcount returning ``int16`` (compatible with NumPy < 2.0)."""
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(arr).astype(np.int16, copy=False)
    a = np.ascontiguousarray(arr)
    nbytes = a.dtype.itemsize
    byte_view = a.view(np.uint8).reshape(a.shape + (nbytes,))
    return _BYTE_POPCOUNT[byte_view].sum(axis=-1, dtype=np.int16)


def _xor_reduce_last_axis(arr: np.ndarray) -> np.ndarray:
    """Bitwise XOR reduction across the trailing limb axis."""
    acc = arr[..., 0].copy()
    for limb in range(1, arr.shape[-1]):
        acc ^= arr[..., limb]
    return acc


def _canonical_complex_from_parts(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Pack real/imag float64 arrays into complex128 with ``-0.0`` cleared."""
    out = np.empty(re.shape, dtype=np.complex128)
    # IEEE-754: (-0.0) + (+0.0) == +0.0, matching MV's no-negative-zero contract.
    out.real = re + 0.0
    out.imag = im + 0.0
    return out


def _canonical_complex(coeffs: np.ndarray) -> np.ndarray:
    c = np.asarray(coeffs, dtype=np.complex128)
    return _canonical_complex_from_parts(c.real, c.imag)


def _multiword_sort_key(codes: np.ndarray) -> np.ndarray:
    """Big-endian fixed-width byte view matching integer ascending order."""
    w = codes.shape[1]
    be = np.ascontiguousarray(codes[:, ::-1], dtype=">u8")
    return be.view(np.dtype((np.void, 8 * w))).ravel()


def int_codes_to_array(n: int, codes: Sequence[int]) -> np.ndarray:
    """Convert validated Python integer Pauli codes to a packed NumPy array."""
    validate_n(n)
    w = word_width_for_n(n)
    if w == 0:
        dtype = code_dtype_for_n(n)
        if not codes:
            return np.empty(0, dtype=dtype)
        limit = 1 << (2 * n)
        for c in codes:
            ci = int(c)
            if ci < 0 or ci >= limit:
                raise ValueError(f"word code must be in [0, 4**{n}), got {c!r}")
        return np.array(codes, dtype=dtype)
    t = len(codes)
    out = np.zeros((t, w), dtype=np.uint64)
    if t == 0:
        return out
    limit = 1 << (2 * n)
    mask64 = (1 << 64) - 1
    for i, c in enumerate(codes):
        ci = int(c)
        if ci < 0 or ci >= limit:
            raise ValueError(f"word code must be in [0, 4**{n}), got {c!r}")
        for limb in range(w):
            out[i, limb] = (ci >> (64 * limb)) & mask64
    return out


def array_to_int_codes(n: int, codes: np.ndarray) -> list[int]:
    """Convert packed single-word or multiword codes back to Python ints."""
    validate_n(n)
    if codes.ndim == 1:
        return [int(x) for x in codes.tolist()]
    w = codes.shape[1]
    out: list[int] = []
    for row in codes.tolist():
        val = 0
        for limb in range(w - 1, -1, -1):
            val = (val << 64) | int(row[limb])
        out.append(val)
    return out


def _single_int_code_from_row(row: np.ndarray) -> int:
    val = 0
    for limb in range(row.shape[0] - 1, -1, -1):
        val = (val << 64) | int(row[limb])
    return val


def _validate_code_array(n: int, codes: np.ndarray) -> np.ndarray:
    """Normalize and validate an incoming packed code array for ``n`` qubits."""
    validate_n(n)
    w = word_width_for_n(n)
    arr = np.asarray(codes)
    if w == 0:
        dtype = code_dtype_for_n(n)
        if arr.ndim != 1:
            raise ValueError(f"single-word codes for n={n} must be 1-D, got shape {arr.shape}")
        if arr.size == 0:
            return np.empty(0, dtype=dtype)
        if arr.dtype == np.object_:
            return int_codes_to_array(n, arr.tolist())
        if np.issubdtype(arr.dtype, np.signedinteger) and bool(np.any(arr < 0)):
            raise ValueError("word codes must be non-negative")
        if n < 32:
            limit = np.uint64(1 << (2 * n))
            if bool(np.any(arr.astype(np.uint64, copy=False) >= limit)):
                raise ValueError(f"word code out of range for n={n}")
        return np.ascontiguousarray(arr, dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != w:
        raise ValueError(
            f"multiword codes for n={n} must have shape (t, {w}), got {arr.shape}"
        )
    if arr.shape[0] == 0:
        return np.empty((0, w), dtype=np.uint64)
    arr_u64 = np.ascontiguousarray(arr, dtype=np.uint64)
    rem = n - 32 * (w - 1)
    if rem < 32:
        top_limit = np.uint64(1 << (2 * rem))
        if bool(np.any(arr_u64[:, -1] >= top_limit)):
            raise ValueError(f"multiword code out of range for n={n}")
    return arr_u64


# ---------------------------------------------------------------------------
# Sort-reduce and dense accumulation primitives (NumPy)
# ---------------------------------------------------------------------------


def _sort_reduce_single(
    codes: np.ndarray,
    re: np.ndarray,
    im: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort 1-D unsigned codes, sum duplicate coefficients, prune ``|c| <= tol``."""
    if codes.size == 0:
        return np.empty(0, dtype=codes.dtype), np.empty(0, dtype=np.complex128)
    order = np.argsort(codes, kind="quicksort")
    sc = codes[order]
    sre = re[order]
    sim = im[order]
    starts = np.empty(sc.size, dtype=np.bool_)
    starts[0] = True
    if sc.size > 1:
        starts[1:] = sc[1:] != sc[:-1]
    idx = np.flatnonzero(starts)
    u_codes = sc[idx]
    u_re = np.add.reduceat(sre, idx)
    u_im = np.add.reduceat(sim, idx)
    if tol > 0.0:
        keep = (u_re * u_re + u_im * u_im) > (tol * tol)
    else:
        keep = (u_re != 0.0) | (u_im != 0.0)
    return u_codes[keep], _canonical_complex_from_parts(u_re[keep], u_im[keep])


def _sort_reduce_multiword(
    codes: np.ndarray,
    re: np.ndarray,
    im: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort 2-D multiword ``uint64`` codes, sum duplicates, prune ``|c| <= tol``."""
    w = codes.shape[1]
    if codes.shape[0] == 0:
        return np.empty((0, w), dtype=np.uint64), np.empty(0, dtype=np.complex128)
    keys = _multiword_sort_key(codes)
    order = np.argsort(keys, kind="quicksort")
    skeys = keys[order]
    starts = np.empty(skeys.size, dtype=np.bool_)
    starts[0] = True
    if skeys.size > 1:
        starts[1:] = skeys[1:] != skeys[:-1]
    idx = np.flatnonzero(starts)
    u_codes = np.ascontiguousarray(codes[order[idx]])
    u_re = np.add.reduceat(re[order], idx)
    u_im = np.add.reduceat(im[order], idx)
    if tol > 0.0:
        keep = (u_re * u_re + u_im * u_im) > (tol * tol)
    else:
        keep = (u_re != 0.0) | (u_im != 0.0)
    return u_codes[keep], _canonical_complex_from_parts(u_re[keep], u_im[keep])


def _merge_sorted_single(
    ca: np.ndarray,
    wa: np.ndarray,
    cb: np.ndarray,
    wb: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate and reduce two single-word arrays."""
    if ca.size == 0:
        if wb.size == 0:
            return cb, wb
        keep = (wb.real * wb.real + wb.imag * wb.imag) > (tol * tol) if tol > 0.0 else (
            (wb.real != 0.0) | (wb.imag != 0.0)
        )
        return cb[keep], _canonical_complex(wb[keep])
    if cb.size == 0:
        keep = (wa.real * wa.real + wa.imag * wa.imag) > (tol * tol) if tol > 0.0 else (
            (wa.real != 0.0) | (wa.imag != 0.0)
        )
        return ca[keep], _canonical_complex(wa[keep])
    codes = np.concatenate((ca, cb))
    re = np.concatenate((wa.real, wb.real))
    im = np.concatenate((wa.imag, wb.imag))
    return _sort_reduce_single(codes, re, im, tol=tol)


def _merge_sorted_multiword(
    ca: np.ndarray,
    wa: np.ndarray,
    cb: np.ndarray,
    wb: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if ca.shape[0] == 0:
        keep = (wb.real * wb.real + wb.imag * wb.imag) > (tol * tol) if tol > 0.0 else (
            (wb.real != 0.0) | (wb.imag != 0.0)
        )
        return cb[keep], _canonical_complex(wb[keep])
    if cb.shape[0] == 0:
        keep = (wa.real * wa.real + wa.imag * wa.imag) > (tol * tol) if tol > 0.0 else (
            (wa.real != 0.0) | (wa.imag != 0.0)
        )
        return ca[keep], _canonical_complex(wa[keep])
    codes = np.vstack((ca, cb))
    re = np.concatenate((wa.real, wb.real))
    im = np.concatenate((wa.imag, wb.imag))
    return _sort_reduce_multiword(codes, re, im, tol=tol)


# ---------------------------------------------------------------------------
# PauliSoA container
# ---------------------------------------------------------------------------


class PauliSoA:
    """Canonical sorted Structure-of-Arrays sparse Pauli sum in ``Cl(2n, C)``.

    Invariants (matching :class:`clifford_qc.multivector.MV`):
    * ``codes`` are strictly ascending and unique (1-D unsigned integer for
      ``n <= 32``; 2-D ``(t, ceil(n/32))`` little-endian ``uint64`` for
      ``n > 32``).
    * ``coeffs`` are ``complex128`` with ``|c| > tol`` (default ``1e-12``) and
      no ``-0.0`` real or imaginary components.
    * Does not populate or read the process-wide ``lru_cache`` word memo.
    """

    __slots__ = ("n", "codes", "coeffs")

    def __init__(
        self,
        n: int,
        codes: np.ndarray,
        coeffs: np.ndarray,
        *,
        tol: float = TOL,
        is_sorted_unique: bool = False,
    ) -> None:
        validate_n(n)
        c_arr = _validate_code_array(n, codes)
        w_arr = np.asarray(coeffs, dtype=np.complex128)
        if w_arr.ndim != 1 or w_arr.shape[0] != c_arr.shape[0]:
            raise ValueError(
                f"coeffs length {w_arr.shape} must match codes length {c_arr.shape[0]}"
            )
        if is_sorted_unique:
            if w_arr.size:
                keep = (
                    (w_arr.real * w_arr.real + w_arr.imag * w_arr.imag) > (tol * tol)
                    if tol > 0.0
                    else ((w_arr.real != 0.0) | (w_arr.imag != 0.0))
                )
                if not bool(np.all(keep)):
                    c_arr = c_arr[keep]
                    w_arr = w_arr[keep]
                w_arr = _canonical_complex(w_arr)
            self.n = n
            self.codes = c_arr
            self.coeffs = w_arr
        else:
            if c_arr.ndim == 1:
                c_arr, w_arr = _sort_reduce_single(c_arr, w_arr.real, w_arr.imag, tol=tol)
            else:
                c_arr, w_arr = _sort_reduce_multiword(c_arr, w_arr.real, w_arr.imag, tol=tol)
            self.n = n
            self.codes = c_arr
            self.coeffs = w_arr

    @classmethod
    def zero(cls, n: int) -> "PauliSoA":
        validate_n(n)
        w = word_width_for_n(n)
        codes = (
            np.empty(0, dtype=code_dtype_for_n(n))
            if w == 0
            else np.empty((0, w), dtype=np.uint64)
        )
        return cls(n, codes, np.empty(0, dtype=np.complex128), is_sorted_unique=True)

    @classmethod
    def scalar(cls, n: int, c: complex = 1.0, *, tol: float = TOL) -> "PauliSoA":
        validate_n(n)
        cv = complex(c)
        if abs(cv) <= tol:
            return cls.zero(n)
        w = word_width_for_n(n)
        codes = (
            np.zeros(1, dtype=code_dtype_for_n(n))
            if w == 0
            else np.zeros((1, w), dtype=np.uint64)
        )
        return cls(n, codes, np.array([cv], dtype=np.complex128), tol=tol, is_sorted_unique=True)

    @classmethod
    def from_terms(
        cls,
        n: int,
        terms: Mapping[int, complex],
        *,
        tol: float = TOL,
    ) -> "PauliSoA":
        validate_n(n)
        if not terms:
            return cls.zero(n)
        keys = list(terms.keys())
        vals = np.array([complex(terms[k]) for k in keys], dtype=np.complex128)
        codes = int_codes_to_array(n, keys)
        return cls(n, codes, vals, tol=tol, is_sorted_unique=False)

    @classmethod
    def from_labels(
        cls,
        n: int,
        terms: Mapping[str, complex],
        *,
        tol: float = TOL,
    ) -> "PauliSoA":
        validate_n(n)
        if not terms:
            return cls.zero(n)
        codes = int_codes_to_array(n, [label_to_code(lbl, n) for lbl in terms])
        vals = np.array([complex(v) for v in terms.values()], dtype=np.complex128)
        return cls(n, codes, vals, tol=tol, is_sorted_unique=False)

    @classmethod
    def from_mv(cls, mv: Any, *, tol: float = TOL) -> "PauliSoA":
        if hasattr(mv, "to_mv") and not hasattr(mv, "terms"):
            mv = mv.to_mv()
        return cls.from_terms(mv.n, mv.terms, tol=tol)

    @classmethod
    def from_arrays(
        cls,
        n: int,
        codes: np.ndarray,
        coeffs: np.ndarray,
        *,
        tol: float = TOL,
        is_sorted_unique: bool = False,
    ) -> "PauliSoA":
        return cls(n, codes, coeffs, tol=tol, is_sorted_unique=is_sorted_unique)

    @property
    def is_multiword(self) -> bool:
        return self.codes.ndim == 2

    def nnz(self) -> int:
        return int(self.coeffs.shape[0])

    def nbytes(self) -> int:
        return int(self.codes.nbytes + self.coeffs.nbytes)

    def copy(self) -> "PauliSoA":
        return PauliSoA(
            self.n,
            self.codes.copy(),
            self.coeffs.copy(),
            tol=0.0,
            is_sorted_unique=True,
        )

    def to_terms(self) -> dict[int, complex]:
        ints = array_to_int_codes(self.n, self.codes)
        return dict(zip(ints, self.coeffs.tolist()))

    def to_labels(self, *, tol: float = TOL) -> dict[str, complex]:
        out: dict[str, complex] = {}
        for code_int, val in zip(array_to_int_codes(self.n, self.codes), self.coeffs.tolist()):
            if abs(val) > tol:
                out[code_to_label(self.n, code_int)] = val
        return out

    def to_mv(self) -> Any:
        from .multivector import MV

        return MV(self.n, self.to_terms())

    def _coerce(self, other: Any) -> "PauliSoA":
        if isinstance(other, PauliSoA):
            if other.n != self.n:
                raise ValueError(f"mixed algebra sizes: {self.n} vs {other.n}")
            return other
        if isinstance(other, Number):
            return PauliSoA.scalar(self.n, complex(cast(Any, other)))
        if hasattr(other, "terms") and hasattr(other, "n"):
            if other.n != self.n:
                raise ValueError(f"mixed algebra sizes: {self.n} vs {other.n}")
            return PauliSoA.from_mv(other)
        raise TypeError(f"unsupported operand type: {type(other)!r}")

    def dagger(self) -> "PauliSoA":
        return PauliSoA(
            self.n,
            self.codes.copy(),
            _canonical_complex(self.coeffs.conjugate()),
            tol=0.0,
            is_sorted_unique=True,
        )

    def scalar_part(self) -> complex:
        if self.coeffs.size == 0:
            return 0.0 + 0j
        if self.codes.ndim == 1:
            return complex(self.coeffs[0]) if int(self.codes[0]) == 0 else (0.0 + 0j)
        return complex(self.coeffs[0]) if not bool(np.any(self.codes[0])) else (0.0 + 0j)

    def trace(self) -> complex:
        return (2 ** self.n) * self.scalar_part()

    def norm_hs(self) -> float:
        l2_sq = float(np.sum(self.coeffs.real * self.coeffs.real + self.coeffs.imag * self.coeffs.imag))
        return math.sqrt((2.0 ** self.n) * l2_sq)

    def __neg__(self) -> "PauliSoA":
        return PauliSoA(
            self.n,
            self.codes.copy(),
            _canonical_complex(-self.coeffs),
            tol=0.0,
            is_sorted_unique=True,
        )

    def __add__(self, other: Any) -> "PauliSoA":
        return soa_add(self, self._coerce(other))

    __radd__ = __add__

    def __sub__(self, other: Any) -> "PauliSoA":
        return soa_add(self, -self._coerce(other))

    def __rsub__(self, other: Any) -> "PauliSoA":
        return soa_add(self._coerce(other), -self)

    def __rmul__(self, scalar: Any) -> "PauliSoA":
        if isinstance(scalar, Number):
            z = complex(cast(Any, scalar))
            return PauliSoA(
                self.n,
                self.codes.copy(),
                self.coeffs * z,
                tol=TOL,
                is_sorted_unique=True,
            )
        return NotImplemented

    def __mul__(self, other: Any) -> "PauliSoA":
        if isinstance(other, Number):
            z = complex(cast(Any, other))
            return PauliSoA(
                self.n,
                self.codes.copy(),
                self.coeffs * z,
                tol=TOL,
                is_sorted_unique=True,
            )
        return soa_mul(self, self._coerce(other))

    def __matmul__(self, other: Any) -> "PauliSoA":
        return self * other

    def is_close(self, other: Any, tol: float = 1e-9) -> bool:
        return (self - self._coerce(other)).norm_hs() < tol

    def commutator(self, other: Any, **kwargs: Any) -> "PauliSoA":
        return soa_commutator(self, self._coerce(other), **kwargs)

    def trace_pairing(self, other: Any) -> complex:
        return soa_trace_pairing(self, self._coerce(other))

    def hs_product(self, other: Any) -> complex:
        return soa_hs_product(self, self._coerce(other))


# ---------------------------------------------------------------------------
# Addition & sorted-array intersections
# ---------------------------------------------------------------------------


def soa_add(a: PauliSoA, b: PauliSoA, *, tol: float = TOL) -> PauliSoA:
    """Add two :class:`PauliSoA` operators in the same algebra."""
    if a.n != b.n:
        raise ValueError(f"mixed algebra sizes: {a.n} vs {b.n}")
    if a.is_multiword:
        codes, coeffs = _merge_sorted_multiword(a.codes, a.coeffs, b.codes, b.coeffs, tol=tol)
    else:
        codes, coeffs = _merge_sorted_single(a.codes, a.coeffs, b.codes, b.coeffs, tol=tol)
    return PauliSoA(a.n, codes, coeffs, tol=0.0, is_sorted_unique=True)


def _intersect_sorted(
    codes_a: np.ndarray,
    coeffs_a: np.ndarray,
    codes_b: np.ndarray,
    coeffs_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned ``(va, vb)`` coefficient arrays on the shared support."""
    if coeffs_a.size == 0 or coeffs_b.size == 0:
        empty = np.empty(0, dtype=np.complex128)
        return empty, empty
    if codes_a.ndim == 1:
        ka, kb = codes_a, codes_b
    else:
        ka, kb = _multiword_sort_key(codes_a), _multiword_sort_key(codes_b)
    if ka.size <= kb.size:
        idx = np.searchsorted(kb, ka)
        in_bounds = idx < kb.size
        valid_pos = np.flatnonzero(in_bounds)
        if valid_pos.size == 0:
            empty = np.empty(0, dtype=np.complex128)
            return empty, empty
        matched_b = idx[valid_pos]
        hit = kb[matched_b] == ka[valid_pos]
        return coeffs_a[valid_pos[hit]], coeffs_b[matched_b[hit]]
    idx = np.searchsorted(ka, kb)
    in_bounds = idx < ka.size
    valid_pos = np.flatnonzero(in_bounds)
    if valid_pos.size == 0:
        empty = np.empty(0, dtype=np.complex128)
        return empty, empty
    matched_a = idx[valid_pos]
    hit = ka[matched_a] == kb[valid_pos]
    return coeffs_a[matched_a[hit]], coeffs_b[valid_pos[hit]]


def soa_trace_pairing(a: PauliSoA, b: PauliSoA) -> complex:
    """Bilinear trace pairing ``Tr(A B) / 2^n = sum_w a_w b_w`` via sorted intersection."""
    if a.n != b.n:
        raise ValueError(f"mixed algebra sizes: {a.n} vs {b.n}")
    va, vb = _intersect_sorted(a.codes, a.coeffs, b.codes, b.coeffs)
    if va.size == 0:
        return 0.0 + 0j
    return complex(np.dot(va, vb))


def soa_hs_product(a: PauliSoA, b: PauliSoA) -> complex:
    """Sesquilinear Hilbert-Schmidt pairing ``Tr(A† B) / 2^n = sum_w conj(a_w) b_w``."""
    if a.n != b.n:
        raise ValueError(f"mixed algebra sizes: {a.n} vs {b.n}")
    va, vb = _intersect_sorted(a.codes, a.coeffs, b.codes, b.coeffs)
    if va.size == 0:
        return 0.0 + 0j
    return complex(np.vdot(va, vb))


def _normalize_occupation_mask(n: int, reference: Any) -> int:
    """Convert a ``DeterminantReference``, ``q0``-leftmost bitstring, or int bitmask."""
    if hasattr(reference, "bits") and isinstance(reference.bits, str):
        bits = reference.bits
        if len(bits) != n or set(bits) - {"0", "1"}:
            raise ValueError(f"determinant bits length {len(bits)} does not match n={n}")
        return sum(1 << q for q, ch in enumerate(bits) if ch == "1")
    if isinstance(reference, str):
        if len(reference) != n or set(reference) - {"0", "1"}:
            raise ValueError(
                f"determinant bitstring must be {n} chars of '0'/'1' (q0 leftmost), got {reference!r}"
            )
        return sum(1 << q for q, ch in enumerate(reference) if ch == "1")
    if isinstance(reference, bool) or not isinstance(reference, (int, np.integer)):
        raise TypeError(
            "reference must be DeterminantReference, a binary bitstring, or an int bitmask"
        )
    mask = int(reference)
    if mask < 0 or mask >= (1 << n):
        raise ValueError(f"occupation_mask must be in [0, 2**{n}), got {mask}")
    return mask


def soa_determinant_expectation(op: PauliSoA, reference: Any) -> complex:
    """Exact ``<det| op |det>`` in ``O(nnz)`` vectorized bit ops.

    ``reference`` may be a :class:`clifford_qc.reference_states.DeterminantReference`,
    a ``q0``-leftmost binary string (e.g. ``"1100"``), or an integer bitmask
    with bit ``j`` indicating whether qubit ``j`` is occupied.
    """
    mask = _normalize_occupation_mask(op.n, reference)
    if op.coeffs.size == 0:
        return 0.0 + 0j
    n = op.n
    if not op.is_multiword:
        dtype = op.codes.dtype
        lo = lane_mask_for_width(n, dtype)
        z = (op.codes >> dtype.type(1)) & lo
        x = (op.codes & lo) ^ z
        diag = x == 0
        if not bool(np.any(diag)):
            return 0.0 + 0j
        z_diag = z[diag]
        occ_even = 0
        for j in range(n):
            if (mask >> j) & 1:
                occ_even |= 1 << (2 * j)
        parity = popcount_array(z_diag & dtype.type(occ_even)) & 1
        signs = np.where(parity == 0, 1.0, -1.0)
        return complex(np.dot(op.coeffs[diag], signs))
    masks = multiword_lane_masks(n)
    z = (op.codes >> np.uint64(1)) & masks[None, :]
    x = (op.codes & masks[None, :]) ^ z
    diag = np.all(x == 0, axis=1)
    if not bool(np.any(diag)):
        return 0.0 + 0j
    z_diag = z[diag]
    w = op.codes.shape[1]
    occ_limbs = np.zeros(w, dtype=np.uint64)
    for j in range(n):
        if (mask >> j) & 1:
            limb, off = divmod(j, 32)
            occ_limbs[limb] |= np.uint64(1 << (2 * off))
    xor_acc = _xor_reduce_last_axis(z_diag & occ_limbs[None, :])
    parity = popcount_array(xor_acc) & 1
    signs = np.where(parity == 0, 1.0, -1.0)
    return complex(np.dot(op.coeffs[diag], signs))


# ---------------------------------------------------------------------------
# NumPy vectorized product & commutator kernels
# ---------------------------------------------------------------------------


def _choose_accumulate(n: int, total_pairs: int, accumulate: str) -> str:
    if accumulate not in ("auto", "dense", "sort"):
        raise ValueError(f"accumulate must be 'auto', 'dense', or 'sort', got {accumulate!r}")
    if accumulate == "dense":
        if n > 12:
            raise ValueError(f"dense accumulation is restricted to n <= 12 (4**n bins), got n={n}")
        return "dense"
    if accumulate == "sort":
        return "sort"
    # Heuristic tuned to memory & cache behavior:
    # n <= 8: 4^8 = 65,536 bins (1 MiB for re+im) -> dense wins above ~8k pairs.
    # n <= 10: 4^10 = 1M bins (16 MiB) -> dense wins above ~150k pairs.
    # n <= 12: 4^12 = 16.8M bins (268 MiB) -> dense wins only above ~2M pairs.
    if n <= 8 and total_pairs >= 8_192:
        return "dense"
    if n <= 10 and total_pairs >= 150_000:
        return "dense"
    if n <= 12 and total_pairs >= 2_500_000:
        return "dense"
    return "sort"


def _numpy_mul_single(
    n: int,
    ca: np.ndarray,
    wa: np.ndarray,
    cb: np.ndarray,
    wb: np.ndarray,
    *,
    tol: float,
    chunk_pairs: int,
    accumulate: str,
) -> tuple[np.ndarray, np.ndarray]:
    na, nb = ca.size, cb.size
    dtype = ca.dtype
    if na == 0 or nb == 0:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.complex128)
    total_pairs = na * nb
    mode = _choose_accumulate(n, total_pairs, accumulate)
    lo = lane_mask_for_width(n, dtype)
    one = dtype.type(1)

    za = (ca >> one) & lo
    xa = (ca & lo) ^ za
    pa = popcount_array(xa & za)

    zb = (cb >> one) & lo
    xb = (cb & lo) ^ zb
    pb = popcount_array(xb & zb)

    rows_per_chunk = max(1, min(na, chunk_pairs // max(1, nb)))

    if mode == "dense":
        nbins = 1 << (2 * n)
        acc_re = np.zeros(nbins, dtype=np.float64)
        acc_im = np.zeros(nbins, dtype=np.float64)
        for start in range(0, na, rows_per_chunk):
            stop = min(na, start + rows_per_chunk)
            c = ca[start:stop, None] ^ cb[None, :]
            zc = (c >> one) & lo
            xc = (c & lo) ^ zc
            e = (
                pa[start:stop, None]
                + pb[None, :]
                - popcount_array(xc & zc)
                + (popcount_array(za[start:stop, None] & xb[None, :]) << 1)
            ) & 3
            prod = np.multiply.outer(wa[start:stop], wb) * _PHASE4_C[e]
            c_flat = c.ravel()
            acc_re += np.bincount(c_flat, weights=prod.real.ravel(), minlength=nbins)
            acc_im += np.bincount(c_flat, weights=prod.imag.ravel(), minlength=nbins)
        if tol > 0.0:
            keep = np.flatnonzero((acc_re * acc_re + acc_im * acc_im) > (tol * tol))
        else:
            keep = np.flatnonzero((acc_re != 0.0) | (acc_im != 0.0))
        return keep.astype(dtype), _canonical_complex_from_parts(acc_re[keep], acc_im[keep])

    chunk_codes: list[np.ndarray] = []
    chunk_coeffs: list[np.ndarray] = []
    for start in range(0, na, rows_per_chunk):
        stop = min(na, start + rows_per_chunk)
        c = ca[start:stop, None] ^ cb[None, :]
        zc = (c >> one) & lo
        xc = (c & lo) ^ zc
        e = (
            pa[start:stop, None]
            + pb[None, :]
            - popcount_array(xc & zc)
            + (popcount_array(za[start:stop, None] & xb[None, :]) << 1)
        ) & 3
        prod = np.multiply.outer(wa[start:stop], wb) * _PHASE4_C[e]
        uc, uw = _sort_reduce_single(c.ravel(), prod.real.ravel(), prod.imag.ravel(), tol=0.0)
        if uc.size:
            chunk_codes.append(uc)
            chunk_coeffs.append(uw)
    if not chunk_codes:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.complex128)
    if len(chunk_codes) == 1:
        uc, uw = chunk_codes[0], chunk_coeffs[0]
        if tol > 0.0:
            keep = (uw.real * uw.real + uw.imag * uw.imag) > (tol * tol)
            return uc[keep], _canonical_complex(uw[keep])
        return uc, uw
    all_c = np.concatenate(chunk_codes)
    all_w = np.concatenate(chunk_coeffs)
    return _sort_reduce_single(all_c, all_w.real, all_w.imag, tol=tol)


def _numpy_mul_multiword(
    n: int,
    ca: np.ndarray,
    wa: np.ndarray,
    cb: np.ndarray,
    wb: np.ndarray,
    *,
    tol: float,
    chunk_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    na, w = ca.shape
    nb = cb.shape[0]
    if na == 0 or nb == 0:
        return np.empty((0, w), dtype=np.uint64), np.empty(0, dtype=np.complex128)
    masks = multiword_lane_masks(n)
    one = np.uint64(1)

    za = (ca >> one) & masks[None, :]
    xa = (ca & masks[None, :]) ^ za
    pa = popcount_array(xa & za).sum(axis=1, dtype=np.int32)

    zb = (cb >> one) & masks[None, :]
    xb = (cb & masks[None, :]) ^ zb
    pb = popcount_array(xb & zb).sum(axis=1, dtype=np.int32)

    # Account for w limbs per pair when sizing chunks.
    eff_chunk = max(1, chunk_pairs // max(1, w))
    rows_per_chunk = max(1, min(na, eff_chunk // max(1, nb)))

    chunk_codes: list[np.ndarray] = []
    chunk_coeffs: list[np.ndarray] = []
    for start in range(0, na, rows_per_chunk):
        stop = min(na, start + rows_per_chunk)
        r = stop - start
        c = ca[start:stop, None, :] ^ cb[None, :, :]
        zc = (c >> one) & masks[None, None, :]
        xc = (c & masks[None, None, :]) ^ zc
        pc = popcount_array(xc & zc).sum(axis=2, dtype=np.int32)
        pzx = popcount_array(za[start:stop, None, :] & xb[None, :, :]).sum(axis=2, dtype=np.int32)
        e = (pa[start:stop, None] + pb[None, :] - pc + (pzx << 1)) & 3
        prod = np.multiply.outer(wa[start:stop], wb) * _PHASE4_C[e]
        uc, uw = _sort_reduce_multiword(
            c.reshape(r * nb, w), prod.real.ravel(), prod.imag.ravel(), tol=0.0
        )
        if uc.shape[0]:
            chunk_codes.append(uc)
            chunk_coeffs.append(uw)
    if not chunk_codes:
        return np.empty((0, w), dtype=np.uint64), np.empty(0, dtype=np.complex128)
    if len(chunk_codes) == 1:
        uc, uw = chunk_codes[0], chunk_coeffs[0]
        if tol > 0.0:
            keep = (uw.real * uw.real + uw.imag * uw.imag) > (tol * tol)
            return uc[keep], _canonical_complex(uw[keep])
        return uc, uw
    all_c = np.vstack(chunk_codes)
    all_w = np.concatenate(chunk_coeffs)
    return _sort_reduce_multiword(all_c, all_w.real, all_w.imag, tol=tol)


def soa_anticommute_mask(n: int, codes_a: np.ndarray, codes_b: np.ndarray) -> np.ndarray:
    """Boolean matrix ``M[i, j]`` indicating whether ``codes_a[i]`` and ``codes_b[j]`` anticommute."""
    ca = _validate_code_array(n, codes_a)
    cb = _validate_code_array(n, codes_b)
    if ca.ndim == 1:
        dtype = ca.dtype
        lo = lane_mask_for_width(n, dtype)
        one = dtype.type(1)
        za = (ca >> one) & lo
        xa = (ca & lo) ^ za
        zb = (cb >> one) & lo
        xb = (cb & lo) ^ zb
        symp = (xa[:, None] & zb[None, :]) ^ (za[:, None] & xb[None, :])
        return (popcount_array(symp) & 1).astype(np.bool_)
    masks = multiword_lane_masks(n)
    one = np.uint64(1)
    za = (ca >> one) & masks[None, :]
    xa = (ca & masks[None, :]) ^ za
    zb = (cb >> one) & masks[None, :]
    xb = (cb & masks[None, :]) ^ zb
    # Parity of sum of popcounts across limbs == popcount of limbwise XOR mod 2.
    symp = _xor_reduce_last_axis(
        (xa[:, None, :] & zb[None, :, :]) ^ (za[:, None, :] & xb[None, :, :])
    )
    return (popcount_array(symp) & 1).astype(np.bool_)


def _numpy_commutator_single(
    n: int,
    ca: np.ndarray,
    wa: np.ndarray,
    cb: np.ndarray,
    wb: np.ndarray,
    *,
    tol: float,
    chunk_pairs: int,
    accumulate: str,
) -> tuple[np.ndarray, np.ndarray]:
    na, nb = ca.size, cb.size
    dtype = ca.dtype
    if na == 0 or nb == 0:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.complex128)
    lo = lane_mask_for_width(n, dtype)
    one = dtype.type(1)
    za = (ca >> one) & lo
    xa = (ca & lo) ^ za
    pa = popcount_array(xa & za)
    zb = (cb >> one) & lo
    xb = (cb & lo) ^ zb
    pb = popcount_array(xb & zb)

    mode = _choose_accumulate(n, (na * nb) // 2, accumulate)
    rows_per_chunk = max(1, min(na, chunk_pairs // max(1, nb)))

    if mode == "dense":
        nbins = 1 << (2 * n)
        acc_re = np.zeros(nbins, dtype=np.float64)
        acc_im = np.zeros(nbins, dtype=np.float64)
        for start in range(0, na, rows_per_chunk):
            stop = min(na, start + rows_per_chunk)
            symp = (xa[start:stop, None] & zb[None, :]) ^ (za[start:stop, None] & xb[None, :])
            ia_loc, ib = np.nonzero(popcount_array(symp) & 1)
            if ia_loc.size == 0:
                continue
            ia = ia_loc + start
            c = ca[ia] ^ cb[ib]
            zc = (c >> one) & lo
            xc = (c & lo) ^ zc
            e = (
                pa[ia] + pb[ib] - popcount_array(xc & zc) + (popcount_array(za[ia] & xb[ib]) << 1)
            ) & 3
            prod = (2.0 * wa[ia] * wb[ib]) * _PHASE4_C[e]
            acc_re += np.bincount(c, weights=prod.real, minlength=nbins)
            acc_im += np.bincount(c, weights=prod.imag, minlength=nbins)
        if tol > 0.0:
            keep = np.flatnonzero((acc_re * acc_re + acc_im * acc_im) > (tol * tol))
        else:
            keep = np.flatnonzero((acc_re != 0.0) | (acc_im != 0.0))
        return keep.astype(dtype), _canonical_complex_from_parts(acc_re[keep], acc_im[keep])

    chunk_codes: list[np.ndarray] = []
    chunk_coeffs: list[np.ndarray] = []
    for start in range(0, na, rows_per_chunk):
        stop = min(na, start + rows_per_chunk)
        symp = (xa[start:stop, None] & zb[None, :]) ^ (za[start:stop, None] & xb[None, :])
        ia_loc, ib = np.nonzero(popcount_array(symp) & 1)
        if ia_loc.size == 0:
            continue
        ia = ia_loc + start
        c = ca[ia] ^ cb[ib]
        zc = (c >> one) & lo
        xc = (c & lo) ^ zc
        e = (
            pa[ia] + pb[ib] - popcount_array(xc & zc) + (popcount_array(za[ia] & xb[ib]) << 1)
        ) & 3
        prod = (2.0 * wa[ia] * wb[ib]) * _PHASE4_C[e]
        uc, uw = _sort_reduce_single(c, prod.real, prod.imag, tol=0.0)
        if uc.size:
            chunk_codes.append(uc)
            chunk_coeffs.append(uw)
    if not chunk_codes:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.complex128)
    if len(chunk_codes) == 1:
        uc, uw = chunk_codes[0], chunk_coeffs[0]
        if tol > 0.0:
            keep = (uw.real * uw.real + uw.imag * uw.imag) > (tol * tol)
            return uc[keep], _canonical_complex(uw[keep])
        return uc, uw
    all_c = np.concatenate(chunk_codes)
    all_w = np.concatenate(chunk_coeffs)
    return _sort_reduce_single(all_c, all_w.real, all_w.imag, tol=tol)


def _numpy_commutator_multiword(
    n: int,
    ca: np.ndarray,
    wa: np.ndarray,
    cb: np.ndarray,
    wb: np.ndarray,
    *,
    tol: float,
    chunk_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    na, w = ca.shape
    nb = cb.shape[0]
    if na == 0 or nb == 0:
        return np.empty((0, w), dtype=np.uint64), np.empty(0, dtype=np.complex128)
    masks = multiword_lane_masks(n)
    one = np.uint64(1)
    za = (ca >> one) & masks[None, :]
    xa = (ca & masks[None, :]) ^ za
    pa = popcount_array(xa & za).sum(axis=1, dtype=np.int32)
    zb = (cb >> one) & masks[None, :]
    xb = (cb & masks[None, :]) ^ zb
    pb = popcount_array(xb & zb).sum(axis=1, dtype=np.int32)

    eff_chunk = max(1, chunk_pairs // max(1, w))
    rows_per_chunk = max(1, min(na, eff_chunk // max(1, nb)))
    chunk_codes: list[np.ndarray] = []
    chunk_coeffs: list[np.ndarray] = []
    for start in range(0, na, rows_per_chunk):
        stop = min(na, start + rows_per_chunk)
        symp = _xor_reduce_last_axis(
            (xa[start:stop, None, :] & zb[None, :, :]) ^ (za[start:stop, None, :] & xb[None, :, :])
        )
        ia_loc, ib = np.nonzero(popcount_array(symp) & 1)
        if ia_loc.size == 0:
            continue
        ia = ia_loc + start
        c = ca[ia] ^ cb[ib]
        zc = (c >> one) & masks[None, :]
        xc = (c & masks[None, :]) ^ zc
        pc = popcount_array(xc & zc).sum(axis=1, dtype=np.int32)
        pzx = popcount_array(za[ia] & xb[ib]).sum(axis=1, dtype=np.int32)
        e = (pa[ia] + pb[ib] - pc + (pzx << 1)) & 3
        prod = (2.0 * wa[ia] * wb[ib]) * _PHASE4_C[e]
        uc, uw = _sort_reduce_multiword(c, prod.real, prod.imag, tol=0.0)
        if uc.shape[0]:
            chunk_codes.append(uc)
            chunk_coeffs.append(uw)
    if not chunk_codes:
        return np.empty((0, w), dtype=np.uint64), np.empty(0, dtype=np.complex128)
    if len(chunk_codes) == 1:
        uc, uw = chunk_codes[0], chunk_coeffs[0]
        if tol > 0.0:
            keep = (uw.real * uw.real + uw.imag * uw.imag) > (tol * tol)
            return uc[keep], _canonical_complex(uw[keep])
        return uc, uw
    all_c = np.vstack(chunk_codes)
    all_w = np.concatenate(chunk_coeffs)
    return _sort_reduce_multiword(all_c, all_w.real, all_w.imag, tol=tol)


# ---------------------------------------------------------------------------
# Optional GPU backends (CuPy fused CUDA kernel / PyTorch CUDA SWAR kernel)
# ---------------------------------------------------------------------------

_CUPY_MUL_KERNEL = None
_CUPY_COMM_MASK_KERNEL = None
_CUPY_COMM_PROD_KERNEL = None


def available_gpu_backends() -> dict[str, dict[str, Any]]:
    """Probe optional CUDA backends (``cupy`` and ``torch``) without failing."""
    found: dict[str, dict[str, Any]] = {}
    try:
        import cupy as cp  # type: ignore[import-not-found]

        n_dev = int(cp.cuda.runtime.getDeviceCount())
        if n_dev > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"].decode("utf-8", errors="replace") if isinstance(props["name"], bytes) else str(props["name"])
            found["cupy"] = {
                "version": str(cp.__version__),
                "devices": n_dev,
                "device_name": name,
            }
    except Exception:
        pass
    try:
        import torch  # type: ignore[import-not-found]

        if bool(torch.cuda.is_available()):
            found["torch"] = {
                "version": str(torch.__version__),
                "devices": int(torch.cuda.device_count()),
                "device_name": str(torch.cuda.get_device_name(0)),
            }
    except Exception:
        pass
    return found


def _get_cupy_kernels() -> tuple[Any, Any, Any]:
    global _CUPY_MUL_KERNEL, _CUPY_COMM_MASK_KERNEL, _CUPY_COMM_PROD_KERNEL
    if _CUPY_MUL_KERNEL is not None:
        return _CUPY_MUL_KERNEL, _CUPY_COMM_MASK_KERNEL, _CUPY_COMM_PROD_KERNEL
    import cupy as cp  # type: ignore[import-not-found]

    _CUPY_MUL_KERNEL = cp.ElementwiseKernel(
        "raw uint64 ca, raw uint64 cb, raw float64 war, raw float64 wai, "
        "raw float64 wbr, raw float64 wbi, uint64 lo, int64 nb",
        "uint64 out_c, float64 out_re, float64 out_im",
        """
        long long ia = i / nb;
        long long ib = i - ia * nb;
        unsigned long long a = ca[ia];
        unsigned long long b = cb[ib];
        unsigned long long c = a ^ b;
        unsigned long long za = (a >> 1) & lo;
        unsigned long long xa = (a & lo) ^ za;
        unsigned long long zb = (b >> 1) & lo;
        unsigned long long xb = (b & lo) ^ zb;
        unsigned long long zc = (c >> 1) & lo;
        unsigned long long xc = (c & lo) ^ zc;
        int e = (__popcll(xa & za) + __popcll(xb & zb) - __popcll(xc & zc)
                 + 2 * __popcll(za & xb)) & 3;
        double ar = war[ia], ai = wai[ia];
        double br = wbr[ib], bi = wbi[ib];
        double pr = ar * br - ai * bi;
        double pi = ar * bi + ai * br;
        double rr = (e == 0) ? pr : ((e == 1) ? -pi : ((e == 2) ? -pr : pi));
        double ri = (e == 0) ? pi : ((e == 1) ?  pr : ((e == 2) ? -pi : -pr));
        out_c = c;
        out_re = rr;
        out_im = ri;
        """,
        name="clifford_qc_soa_mul_u64",
    )

    _CUPY_COMM_MASK_KERNEL = cp.ElementwiseKernel(
        "raw uint64 ca, raw uint64 cb, uint64 lo, int64 nb",
        "uint8 out_mask",
        """
        long long ia = i / nb;
        long long ib = i - ia * nb;
        unsigned long long a = ca[ia];
        unsigned long long b = cb[ib];
        unsigned long long za = (a >> 1) & lo;
        unsigned long long xa = (a & lo) ^ za;
        unsigned long long zb = (b >> 1) & lo;
        unsigned long long xb = (b & lo) ^ zb;
        out_mask = (unsigned char)(__popcll((xa & zb) ^ (za & xb)) & 1);
        """,
        name="clifford_qc_soa_anticomm_u64",
    )

    _CUPY_COMM_PROD_KERNEL = cp.ElementwiseKernel(
        "int64 flat_idx, raw uint64 ca, raw uint64 cb, raw float64 war, raw float64 wai, "
        "raw float64 wbr, raw float64 wbi, uint64 lo, int64 nb",
        "uint64 out_c, float64 out_re, float64 out_im",
        """
        long long ia = flat_idx / nb;
        long long ib = flat_idx - ia * nb;
        unsigned long long a = ca[ia];
        unsigned long long b = cb[ib];
        unsigned long long c = a ^ b;
        unsigned long long za = (a >> 1) & lo;
        unsigned long long xa = (a & lo) ^ za;
        unsigned long long zb = (b >> 1) & lo;
        unsigned long long xb = (b & lo) ^ zb;
        unsigned long long zc = (c >> 1) & lo;
        unsigned long long xc = (c & lo) ^ zc;
        int e = (__popcll(xa & za) + __popcll(xb & zb) - __popcll(xc & zc)
                 + 2 * __popcll(za & xb)) & 3;
        double ar = war[ia], ai = wai[ia];
        double br = wbr[ib], bi = wbi[ib];
        double pr = 2.0 * (ar * br - ai * bi);
        double pi = 2.0 * (ar * bi + ai * br);
        double rr = (e == 0) ? pr : ((e == 1) ? -pi : ((e == 2) ? -pr : pi));
        double ri = (e == 0) ? pi : ((e == 1) ?  pr : ((e == 2) ? -pi : -pr));
        out_c = c;
        out_re = rr;
        out_im = ri;
        """,
        name="clifford_qc_soa_comm_prod_u64",
    )
    return _CUPY_MUL_KERNEL, _CUPY_COMM_MASK_KERNEL, _CUPY_COMM_PROD_KERNEL


def _cupy_sort_reduce(cp: Any, codes: Any, re: Any, im: Any, *, tol: float) -> tuple[Any, Any, Any]:
    """GPU sort + zero-cancellation per-group bincount reduction."""
    if codes.size == 0:
        return cp.empty(0, dtype=cp.uint64), cp.empty(0, dtype=cp.float64), cp.empty(0, dtype=cp.float64)
    order = cp.argsort(codes)
    sc = codes[order]
    sre = re[order]
    sim = im[order]
    is_new = cp.empty(sc.size, dtype=cp.bool_)
    is_new[0] = True
    if sc.size > 1:
        is_new[1:] = sc[1:] != sc[:-1]
    starts = cp.flatnonzero(is_new)
    inv = cp.cumsum(is_new, dtype=cp.int64) - 1
    u_count = int(starts.size)
    u_codes = sc[starts]
    u_re = cp.bincount(inv, weights=sre, minlength=u_count)
    u_im = cp.bincount(inv, weights=sim, minlength=u_count)
    if tol > 0.0:
        keep = (u_re * u_re + u_im * u_im) > (tol * tol)
    else:
        keep = (u_re != 0.0) | (u_im != 0.0)
    return u_codes[keep], u_re[keep], u_im[keep]


def _cupy_mul_or_comm_single(
    n: int,
    ca: np.ndarray,
    wa: np.ndarray,
    cb: np.ndarray,
    wb: np.ndarray,
    *,
    commutator: bool,
    tol: float,
    chunk_pairs: int,
    accumulate: str,
) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp  # type: ignore[import-not-found]

    dtype = ca.dtype
    na, nb = ca.size, cb.size
    if na == 0 or nb == 0:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.complex128)
    mul_k, mask_k, comm_k = _get_cupy_kernels()
    lo = np.uint64(((1 << (2 * n)) - 1) // 3 if n > 0 else 0)

    d_ca = cp.asarray(ca, dtype=cp.uint64)
    d_cb = cp.asarray(cb, dtype=cp.uint64)
    d_war = cp.asarray(np.ascontiguousarray(wa.real), dtype=cp.float64)
    d_wai = cp.asarray(np.ascontiguousarray(wa.imag), dtype=cp.float64)
    d_wbr = cp.asarray(np.ascontiguousarray(wb.real), dtype=cp.float64)
    d_wbi = cp.asarray(np.ascontiguousarray(wb.imag), dtype=cp.float64)

    use_dense = (accumulate == "dense") or (
        accumulate == "auto" and n <= 12 and (na * nb) >= (16_384 if n <= 10 else 1_000_000)
    )
    if accumulate == "dense" and n > 12:
        raise ValueError(f"dense accumulation is restricted to n <= 12, got n={n}")

    rows_per_chunk = max(1, min(na, chunk_pairs // max(1, nb)))
    if use_dense:
        nbins = 1 << (2 * n)
        acc_re = cp.zeros(nbins, dtype=cp.float64)
        acc_im = cp.zeros(nbins, dtype=cp.float64)
        for start in range(0, na, rows_per_chunk):
            stop = min(na, start + rows_per_chunk)
            r = stop - start
            count = r * nb
            if commutator:
                mask = cp.empty(count, dtype=cp.uint8)
                mask_k(d_ca[start:stop], d_cb, lo, np.int64(nb), mask)
                active = cp.flatnonzero(mask).astype(cp.int64, copy=False)
                if active.size == 0:
                    continue
                out_c = cp.empty(active.size, dtype=cp.uint64)
                out_re = cp.empty(active.size, dtype=cp.float64)
                out_im = cp.empty(active.size, dtype=cp.float64)
                comm_k(
                    active,
                    d_ca[start:stop],
                    d_cb,
                    d_war[start:stop],
                    d_wai[start:stop],
                    d_wbr,
                    d_wbi,
                    lo,
                    np.int64(nb),
                    out_c,
                    out_re,
                    out_im,
                )
            else:
                out_c = cp.empty(count, dtype=cp.uint64)
                out_re = cp.empty(count, dtype=cp.float64)
                out_im = cp.empty(count, dtype=cp.float64)
                mul_k(
                    d_ca[start:stop],
                    d_cb,
                    d_war[start:stop],
                    d_wai[start:stop],
                    d_wbr,
                    d_wbi,
                    lo,
                    np.int64(nb),
                    out_c,
                    out_re,
                    out_im,
                )
            c_idx = out_c.astype(cp.int64)
            acc_re += cp.bincount(c_idx, weights=out_re, minlength=nbins)
            acc_im += cp.bincount(c_idx, weights=out_im, minlength=nbins)
        if tol > 0.0:
            keep = cp.flatnonzero((acc_re * acc_re + acc_im * acc_im) > (tol * tol))
        else:
            keep = cp.flatnonzero((acc_re != 0.0) | (acc_im != 0.0))
        h_codes = cp.asnumpy(keep).astype(dtype)
        h_re = cp.asnumpy(acc_re[keep])
        h_im = cp.asnumpy(acc_im[keep])
        return h_codes, _canonical_complex_from_parts(h_re, h_im)

    c_parts: list[Any] = []
    re_parts: list[Any] = []
    im_parts: list[Any] = []
    for start in range(0, na, rows_per_chunk):
        stop = min(na, start + rows_per_chunk)
        r = stop - start
        count = r * nb
        if commutator:
            mask = cp.empty(count, dtype=cp.uint8)
            mask_k(d_ca[start:stop], d_cb, lo, np.int64(nb), mask)
            active = cp.flatnonzero(mask).astype(cp.int64, copy=False)
            if active.size == 0:
                continue
            out_c = cp.empty(active.size, dtype=cp.uint64)
            out_re = cp.empty(active.size, dtype=cp.float64)
            out_im = cp.empty(active.size, dtype=cp.float64)
            comm_k(
                active,
                d_ca[start:stop],
                d_cb,
                d_war[start:stop],
                d_wai[start:stop],
                d_wbr,
                d_wbi,
                lo,
                np.int64(nb),
                out_c,
                out_re,
                out_im,
            )
        else:
            out_c = cp.empty(count, dtype=cp.uint64)
            out_re = cp.empty(count, dtype=cp.float64)
            out_im = cp.empty(count, dtype=cp.float64)
            mul_k(
                d_ca[start:stop],
                d_cb,
                d_war[start:stop],
                d_wai[start:stop],
                d_wbr,
                d_wbi,
                lo,
                np.int64(nb),
                out_c,
                out_re,
                out_im,
            )
        uc, ure, uim = _cupy_sort_reduce(cp, out_c, out_re, out_im, tol=0.0)
        if uc.size:
            c_parts.append(uc)
            re_parts.append(ure)
            im_parts.append(uim)
    if not c_parts:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.complex128)
    if len(c_parts) == 1:
        uc, ure, uim = c_parts[0], re_parts[0], im_parts[0]
        if tol > 0.0:
            keep = (ure * ure + uim * uim) > (tol * tol)
            uc, ure, uim = uc[keep], ure[keep], uim[keep]
    else:
        uc, ure, uim = _cupy_sort_reduce(
            cp, cp.concatenate(c_parts), cp.concatenate(re_parts), cp.concatenate(im_parts), tol=tol
        )
    return cp.asnumpy(uc).astype(dtype), _canonical_complex_from_parts(
        cp.asnumpy(ure), cp.asnumpy(uim)
    )


_CUPY_MW_MUL_KERNEL = None
_CUPY_MW_COMM_MASK_KERNEL = None
_CUPY_MW_COMM_PROD_KERNEL = None


def _get_cupy_multiword_kernels() -> tuple[Any, Any, Any]:
    global _CUPY_MW_MUL_KERNEL, _CUPY_MW_COMM_MASK_KERNEL, _CUPY_MW_COMM_PROD_KERNEL
    if _CUPY_MW_MUL_KERNEL is not None:
        return _CUPY_MW_MUL_KERNEL, _CUPY_MW_COMM_MASK_KERNEL, _CUPY_MW_COMM_PROD_KERNEL
    import cupy as cp  # type: ignore[import-not-found]

    _CUPY_MW_MUL_KERNEL = cp.ElementwiseKernel(
        "raw uint64 ca, raw uint64 cb, raw float64 war, raw float64 wai, "
        "raw float64 wbr, raw float64 wbi, raw uint64 masks, int64 nb, int64 w",
        "raw uint64 out_c, float64 out_re, float64 out_im",
        """
        long long ia = i / nb;
        long long ib = i - ia * nb;
        int e = 0;
        for (long long k = 0; k < w; ++k) {
            unsigned long long lo = masks[k];
            unsigned long long a = ca[ia * w + k];
            unsigned long long b = cb[ib * w + k];
            unsigned long long c = a ^ b;
            out_c[i * w + k] = c;
            unsigned long long za = (a >> 1) & lo;
            unsigned long long xa = (a & lo) ^ za;
            unsigned long long zb = (b >> 1) & lo;
            unsigned long long xb = (b & lo) ^ zb;
            unsigned long long zc = (c >> 1) & lo;
            unsigned long long xc = (c & lo) ^ zc;
            e += __popcll(xa & za) + __popcll(xb & zb) - __popcll(xc & zc)
                 + 2 * __popcll(za & xb);
        }
        e &= 3;
        double ar = war[ia], ai = wai[ia];
        double br = wbr[ib], bi = wbi[ib];
        double pr = ar * br - ai * bi;
        double pi = ar * bi + ai * br;
        out_re = (e == 0) ? pr : ((e == 1) ? -pi : ((e == 2) ? -pr : pi));
        out_im = (e == 0) ? pi : ((e == 1) ?  pr : ((e == 2) ? -pi : -pr));
        """,
        name="clifford_qc_soa_mw_mul_u64",
    )

    _CUPY_MW_COMM_MASK_KERNEL = cp.ElementwiseKernel(
        "raw uint64 ca, raw uint64 cb, raw uint64 masks, int64 nb, int64 w",
        "uint8 out_mask",
        """
        long long ia = i / nb;
        long long ib = i - ia * nb;
        unsigned long long acc = 0ULL;
        for (long long k = 0; k < w; ++k) {
            unsigned long long lo = masks[k];
            unsigned long long a = ca[ia * w + k];
            unsigned long long b = cb[ib * w + k];
            unsigned long long za = (a >> 1) & lo;
            unsigned long long xa = (a & lo) ^ za;
            unsigned long long zb = (b >> 1) & lo;
            unsigned long long xb = (b & lo) ^ zb;
            acc ^= (xa & zb) ^ (za & xb);
        }
        out_mask = (unsigned char)(__popcll(acc) & 1);
        """,
        name="clifford_qc_soa_mw_anticomm_u64",
    )

    _CUPY_MW_COMM_PROD_KERNEL = cp.ElementwiseKernel(
        "int64 flat_idx, raw uint64 ca, raw uint64 cb, raw float64 war, raw float64 wai, "
        "raw float64 wbr, raw float64 wbi, raw uint64 masks, int64 nb, int64 w",
        "raw uint64 out_c, float64 out_re, float64 out_im",
        """
        long long ia = flat_idx / nb;
        long long ib = flat_idx - ia * nb;
        int e = 0;
        for (long long k = 0; k < w; ++k) {
            unsigned long long lo = masks[k];
            unsigned long long a = ca[ia * w + k];
            unsigned long long b = cb[ib * w + k];
            unsigned long long c = a ^ b;
            out_c[i * w + k] = c;
            unsigned long long za = (a >> 1) & lo;
            unsigned long long xa = (a & lo) ^ za;
            unsigned long long zb = (b >> 1) & lo;
            unsigned long long xb = (b & lo) ^ zb;
            unsigned long long zc = (c >> 1) & lo;
            unsigned long long xc = (c & lo) ^ zc;
            e += __popcll(xa & za) + __popcll(xb & zb) - __popcll(xc & zc)
                 + 2 * __popcll(za & xb);
        }
        e &= 3;
        double ar = war[ia], ai = wai[ia];
        double br = wbr[ib], bi = wbi[ib];
        double pr = 2.0 * (ar * br - ai * bi);
        double pi = 2.0 * (ar * bi + ai * br);
        out_re = (e == 0) ? pr : ((e == 1) ? -pi : ((e == 2) ? -pr : pi));
        out_im = (e == 0) ? pi : ((e == 1) ?  pr : ((e == 2) ? -pi : -pr));
        """,
        name="clifford_qc_soa_mw_comm_prod_u64",
    )
    return _CUPY_MW_MUL_KERNEL, _CUPY_MW_COMM_MASK_KERNEL, _CUPY_MW_COMM_PROD_KERNEL


def _cupy_sort_reduce_multiword(
    cp: Any, codes: Any, re: Any, im: Any, *, tol: float
) -> tuple[Any, Any, Any]:
    w = int(codes.shape[1])
    if codes.shape[0] == 0:
        return (
            cp.empty((0, w), dtype=cp.uint64),
            cp.empty(0, dtype=cp.float64),
            cp.empty(0, dtype=cp.float64),
        )
    # Little-endian limbs: limb[0] is LSB, limb[w-1] is MSB (primary lexsort key).
    order = cp.lexsort(codes.T)
    sc = codes[order]
    sre = re[order]
    sim = im[order]
    is_new = cp.empty(sc.shape[0], dtype=cp.bool_)
    is_new[0] = True
    if sc.shape[0] > 1:
        is_new[1:] = cp.any(sc[1:] != sc[:-1], axis=1)
    starts = cp.flatnonzero(is_new)
    inv = cp.cumsum(is_new, dtype=cp.int64) - 1
    u_count = int(starts.size)
    u_codes = sc[starts]
    u_re = cp.bincount(inv, weights=sre, minlength=u_count)
    u_im = cp.bincount(inv, weights=sim, minlength=u_count)
    if tol > 0.0:
        keep = (u_re * u_re + u_im * u_im) > (tol * tol)
    else:
        keep = (u_re != 0.0) | (u_im != 0.0)
    return u_codes[keep], u_re[keep], u_im[keep]


def _cupy_mul_or_comm_multiword(
    n: int,
    ca: np.ndarray,
    wa: np.ndarray,
    cb: np.ndarray,
    wb: np.ndarray,
    *,
    commutator: bool,
    tol: float,
    chunk_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp  # type: ignore[import-not-found]

    na, w = ca.shape
    nb = cb.shape[0]
    if na == 0 or nb == 0:
        return np.empty((0, w), dtype=np.uint64), np.empty(0, dtype=np.complex128)
    mul_k, mask_k, comm_k = _get_cupy_multiword_kernels()
    d_masks = cp.asarray(multiword_lane_masks(n), dtype=cp.uint64)
    d_ca = cp.asarray(np.ascontiguousarray(ca), dtype=cp.uint64)
    d_cb = cp.asarray(np.ascontiguousarray(cb), dtype=cp.uint64)
    d_war = cp.asarray(np.ascontiguousarray(wa.real), dtype=cp.float64)
    d_wai = cp.asarray(np.ascontiguousarray(wa.imag), dtype=cp.float64)
    d_wbr = cp.asarray(np.ascontiguousarray(wb.real), dtype=cp.float64)
    d_wbi = cp.asarray(np.ascontiguousarray(wb.imag), dtype=cp.float64)

    eff_chunk = max(1, chunk_pairs // max(1, w))
    rows_per_chunk = max(1, min(na, eff_chunk // max(1, nb)))
    c_parts: list[Any] = []
    re_parts: list[Any] = []
    im_parts: list[Any] = []
    for start in range(0, na, rows_per_chunk):
        stop = min(na, start + rows_per_chunk)
        r = stop - start
        count = r * nb
        if commutator:
            mask = cp.empty(count, dtype=cp.uint8)
            mask_k(d_ca[start:stop], d_cb, d_masks, np.int64(nb), np.int64(w), mask)
            active = cp.flatnonzero(mask).astype(cp.int64, copy=False)
            if active.size == 0:
                continue
            out_c = cp.empty((int(active.size), w), dtype=cp.uint64)
            out_re = cp.empty(int(active.size), dtype=cp.float64)
            out_im = cp.empty(int(active.size), dtype=cp.float64)
            comm_k(
                active,
                d_ca[start:stop],
                d_cb,
                d_war[start:stop],
                d_wai[start:stop],
                d_wbr,
                d_wbi,
                d_masks,
                np.int64(nb),
                np.int64(w),
                out_c,
                out_re,
                out_im,
            )
        else:
            out_c = cp.empty((count, w), dtype=cp.uint64)
            out_re = cp.empty(count, dtype=cp.float64)
            out_im = cp.empty(count, dtype=cp.float64)
            mul_k(
                d_ca[start:stop],
                d_cb,
                d_war[start:stop],
                d_wai[start:stop],
                d_wbr,
                d_wbi,
                d_masks,
                np.int64(nb),
                np.int64(w),
                out_c,
                out_re,
                out_im,
            )
        uc, ure, uim = _cupy_sort_reduce_multiword(cp, out_c, out_re, out_im, tol=0.0)
        if uc.shape[0]:
            c_parts.append(uc)
            re_parts.append(ure)
            im_parts.append(uim)
    if not c_parts:
        return np.empty((0, w), dtype=np.uint64), np.empty(0, dtype=np.complex128)
    if len(c_parts) == 1:
        uc, ure, uim = c_parts[0], re_parts[0], im_parts[0]
        if tol > 0.0:
            keep = (ure * ure + uim * uim) > (tol * tol)
            uc, ure, uim = uc[keep], ure[keep], uim[keep]
    else:
        uc, ure, uim = _cupy_sort_reduce_multiword(
            cp,
            cp.vstack(c_parts),
            cp.concatenate(re_parts),
            cp.concatenate(im_parts),
            tol=tol,
        )
    return cp.asnumpy(uc).astype(np.uint64), _canonical_complex_from_parts(
        cp.asnumpy(ure), cp.asnumpy(uim)
    )


def synchronize_backend(backend: str) -> None:
    """Block until asynchronous GPU work on ``backend`` finishes (no-op on NumPy)."""
    if backend == "cupy":
        import cupy as cp  # type: ignore[import-not-found]

        cp.cuda.Stream.null.synchronize()
    elif backend == "torch":
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _torch_popcount_i64(x: Any) -> Any:
    """Bit-exact SWAR popcount for non-negative ``torch.int64`` (<= 62 bits)."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((x * 0x0101010101010101) >> 56) & 0x7F


def _torch_sort_reduce(torch: Any, codes: Any, re: Any, im: Any, *, tol: float) -> tuple[Any, Any, Any]:
    if codes.numel() == 0:
        dev = codes.device
        return (
            torch.empty(0, dtype=torch.int64, device=dev),
            torch.empty(0, dtype=torch.float64, device=dev),
            torch.empty(0, dtype=torch.float64, device=dev),
        )
    sc, order = torch.sort(codes)
    sre = re[order]
    sim = im[order]
    is_new = torch.ones(sc.numel(), dtype=torch.bool, device=sc.device)
    if sc.numel() > 1:
        is_new[1:] = sc[1:] != sc[:-1]
    starts = torch.nonzero(is_new, as_tuple=False).squeeze(-1)
    inv = torch.cumsum(is_new.to(torch.int64), dim=0) - 1
    u_count = int(starts.numel())
    u_codes = sc[starts]
    u_re = torch.zeros(u_count, dtype=torch.float64, device=sc.device).scatter_add_(0, inv, sre)
    u_im = torch.zeros(u_count, dtype=torch.float64, device=sc.device).scatter_add_(0, inv, sim)
    if tol > 0.0:
        keep = (u_re * u_re + u_im * u_im) > (tol * tol)
    else:
        keep = (u_re != 0.0) | (u_im != 0.0)
    return u_codes[keep], u_re[keep], u_im[keep]


def _torch_mul_or_comm_single(
    n: int,
    ca: np.ndarray,
    wa: np.ndarray,
    cb: np.ndarray,
    wb: np.ndarray,
    *,
    commutator: bool,
    tol: float,
    chunk_pairs: int,
    accumulate: str,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    import torch  # type: ignore[import-not-found]

    if n > 31:
        # Delegate n == 32 (unsigned bit 63) cleanly to NumPy/CuPy fallback.
        if commutator:
            return _numpy_commutator_single(
                n, ca, wa, cb, wb, tol=tol, chunk_pairs=chunk_pairs, accumulate=accumulate
            )
        return _numpy_mul_single(
            n, ca, wa, cb, wb, tol=tol, chunk_pairs=chunk_pairs, accumulate=accumulate
        )

    dtype = ca.dtype
    na, nb = ca.size, cb.size
    if na == 0 or nb == 0:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.complex128)
    dev = torch.device(device)
    lo = int(((1 << (2 * n)) - 1) // 3) if n > 0 else 0

    t_ca = torch.as_tensor(ca.astype(np.int64), dtype=torch.int64, device=dev)
    t_cb = torch.as_tensor(cb.astype(np.int64), dtype=torch.int64, device=dev)
    t_war = torch.as_tensor(np.ascontiguousarray(wa.real), dtype=torch.float64, device=dev)
    t_wai = torch.as_tensor(np.ascontiguousarray(wa.imag), dtype=torch.float64, device=dev)
    t_wbr = torch.as_tensor(np.ascontiguousarray(wb.real), dtype=torch.float64, device=dev)
    t_wbi = torch.as_tensor(np.ascontiguousarray(wb.imag), dtype=torch.float64, device=dev)
    ph_re = torch.tensor([1.0, 0.0, -1.0, 0.0], dtype=torch.float64, device=dev)
    ph_im = torch.tensor([0.0, 1.0, 0.0, -1.0], dtype=torch.float64, device=dev)

    za = (t_ca >> 1) & lo
    xa = (t_ca & lo) ^ za
    pa = _torch_popcount_i64(xa & za)
    zb = (t_cb >> 1) & lo
    xb = (t_cb & lo) ^ zb
    pb = _torch_popcount_i64(xb & zb)

    use_dense = (accumulate == "dense") or (
        accumulate == "auto" and n <= 12 and (na * nb) >= (16_384 if n <= 10 else 1_000_000)
    )
    if accumulate == "dense" and n > 12:
        raise ValueError(f"dense accumulation is restricted to n <= 12, got n={n}")

    rows_per_chunk = max(1, min(na, chunk_pairs // max(1, nb)))
    if use_dense:
        nbins = 1 << (2 * n)
        acc_re = torch.zeros(nbins, dtype=torch.float64, device=dev)
        acc_im = torch.zeros(nbins, dtype=torch.float64, device=dev)
        for start in range(0, na, rows_per_chunk):
            stop = min(na, start + rows_per_chunk)
            if commutator:
                symp = (xa[start:stop, None] & zb[None, :]) ^ (za[start:stop, None] & xb[None, :])
                idx_2d = torch.nonzero(_torch_popcount_i64(symp) & 1, as_tuple=False)
                if idx_2d.numel() == 0:
                    continue
                ia = idx_2d[:, 0] + start
                ib = idx_2d[:, 1]
                c = t_ca[ia] ^ t_cb[ib]
                zc = (c >> 1) & lo
                xc = (c & lo) ^ zc
                e = (
                    pa[ia] + pb[ib] - _torch_popcount_i64(xc & zc)
                    + (_torch_popcount_i64(za[ia] & xb[ib]) << 1)
                ) & 3
                pr = 2.0 * (t_war[ia] * t_wbr[ib] - t_wai[ia] * t_wbi[ib])
                pi = 2.0 * (t_war[ia] * t_wbi[ib] + t_wai[ia] * t_wbr[ib])
                rr = pr * ph_re[e] - pi * ph_im[e]
                ri = pr * ph_im[e] + pi * ph_re[e]
                acc_re.scatter_add_(0, c, rr)
                acc_im.scatter_add_(0, c, ri)
            else:
                c = (t_ca[start:stop, None] ^ t_cb[None, :]).reshape(-1)
                zc = (c >> 1) & lo
                xc = (c & lo) ^ zc
                e = (
                    pa[start:stop, None]
                    + pb[None, :]
                    - _torch_popcount_i64(xc & zc).reshape(stop - start, nb)
                    + (_torch_popcount_i64(za[start:stop, None] & xb[None, :]) << 1)
                ).reshape(-1) & 3
                pr = (
                    t_war[start:stop, None] * t_wbr[None, :]
                    - t_wai[start:stop, None] * t_wbi[None, :]
                ).reshape(-1)
                pi = (
                    t_war[start:stop, None] * t_wbi[None, :]
                    + t_wai[start:stop, None] * t_wbr[None, :]
                ).reshape(-1)
                rr = pr * ph_re[e] - pi * ph_im[e]
                ri = pr * ph_im[e] + pi * ph_re[e]
                acc_re.scatter_add_(0, c, rr)
                acc_im.scatter_add_(0, c, ri)
        if tol > 0.0:
            keep = torch.nonzero((acc_re * acc_re + acc_im * acc_im) > (tol * tol), as_tuple=False).squeeze(-1)
        else:
            keep = torch.nonzero((acc_re != 0.0) | (acc_im != 0.0), as_tuple=False).squeeze(-1)
        h_codes = keep.cpu().numpy().astype(dtype)
        h_re = acc_re[keep].cpu().numpy()
        h_im = acc_im[keep].cpu().numpy()
        return h_codes, _canonical_complex_from_parts(h_re, h_im)

    c_parts: list[Any] = []
    re_parts: list[Any] = []
    im_parts: list[Any] = []
    for start in range(0, na, rows_per_chunk):
        stop = min(na, start + rows_per_chunk)
        if commutator:
            symp = (xa[start:stop, None] & zb[None, :]) ^ (za[start:stop, None] & xb[None, :])
            idx_2d = torch.nonzero(_torch_popcount_i64(symp) & 1, as_tuple=False)
            if idx_2d.numel() == 0:
                continue
            ia = idx_2d[:, 0] + start
            ib = idx_2d[:, 1]
            c = t_ca[ia] ^ t_cb[ib]
            zc = (c >> 1) & lo
            xc = (c & lo) ^ zc
            e = (
                pa[ia] + pb[ib] - _torch_popcount_i64(xc & zc)
                + (_torch_popcount_i64(za[ia] & xb[ib]) << 1)
            ) & 3
            pr = 2.0 * (t_war[ia] * t_wbr[ib] - t_wai[ia] * t_wbi[ib])
            pi = 2.0 * (t_war[ia] * t_wbi[ib] + t_wai[ia] * t_wbr[ib])
            rr = pr * ph_re[e] - pi * ph_im[e]
            ri = pr * ph_im[e] + pi * ph_re[e]
        else:
            c = (t_ca[start:stop, None] ^ t_cb[None, :]).reshape(-1)
            zc = (c >> 1) & lo
            xc = (c & lo) ^ zc
            e = (
                pa[start:stop, None]
                + pb[None, :]
                - _torch_popcount_i64(xc & zc).reshape(stop - start, nb)
                + (_torch_popcount_i64(za[start:stop, None] & xb[None, :]) << 1)
            ).reshape(-1) & 3
            pr = (
                t_war[start:stop, None] * t_wbr[None, :]
                - t_wai[start:stop, None] * t_wbi[None, :]
            ).reshape(-1)
            pi = (
                t_war[start:stop, None] * t_wbi[None, :]
                + t_wai[start:stop, None] * t_wbr[None, :]
            ).reshape(-1)
            rr = pr * ph_re[e] - pi * ph_im[e]
            ri = pr * ph_im[e] + pi * ph_re[e]
        uc, ure, uim = _torch_sort_reduce(torch, c, rr, ri, tol=0.0)
        if uc.numel():
            c_parts.append(uc)
            re_parts.append(ure)
            im_parts.append(uim)
    if not c_parts:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.complex128)
    if len(c_parts) == 1:
        uc, ure, uim = c_parts[0], re_parts[0], im_parts[0]
        if tol > 0.0:
            keep = (ure * ure + uim * uim) > (tol * tol)
            uc, ure, uim = uc[keep], ure[keep], uim[keep]
    else:
        uc, ure, uim = _torch_sort_reduce(
            torch, torch.cat(c_parts), torch.cat(re_parts), torch.cat(im_parts), tol=tol
        )
    return uc.cpu().numpy().astype(dtype), _canonical_complex_from_parts(
        ure.cpu().numpy(), uim.cpu().numpy()
    )


def _resolve_backend(backend: str, total_pairs: int, n: int) -> str:
    if backend not in ("numpy", "cupy", "torch", "auto"):
        raise ValueError(f"backend must be 'numpy', 'cupy', 'torch', or 'auto', got {backend!r}")
    if backend == "numpy":
        return "numpy"
    if backend == "auto":
        # GPU dispatch only pays off on larger pair counts; keep small calls on NumPy.
        if total_pairs < 250_000:
            return "numpy"
        gpus = available_gpu_backends()
        if "cupy" in gpus:
            return "cupy"
        if "torch" in gpus and n <= 31:
            return "torch"
        return "numpy"
    gpus = available_gpu_backends()
    if backend not in gpus:
        raise RuntimeError(f"requested GPU backend {backend!r} is unavailable (available: {sorted(gpus)})")
    return backend


# ---------------------------------------------------------------------------
# Public product and commutator entry points
# ---------------------------------------------------------------------------


def soa_mul(
    a: PauliSoA,
    b: PauliSoA,
    *,
    tol: float = TOL,
    chunk_pairs: int | None = None,
    accumulate: str = "auto",
    backend: str = "numpy",
) -> PauliSoA:
    """Multiply two :class:`PauliSoA` operators via batched XOR + popcount."""
    if a.n != b.n:
        raise ValueError(f"mixed algebra sizes: {a.n} vs {b.n}")
    n = a.n
    total_pairs = a.nnz() * b.nnz()
    if total_pairs == 0:
        return PauliSoA.zero(n)
    chosen = _resolve_backend(backend, total_pairs, n)
    chunk = chunk_pairs if chunk_pairs is not None else (
        DEFAULT_GPU_CHUNK_PAIRS if chosen != "numpy" else DEFAULT_CPU_CHUNK_PAIRS
    )
    if chunk < 1:
        raise ValueError(f"chunk_pairs must be >= 1, got {chunk}")
    if a.is_multiword:
        if chosen == "cupy":
            codes, coeffs = _cupy_mul_or_comm_multiword(
                n,
                a.codes,
                a.coeffs,
                b.codes,
                b.coeffs,
                commutator=False,
                tol=tol,
                chunk_pairs=chunk,
            )
        else:
            codes, coeffs = _numpy_mul_multiword(
                n, a.codes, a.coeffs, b.codes, b.coeffs, tol=tol, chunk_pairs=chunk
            )
    elif chosen == "cupy":
        codes, coeffs = _cupy_mul_or_comm_single(
            n,
            a.codes,
            a.coeffs,
            b.codes,
            b.coeffs,
            commutator=False,
            tol=tol,
            chunk_pairs=chunk,
            accumulate=accumulate,
        )
    elif chosen == "torch":
        codes, coeffs = _torch_mul_or_comm_single(
            n,
            a.codes,
            a.coeffs,
            b.codes,
            b.coeffs,
            commutator=False,
            tol=tol,
            chunk_pairs=chunk,
            accumulate=accumulate,
        )
    else:
        codes, coeffs = _numpy_mul_single(
            n,
            a.codes,
            a.coeffs,
            b.codes,
            b.coeffs,
            tol=tol,
            chunk_pairs=chunk,
            accumulate=accumulate,
        )
    return PauliSoA(n, codes, coeffs, tol=0.0, is_sorted_unique=True)


def soa_commutator(
    a: PauliSoA,
    b: PauliSoA,
    *,
    tol: float = TOL,
    chunk_pairs: int | None = None,
    accumulate: str = "auto",
    backend: str = "numpy",
) -> PauliSoA:
    """Compute ``[A, B] = A B - B A`` with symplectic anticommutation pre-filtering."""
    if a.n != b.n:
        raise ValueError(f"mixed algebra sizes: {a.n} vs {b.n}")
    n = a.n
    total_pairs = a.nnz() * b.nnz()
    if total_pairs == 0:
        return PauliSoA.zero(n)
    chosen = _resolve_backend(backend, total_pairs, n)
    chunk = chunk_pairs if chunk_pairs is not None else (
        DEFAULT_GPU_CHUNK_PAIRS if chosen != "numpy" else DEFAULT_CPU_CHUNK_PAIRS
    )
    if chunk < 1:
        raise ValueError(f"chunk_pairs must be >= 1, got {chunk}")
    if a.is_multiword:
        if chosen == "cupy":
            codes, coeffs = _cupy_mul_or_comm_multiword(
                n,
                a.codes,
                a.coeffs,
                b.codes,
                b.coeffs,
                commutator=True,
                tol=tol,
                chunk_pairs=chunk,
            )
        else:
            codes, coeffs = _numpy_commutator_multiword(
                n, a.codes, a.coeffs, b.codes, b.coeffs, tol=tol, chunk_pairs=chunk
            )
    elif chosen == "cupy":
        codes, coeffs = _cupy_mul_or_comm_single(
            n,
            a.codes,
            a.coeffs,
            b.codes,
            b.coeffs,
            commutator=True,
            tol=tol,
            chunk_pairs=chunk,
            accumulate=accumulate,
        )
    elif chosen == "torch":
        codes, coeffs = _torch_mul_or_comm_single(
            n,
            a.codes,
            a.coeffs,
            b.codes,
            b.coeffs,
            commutator=True,
            tol=tol,
            chunk_pairs=chunk,
            accumulate=accumulate,
        )
    else:
        codes, coeffs = _numpy_commutator_single(
            n,
            a.codes,
            a.coeffs,
            b.codes,
            b.coeffs,
            tol=tol,
            chunk_pairs=chunk,
            accumulate=accumulate,
        )
    return PauliSoA(n, codes, coeffs, tol=0.0, is_sorted_unique=True)


def soa_mul_terms(
    n: int,
    terms_a: Mapping[int, complex],
    terms_b: Mapping[int, complex],
    *,
    tol: float = TOL,
    accumulate: str = "auto",
    backend: str = "numpy",
) -> dict[int, complex]:
    """Multiply two ``dict[int, complex]`` term maps via SoA without touching memos."""
    if not terms_a or not terms_b:
        return {}
    sa = PauliSoA.from_terms(n, terms_a, tol=0.0)
    sb = PauliSoA.from_terms(n, terms_b, tol=0.0)
    prod = soa_mul(sa, sb, tol=tol, accumulate=accumulate, backend=backend)
    return prod.to_terms()


def soa_commutator_terms(
    n: int,
    terms_a: Mapping[int, complex],
    terms_b: Mapping[int, complex],
    *,
    tol: float = 0.0,
    accumulate: str = "auto",
    backend: str = "numpy",
) -> dict[int, complex]:
    """Compute ``[A, B]`` on ``dict[int, complex]`` term maps via symplectic pre-filter."""
    if not terms_a or not terms_b:
        return {}
    sa = PauliSoA.from_terms(n, terms_a, tol=0.0)
    sb = PauliSoA.from_terms(n, terms_b, tol=0.0)
    comm = soa_commutator(sa, sb, tol=tol, accumulate=accumulate, backend=backend)
    return comm.to_terms()


# ---------------------------------------------------------------------------
# Multiword-safe global word table & single-pass collated SoA bank
# ---------------------------------------------------------------------------


class SoAWordTable:
    """Sorted canonical Pauli-word universe supporting arbitrary qubit counts ``n``.

    Unlike :class:`clifford_qc.subspace.packed.GlobalWordTable` (which stores
    ``int64`` codes and therefore stops at 31 qubits), ``SoAWordTable`` stores
    width-selected unsigned codes (``uint16``/``uint32``/``uint64`` for
    ``n <= 32`` and 2-D ``uint64`` limbs for ``n > 32``) in canonical sorted
    order and resolves lookups by vectorized ``np.searchsorted``.
    """

    __slots__ = ("n", "codes", "_sort_keys")

    def __init__(self, n: int, sorted_unique_codes: np.ndarray) -> None:
        validate_n(n)
        self.n = n
        self.codes = _validate_code_array(n, sorted_unique_codes)
        self._sort_keys = (
            self.codes if self.codes.ndim == 1 else _multiword_sort_key(self.codes)
        )

    @classmethod
    def from_code_arrays(cls, n: int, arrays: Sequence[np.ndarray]) -> "SoAWordTable":
        validate_n(n)
        w = word_width_for_n(n)
        non_empty = [a for a in arrays if a.shape[0] > 0]
        if not non_empty:
            empty = (
                np.empty(0, dtype=code_dtype_for_n(n))
                if w == 0
                else np.empty((0, w), dtype=np.uint64)
            )
            return cls(n, empty)
        if w == 0:
            merged = np.concatenate(non_empty)
            unique = np.unique(merged)
            return cls(n, unique)
        merged = np.vstack(non_empty)
        keys = _multiword_sort_key(merged)
        order = np.argsort(keys, kind="quicksort")
        skeys = keys[order]
        starts = np.empty(skeys.size, dtype=np.bool_)
        starts[0] = True
        if skeys.size > 1:
            starts[1:] = skeys[1:] != skeys[:-1]
        idx = np.flatnonzero(starts)
        return cls(n, np.ascontiguousarray(merged[order[idx]]))

    def __len__(self) -> int:
        return int(self.codes.shape[0])

    def nbytes(self) -> int:
        return int(self.codes.nbytes)

    def lookup_many(self, codes: np.ndarray) -> np.ndarray:
        """Map an array of codes to their ``int32`` indices in this table (``-1`` if absent)."""
        arr = _validate_code_array(self.n, codes)
        if arr.shape[0] == 0:
            return np.empty(0, dtype=np.int32)
        if len(self) == 0:
            return np.full(arr.shape[0], -1, dtype=np.int32)
        keys = arr if arr.ndim == 1 else _multiword_sort_key(arr)
        pos = np.searchsorted(self._sort_keys, keys)
        out = np.full(keys.size, -1, dtype=np.int32)
        in_bounds = pos < self._sort_keys.size
        valid = np.flatnonzero(in_bounds)
        if valid.size:
            hit = self._sort_keys[pos[valid]] == keys[valid]
            out[valid[hit]] = pos[valid[hit]].astype(np.int32)
        return out

    def lookup(self, code: int) -> int | None:
        validate_word_code(self.n, code)
        idx = int(self.lookup_many(int_codes_to_array(self.n, [code]))[0])
        return None if idx < 0 else idx

    def int_codes(self) -> list[int]:
        return array_to_int_codes(self.n, self.codes)

    def code(self, index: int) -> int:
        if self.codes.ndim == 1:
            return int(self.codes[index])
        return _single_int_code_from_row(self.codes[index])


@dataclass(frozen=True)
class GlobalSoABank:
    """CSR-packed, globally collated ``S_ij = A_i† A_j`` and ``H_ij = A_i† H A_j`` bank.

    All upper-triangle pairs ``0 <= i <= j < m`` are built directly in SoA and
    collated in one global pass into a single sorted :class:`SoAWordTable`.
    """

    n: int
    m: int
    labels: tuple[str, ...]
    word_table: SoAWordTable
    pair_keys: tuple[tuple[int, int], ...]
    s_offsets: np.ndarray  # shape (num_pairs + 1,), int64
    s_word_indices: np.ndarray  # shape (nnz_s,), int32
    s_coeffs: np.ndarray  # shape (nnz_s,), complex128
    h_offsets: np.ndarray  # shape (num_pairs + 1,), int64
    h_word_indices: np.ndarray  # shape (nnz_h,), int32
    h_coeffs: np.ndarray  # shape (nnz_h,), complex128

    @property
    def word_universe_size(self) -> int:
        return len(self.word_table)

    @property
    def total_coefficients(self) -> int:
        return int(self.s_coeffs.size + self.h_coeffs.size)

    def nbytes(self) -> int:
        return int(
            self.word_table.nbytes()
            + self.s_offsets.nbytes
            + self.s_word_indices.nbytes
            + self.s_coeffs.nbytes
            + self.h_offsets.nbytes
            + self.h_word_indices.nbytes
            + self.h_coeffs.nbytes
        )

    def bytes_per_coefficient(self) -> float:
        total = self.total_coefficients
        return (self.nbytes() / total) if total else 0.0

    def evaluate_from_word_expectations(
        self, expectations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble Hermitian ``(S, H)`` from one expectation value per word in ``word_table``."""
        exp = np.asarray(expectations, dtype=np.complex128)
        if exp.shape != (len(self.word_table),):
            raise ValueError(
                f"expected {len(self.word_table)} expectations, got shape {exp.shape}"
            )
        S = np.zeros((self.m, self.m), dtype=np.complex128)
        H = np.zeros((self.m, self.m), dtype=np.complex128)

        # Evaluate all S and H rows via two vectorized weighted reductions.
        s_prod = self.s_coeffs * exp[self.s_word_indices]
        h_prod = self.h_coeffs * exp[self.h_word_indices]

        for p_idx, (i, j) in enumerate(self.pair_keys):
            s0, s1 = int(self.s_offsets[p_idx]), int(self.s_offsets[p_idx + 1])
            h0, h1 = int(self.h_offsets[p_idx]), int(self.h_offsets[p_idx + 1])
            s_val = complex(np.sum(s_prod[s0:s1])) if s1 > s0 else 0j
            h_val = complex(np.sum(h_prod[h0:h1])) if h1 > h0 else 0j
            S[i, j] = s_val
            H[i, j] = h_val
            if i != j:
                S[j, i] = s_val.conjugate()
                H[j, i] = h_val.conjugate()
        return S, H

    def evaluate_density_soa(self, rho: PauliSoA) -> tuple[np.ndarray, np.ndarray]:
        """Exact ``(S, H)`` against a Pauli-basis density operator ``rho`` via sorted lookup."""
        if rho.n != self.n:
            raise ValueError(f"rho has n={rho.n}, bank has n={self.n}")
        scale = float(2 ** self.n)
        exp = np.zeros(len(self.word_table), dtype=np.complex128)
        if rho.nnz() > 0 and len(self.word_table) > 0:
            idx = self.word_table.lookup_many(rho.codes)
            hit = idx >= 0
            if bool(np.any(hit)):
                exp[idx[hit]] = scale * rho.coeffs[hit]
        return self.evaluate_from_word_expectations(exp)

    def evaluate_determinant(self, reference: Any) -> tuple[np.ndarray, np.ndarray]:
        """Exact ``(S, H)`` on a computational-basis determinant in ``O(W + nnz)`` time."""
        mask = _normalize_occupation_mask(self.n, reference)
        w_count = len(self.word_table)
        exp = np.zeros(w_count, dtype=np.complex128)
        if w_count > 0:
            codes = self.word_table.codes
            n = self.n
            if codes.ndim == 1:
                dtype = codes.dtype
                lo = lane_mask_for_width(n, dtype)
                z = (codes >> dtype.type(1)) & lo
                x = (codes & lo) ^ z
                diag = x == 0
                if bool(np.any(diag)):
                    occ_even = 0
                    for j in range(n):
                        if (mask >> j) & 1:
                            occ_even |= 1 << (2 * j)
                    parity = popcount_array(z[diag] & dtype.type(occ_even)) & 1
                    exp[diag] = np.where(parity == 0, 1.0, -1.0)
            else:
                masks = multiword_lane_masks(n)
                z = (codes >> np.uint64(1)) & masks[None, :]
                x = (codes & masks[None, :]) ^ z
                diag = np.all(x == 0, axis=1)
                if bool(np.any(diag)):
                    w = codes.shape[1]
                    occ_limbs = np.zeros(w, dtype=np.uint64)
                    for j in range(n):
                        if (mask >> j) & 1:
                            limb, off = divmod(j, 32)
                            occ_limbs[limb] |= np.uint64(1 << (2 * off))
                    xor_acc = _xor_reduce_last_axis(z[diag] & occ_limbs[None, :])
                    parity = popcount_array(xor_acc) & 1
                    exp[diag] = np.where(parity == 0, 1.0, -1.0)
        return self.evaluate_from_word_expectations(exp)


def _as_soa_generator(item: Any) -> tuple[str, PauliSoA]:
    if isinstance(item, PauliSoA):
        return ("", item)
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        label, op = item
        return (label, op if isinstance(op, PauliSoA) else PauliSoA.from_mv(op))
    if hasattr(item, "mv"):
        return (str(getattr(item, "label", "")), PauliSoA.from_mv(item.mv))
    if hasattr(item, "terms") and hasattr(item, "n"):
        return ("", PauliSoA.from_mv(item))
    raise TypeError(f"unsupported generator type: {type(item)!r}")


def build_soa_bank(
    hamiltonian: Any,
    generators: Iterable[Any],
    *,
    tol: float = TOL,
    accumulate: str = "auto",
    backend: str = "numpy",
    chunk_pairs: int | None = None,
) -> GlobalSoABank:
    """Build and globally collate all upper-triangle ``S_ij`` and ``H_ij`` operators in SoA.

    Precomputes ``H A_j`` once per generator (``m`` products instead of
    ``m(m+1)/2``), computes ``A_i† A_j`` and ``A_i† (H A_j)`` via
    :func:`soa_mul`, and performs one global collate into :class:`SoAWordTable`.
    """
    h_soa = hamiltonian if isinstance(hamiltonian, PauliSoA) else PauliSoA.from_mv(hamiltonian)
    n = h_soa.n
    parsed = [_as_soa_generator(g) for g in generators]
    m = len(parsed)
    labels = tuple(lbl or f"g{i}" for i, (lbl, _) in enumerate(parsed))
    ops = [op for _, op in parsed]
    for op in ops:
        if op.n != n:
            raise ValueError(f"generator n={op.n} disagrees with hamiltonian n={n}")
    daggers = [op.dagger() for op in ops]
    h_ops = [
        soa_mul(h_soa, op, tol=tol, accumulate=accumulate, backend=backend, chunk_pairs=chunk_pairs)
        for op in ops
    ]

    pair_keys: list[tuple[int, int]] = []
    s_ops: list[PauliSoA] = []
    h_pair_ops: list[PauliSoA] = []
    code_arrays: list[np.ndarray] = []

    for i in range(m):
        ai_dag = daggers[i]
        for j in range(i, m):
            pair_keys.append((i, j))
            s_ij = soa_mul(
                ai_dag,
                ops[j],
                tol=tol,
                accumulate=accumulate,
                backend=backend,
                chunk_pairs=chunk_pairs,
            )
            h_ij = soa_mul(
                ai_dag,
                h_ops[j],
                tol=tol,
                accumulate=accumulate,
                backend=backend,
                chunk_pairs=chunk_pairs,
            )
            s_ops.append(s_ij)
            h_pair_ops.append(h_ij)
            if s_ij.nnz():
                code_arrays.append(s_ij.codes)
            if h_ij.nnz():
                code_arrays.append(h_ij.codes)

    word_table = SoAWordTable.from_code_arrays(n, code_arrays)

    num_pairs = len(pair_keys)
    s_offsets = np.zeros(num_pairs + 1, dtype=np.int64)
    h_offsets = np.zeros(num_pairs + 1, dtype=np.int64)
    for k in range(num_pairs):
        s_offsets[k + 1] = s_offsets[k] + s_ops[k].nnz()
        h_offsets[k + 1] = h_offsets[k] + h_pair_ops[k].nnz()

    nnz_s = int(s_offsets[-1])
    nnz_h = int(h_offsets[-1])
    s_word_indices = np.empty(nnz_s, dtype=np.int32)
    s_coeffs = np.empty(nnz_s, dtype=np.complex128)
    h_word_indices = np.empty(nnz_h, dtype=np.int32)
    h_coeffs = np.empty(nnz_h, dtype=np.complex128)

    # Single vectorized lookup per concatenated array rather than per-pair Python loops.
    if nnz_s > 0:
        w = word_width_for_n(n)
        s_all_codes = (
            np.concatenate([op.codes for op in s_ops if op.nnz()])
            if w == 0
            else np.vstack([op.codes for op in s_ops if op.nnz()])
        )
        s_word_indices[:] = word_table.lookup_many(s_all_codes)
        s_coeffs[:] = np.concatenate([op.coeffs for op in s_ops if op.nnz()])
    if nnz_h > 0:
        w = word_width_for_n(n)
        h_all_codes = (
            np.concatenate([op.codes for op in h_pair_ops if op.nnz()])
            if w == 0
            else np.vstack([op.codes for op in h_pair_ops if op.nnz()])
        )
        h_word_indices[:] = word_table.lookup_many(h_all_codes)
        h_coeffs[:] = np.concatenate([op.coeffs for op in h_pair_ops if op.nnz()])

    return GlobalSoABank(
        n=n,
        m=m,
        labels=labels,
        word_table=word_table,
        pair_keys=tuple(pair_keys),
        s_offsets=s_offsets,
        s_word_indices=s_word_indices,
        s_coeffs=s_coeffs,
        h_offsets=h_offsets,
        h_word_indices=h_word_indices,
        h_coeffs=h_coeffs,
    )


__all__ = [
    "DEFAULT_CPU_CHUNK_PAIRS",
    "DEFAULT_GPU_CHUNK_PAIRS",
    "GlobalSoABank",
    "PauliSoA",
    "SoAWordTable",
    "TOL",
    "array_to_int_codes",
    "available_gpu_backends",
    "build_soa_bank",
    "code_dtype_for_n",
    "int_codes_to_array",
    "lane_mask_for_width",
    "multiword_lane_masks",
    "popcount_array",
    "soa_add",
    "soa_anticommute_mask",
    "soa_commutator",
    "soa_commutator_terms",
    "soa_determinant_expectation",
    "soa_hs_product",
    "soa_mul",
    "soa_mul_terms",
    "soa_trace_pairing",
    "synchronize_backend",
    "word_width_for_n",
]

"""Strict, dependency-light FCIDUMP ingestion.

FCIDUMP is the smallest widely used interchange boundary for an electronic
active-space Hamiltonian: a restricted spatial-orbital one-electron matrix,
two-electron repulsion integrals in chemist notation, the nuclear/frozen-core
constant, and the target ``(N, M_S)`` sector.  This module parses that boundary
with only NumPy and maps it through :mod:`clifford_qc`'s own Jordan--Wigner
operators.  It does not run SCF, orbital localization, active-space selection,
or embedding.

The supported contract is deliberately narrow and explicit:

* real, restricted FCIDUMP records (``IUHF=0``);
* one-based spatial-orbital indices with the usual zero sentinels;
* packed one- and two-electron integrals restored by their exact permutation
  symmetries;
* interleaved spin orbitals, ``2*p=alpha`` and ``2*p+1=beta``;
* a reference determinant whose occupations agree with ``NELEC`` and ``MS2``.

Unrestricted and complex extensions are rejected rather than silently read
under the wrong convention.  The resulting :class:`~clifford_qc.models.Model`
is the same object used by A-CASE, the sector backend, and projected
observables.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..fermion import c_op, cdag_op
from ..ir import PauliSum, Program
from ..multivector import MV
from .spin import Model

_HEADER_START = re.compile(r"^\s*&(FCI|FCIDUMP)\b", re.IGNORECASE)
_HEADER_END = re.compile(r"&END\b|/", re.IGNORECASE)
_KEY = r"[A-Za-z_][A-Za-z0-9_]*"


@dataclass(frozen=True)
class FCIDump:
    """Parsed restricted FCIDUMP integrals in chemist notation.

    ``two_body[p,q,r,s]`` is ``(pq|rs)``.  Arrays are full, not packed; the
    parser restores the usual eightfold real-orbital symmetry.
    """

    n_orbitals: int
    n_electrons: int
    ms2: int
    one_body: np.ndarray
    two_body: np.ndarray
    core_energy: float
    orbsym: tuple[int, ...]
    isym: int
    source_path: str
    source_sha256: str

    @property
    def sz(self) -> float:
        return 0.5 * self.ms2


def _without_comment(line: str) -> str:
    """Remove FCIDUMP comments while leaving Fortran exponents untouched."""
    for marker in ("!", "#"):
        line = line.split(marker, 1)[0]
    return line.strip()


def _header_value(header: str, key: str, *, required: bool = False,
                  default: int | None = None) -> int:
    match = re.search(
        rf"\b{re.escape(key)}\s*=\s*([^,\s/&]+)", header, re.IGNORECASE)
    if match is None:
        if required:
            raise ValueError(f"FCIDUMP header is missing {key}")
        if default is None:
            raise ValueError(f"FCIDUMP header is missing {key}")
        return default
    try:
        return int(match.group(1))
    except ValueError as error:
        raise ValueError(f"FCIDUMP header {key} must be an integer") from error


def _header_list(header: str, key: str) -> tuple[int, ...]:
    match = re.search(
        rf"\b{re.escape(key)}\s*=\s*(.*?)(?=\b{_KEY}\s*=|&END\b|/|$)",
        header, re.IGNORECASE)
    if match is None:
        return ()
    fields = [part for part in re.split(r"[\s,]+", match.group(1).strip())
              if part]
    try:
        return tuple(int(field) for field in fields)
    except ValueError as error:
        raise ValueError(f"FCIDUMP header {key} must contain integers") from error


def _finite_fortran_float(token: str, *, line_number: int) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(
            f"FCIDUMP line {line_number} has an invalid coefficient") from error
    if not math.isfinite(value):
        raise ValueError(f"FCIDUMP line {line_number} coefficient must be finite")
    return value


def _assign(array: np.ndarray, seen: np.ndarray, index: tuple[int, ...],
            value: float, *, label: str, line_number: int,
            tolerance: float) -> None:
    if seen[index] and abs(float(array[index]) - value) > tolerance:
        raise ValueError(
            f"FCIDUMP line {line_number} conflicts with an earlier {label} "
            f"integral at {tuple(i + 1 for i in index)}")
    array[index] = value
    seen[index] = True


def _two_body_symmetry(i: int, j: int, k: int, l: int
                       ) -> set[tuple[int, int, int, int]]:
    """Eightfold symmetry of real chemist integrals ``(ij|kl)``."""
    return {
        (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
        (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i),
    }


def read_fcidump(path: str | Path, *, duplicate_tolerance: float = 1e-12
                 ) -> FCIDump:
    """Parse a real restricted FCIDUMP file and restore its packed integrals."""
    if duplicate_tolerance < 0.0 or not math.isfinite(duplicate_tolerance):
        raise ValueError("duplicate_tolerance must be non-negative and finite")
    source = Path(path)
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("FCIDUMP must be UTF-8 text") from error
    lines = text.splitlines()

    header_lines: list[str] = []
    body_start: int | None = None
    started = False
    for index, original in enumerate(lines):
        line = _without_comment(original)
        if not line:
            continue
        if not started:
            if _HEADER_START.match(line) is None:
                raise ValueError("FCIDUMP must start with &FCI or &FCIDUMP")
            started = True
        header_lines.append(line)
        if _HEADER_END.search(line):
            body_start = index + 1
            break
    if not started or body_start is None:
        raise ValueError("FCIDUMP header is missing &END or /")
    header = " ".join(header_lines)

    n_orbitals = _header_value(header, "NORB", required=True)
    n_electrons = _header_value(header, "NELEC", required=True)
    ms2 = _header_value(header, "MS2", default=0)
    iuhf = _header_value(header, "IUHF", default=0)
    isym = _header_value(header, "ISYM", default=1)
    orbsym = _header_list(header, "ORBSYM")
    if n_orbitals <= 0:
        raise ValueError("FCIDUMP NORB must be positive")
    if not 0 <= n_electrons <= 2 * n_orbitals:
        raise ValueError("FCIDUMP NELEC must be in [0, 2*NORB]")
    if abs(ms2) > n_electrons or (n_electrons + ms2) % 2:
        raise ValueError("FCIDUMP NELEC and MS2 do not define an integer spin sector")
    if iuhf != 0:
        raise NotImplementedError(
            "unrestricted FCIDUMP (IUHF=1) is not supported; convert to a "
            "restricted spatial-orbital record first")
    if orbsym and len(orbsym) != n_orbitals:
        raise ValueError("FCIDUMP ORBSYM must contain exactly NORB entries")
    if not orbsym:
        orbsym = (1,) * n_orbitals

    one_body = np.zeros((n_orbitals, n_orbitals), dtype=float)
    two_body = np.zeros(
        (n_orbitals, n_orbitals, n_orbitals, n_orbitals), dtype=float)
    one_seen = np.zeros_like(one_body, dtype=bool)
    two_seen = np.zeros_like(two_body, dtype=bool)
    core_energy = 0.0
    core_seen = False

    for offset, original in enumerate(lines[body_start:], start=body_start + 1):
        line = _without_comment(original)
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"FCIDUMP line {offset} must contain coefficient and four indices")
        value = _finite_fortran_float(fields[0], line_number=offset)
        try:
            indices = tuple(int(field) for field in fields[1:])
        except ValueError as error:
            raise ValueError(
                f"FCIDUMP line {offset} indices must be integers") from error
        if any(index < 0 or index > n_orbitals for index in indices):
            raise ValueError(
                f"FCIDUMP line {offset} indices must be in [0, NORB]")
        i, j, k, l = indices
        if indices == (0, 0, 0, 0):
            if core_seen and abs(core_energy - value) > duplicate_tolerance:
                raise ValueError(
                    f"FCIDUMP line {offset} conflicts with the core energy")
            core_energy = value
            core_seen = True
        elif i > 0 and j > 0 and k == 0 and l == 0:
            for index in {(i - 1, j - 1), (j - 1, i - 1)}:
                _assign(one_body, one_seen, index, value, label="one-body",
                        line_number=offset, tolerance=duplicate_tolerance)
        elif all(index > 0 for index in indices):
            for index in _two_body_symmetry(i - 1, j - 1, k - 1, l - 1):
                _assign(two_body, two_seen, index, value, label="two-body",
                        line_number=offset, tolerance=duplicate_tolerance)
        else:
            raise ValueError(
                f"FCIDUMP line {offset} uses an invalid zero-index sentinel pattern")

    return FCIDump(
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
        ms2=ms2,
        one_body=one_body,
        two_body=two_body,
        core_energy=float(core_energy),
        orbsym=orbsym,
        isym=isym,
        source_path=str(source),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _reference_occupations(n_orbitals: int, n_electrons: int,
                           ms2: int) -> tuple[int, ...]:
    if not 0 <= n_electrons <= 2 * n_orbitals:
        raise ValueError("n_electrons must be in [0, 2*NORB]")
    if abs(ms2) > n_electrons or (n_electrons + ms2) % 2:
        raise ValueError("n_electrons and ms2 do not define an integer spin sector")
    n_alpha = (n_electrons + ms2) // 2
    n_beta = n_electrons - n_alpha
    if n_alpha > n_orbitals or n_beta > n_orbitals:
        raise ValueError("requested spin sector does not fit in NORB orbitals")
    return tuple([2 * p for p in range(n_alpha)]
                 + [2 * p + 1 for p in range(n_beta)])


def _reference_program(n_qubits: int, occupied: Iterable[int]) -> Program:
    program = Program(n_qubits)
    for index in sorted(occupied):
        program.clifford("X", index)
    return program


def model_from_fcidump(data: FCIDump, *, name: str | None = None,
                       n_electrons: int | None = None, ms2: int | None = None,
                       integral_tolerance: float = 1e-12) -> Model:
    """Map parsed chemist integrals to a Jordan--Wigner qubit ``Model``.

    The two-electron term is assembled directly from

    ``1/2 sum_(pqrs) sum_(sigma,tau) (pq|rs)
      a†_(p,sigma) a†_(r,tau) a_(s,tau) a_(q,sigma)``

    with ``p q r s`` spatial orbitals and ``sigma tau`` spins -- spelled out
    because this docstring *is* the human-checkable statement of the convention.
    An earlier version wrote both an orbital index and a spin label as ``s``,
    which reads as two different operators depending on which one you take
    ``a_(s,t)`` to mean; the code was right and the formula was not. Getting
    this exact matters here more than usual: a chemist-to-physicist reindexing
    error is invisible in the assembled Hamiltonian and shows up only as a wrong
    correlation energy.

    Assembling in chemist notation avoids that transpose entirely rather than
    performing it correctly, and gives the optional OpenFermion/PySCF
    comparison tests an independent implementation to check.
    """
    if integral_tolerance < 0.0 or not math.isfinite(integral_tolerance):
        raise ValueError("integral_tolerance must be non-negative and finite")
    if (n_electrons is not None
            and (not isinstance(n_electrons, int) or isinstance(n_electrons, bool))):
        raise TypeError("n_electrons override must be an integer")
    if ms2 is not None and (not isinstance(ms2, int) or isinstance(ms2, bool)):
        raise TypeError("ms2 override must be an integer")
    electrons = data.n_electrons if n_electrons is None else n_electrons
    spin_twice = data.ms2 if ms2 is None else ms2
    occupied = _reference_occupations(data.n_orbitals, electrons, spin_twice)
    n_qubits = 2 * data.n_orbitals

    total = MV(n_qubits, {0: complex(data.core_energy)})
    for p in range(data.n_orbitals):
        for q in range(data.n_orbitals):
            coefficient = float(data.one_body[p, q])
            if abs(coefficient) <= integral_tolerance:
                continue
            for spin in (0, 1):
                left, right = 2 * p + spin, 2 * q + spin
                total = total + coefficient * (
                    cdag_op(n_qubits, left) * c_op(n_qubits, right))

    for p in range(data.n_orbitals):
        for q in range(data.n_orbitals):
            for r in range(data.n_orbitals):
                for s in range(data.n_orbitals):
                    coefficient = 0.5 * float(data.two_body[p, q, r, s])
                    if abs(coefficient) <= integral_tolerance:
                        continue
                    for spin in (0, 1):
                        for other_spin in (0, 1):
                            total = total + coefficient * (
                                cdag_op(n_qubits, 2 * p + spin)
                                * cdag_op(n_qubits, 2 * r + other_spin)
                                * c_op(n_qubits, 2 * s + other_spin)
                                * c_op(n_qubits, 2 * q + spin))

    if not total.is_hermitian(1e-10):
        raise ValueError("FCIDUMP assembled a non-Hermitian Hamiltonian")
    terms: dict[int, complex] = {}
    for code, coefficient in total.terms.items():
        if abs(coefficient.imag) > 1e-10:
            raise ValueError("real restricted FCIDUMP produced a complex Pauli coefficient")
        if abs(coefficient.real) > integral_tolerance:
            terms[code] = complex(float(coefficient.real))
    hamiltonian = PauliSum(n_qubits, terms)

    label = name or f"fcidump({Path(data.source_path).name})"
    if not isinstance(label, str) or not label.strip():
        raise ValueError("name must be a non-empty string")
    metadata = {
        "kind": "molecular",
        "source": "fcidump",
        "source_kind": "fcidump",
        "path": data.source_path,
        "source_sha256": data.source_sha256,
        "n_spatial_orbitals": data.n_orbitals,
        "spin_orbitals": n_qubits,
        "spin_convention": "interleaved",
        "n_electrons": electrons,
        "ms2": spin_twice,
        "sz": 0.5 * spin_twice,
        "core_energy": data.core_energy,
        "orbsym": list(data.orbsym),
        "isym": data.isym,
        "energy_unit": "hartree",
        "reference_occupied_spin_orbitals": list(sorted(occupied)),
        "integral_tolerance": integral_tolerance,
    }
    return Model(
        name=label.strip(),
        n=n_qubits,
        hamiltonian=hamiltonian,
        reference=_reference_program(n_qubits, occupied),
        hva_layers=(),
        metadata=metadata,
    )


def fcidump_model(path: str | Path, *, name: str | None = None,
                  n_electrons: int | None = None, ms2: int | None = None,
                  integral_tolerance: float = 1e-12) -> Model:
    """Load a restricted FCIDUMP into the common correlated ``Model`` API."""
    return model_from_fcidump(
        read_fcidump(path),
        name=name,
        n_electrons=n_electrons,
        ms2=ms2,
        integral_tolerance=integral_tolerance,
    )


__all__ = [
    "FCIDump",
    "fcidump_model",
    "model_from_fcidump",
    "read_fcidump",
]

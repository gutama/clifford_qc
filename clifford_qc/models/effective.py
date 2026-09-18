r"""Versioned ingestion for small downfolded correlated Hamiltonians.

This module is the narrow interface between an upstream DFT/Wannier/embedding
workflow and :mod:`clifford_qc`.  It deliberately does not parse a particular
electronic-structure package.  Instead it accepts the smallest useful common
denominator for a spin-independent, single-orbital Wannier model,

.. math::

    H = \sum_{ij,\sigma} h_{ij} c^\dagger_{i\sigma} c_{j\sigma}
        + \sum_i U_i n_{i\uparrow}n_{i\downarrow}
        - \mu \sum_{i,\sigma} n_{i\sigma}.

The one-body matrix may contain onsite energies, long-range hopping, and
complex Hermitian phases.  The interaction is the local term supplied by an
embedding or downfolding step.  The output is the same ``Model`` used by the
built-in lattice systems, so A-CASE, exact sector references, and projected
material observables require no special downstream path.

Input is versioned because orbital order, spin order, and energy units are part
of the scientific result, not incidental JSON details.  Complex numbers are
encoded as ``[real, imag]``; ordinary JSON numbers remain the common real case.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..fermion import c_op, cdag_op
from ..ir import PauliSum, Program
from ..multivector import MV
from .lattice import SPIN_DOWN, SPIN_UP, spin_orbital
from .metadata import FERMIONIC_LATTICE, model_metadata
from .spin import Model

EFFECTIVE_HAMILTONIAN_SCHEMA = "clifford_qc.effective_hamiltonian.v1"


def _finite_complex(value: Any, *, field: str) -> complex:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a number, not bool")
    if isinstance(value, (int, float)):
        out = complex(float(value))
    elif (isinstance(value, Sequence) and not isinstance(value, (str, bytes))
          and len(value) == 2):
        out = complex(float(value[0]), float(value[1]))
    else:
        raise TypeError(f"{field} must be a real number or [real, imag]")
    if not (math.isfinite(out.real) and math.isfinite(out.imag)):
        raise ValueError(f"{field} must be finite")
    return out


def _finite_real(value: Any, *, field: str) -> float:
    out = _finite_complex(value, field=field)
    if abs(out.imag) > 1e-14:
        raise ValueError(f"{field} must be real")
    return float(out.real)


def _one_body_matrix(payload: Mapping[str, Any]) -> np.ndarray:
    raw = payload.get("one_body")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("one_body must be a non-empty square matrix")
    sites = len(raw)
    matrix = np.zeros((sites, sites), dtype=complex)
    for i, row in enumerate(raw):
        if (not isinstance(row, Sequence) or isinstance(row, (str, bytes))
                or len(row) != sites):
            raise ValueError("one_body must be a non-empty square matrix")
        for j, value in enumerate(row):
            matrix[i, j] = _finite_complex(value, field=f"one_body[{i}][{j}]")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0.0):
        error = float(np.max(np.abs(matrix - matrix.conj().T)))
        raise ValueError(f"one_body must be Hermitian (max mismatch {error:.3e})")
    return matrix


def _onsite_interactions(payload: Mapping[str, Any], sites: int) -> tuple[float, ...]:
    raw = payload.get("onsite_u", 0.0)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != sites:
            raise ValueError(f"onsite_u must have {sites} entries")
        return tuple(_finite_real(value, field=f"onsite_u[{i}]")
                     for i, value in enumerate(raw))
    value = _finite_real(raw, field="onsite_u")
    return (value,) * sites


def _reference(payload: Mapping[str, Any], n_qubits: int) -> tuple[Program, tuple[int, ...]]:
    raw = payload.get("reference_occupied_spin_orbitals")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError(
            "reference_occupied_spin_orbitals must name a non-empty determinant")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in raw):
        raise TypeError("reference spin-orbital indices must be integers")
    occupied = tuple(raw)
    if len(set(occupied)) != len(occupied):
        raise ValueError("reference spin-orbital indices must be unique")
    if any(index < 0 or index >= n_qubits for index in occupied):
        raise ValueError(f"reference spin-orbital indices must be in [0, {n_qubits})")
    program = Program(n_qubits)
    for index in sorted(occupied):
        program.clifford("X", index)
    return program, tuple(sorted(occupied))


def _serializable_complex_matrix(matrix: np.ndarray) -> list[list[float | list[float]]]:
    out: list[list[float | list[float]]] = []
    for row in matrix:
        encoded = []
        for value in row:
            if abs(value.imag) <= 1e-14:
                encoded.append(float(value.real))
            else:
                encoded.append([float(value.real), float(value.imag)])
        out.append(encoded)
    return out


def effective_hamiltonian(payload: Mapping[str, Any]) -> Model:
    """Build a correlated ``Model`` from a versioned downfolded-Hamiltonian record.

    ``reference_occupied_spin_orbitals`` uses the package's interleaved order:
    ``2*site`` is spin up and ``2*site+1`` is spin down.  An optional ``sector``
    mapping is checked against that determinant; this catches the most damaging
    interchange error before any eigensolver runs.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("effective-Hamiltonian input must be a mapping")
    schema = payload.get("schema")
    if schema != EFFECTIVE_HAMILTONIAN_SCHEMA:
        raise ValueError(
            f"schema must be {EFFECTIVE_HAMILTONIAN_SCHEMA!r}, got {schema!r}")
    matrix = _one_body_matrix(payload)
    sites = matrix.shape[0]
    n_qubits = 2 * sites
    onsite_u = _onsite_interactions(payload, sites)
    chemical_potential = _finite_real(
        payload.get("chemical_potential", 0.0), field="chemical_potential")
    reference, occupied = _reference(payload, n_qubits)

    n_electrons = len(occupied)
    sz = float(sum(0.5 if index % 2 == SPIN_UP else -0.5
                   for index in occupied))
    sector = payload.get("sector", {})
    if sector is not None:
        if not isinstance(sector, Mapping):
            raise TypeError("sector must be a mapping")
        if "n_electrons" in sector:
            declared_n = _finite_real(
                sector["n_electrons"], field="sector.n_electrons")
            if not declared_n.is_integer():
                raise ValueError("sector.n_electrons must be an integer")
            if int(declared_n) != n_electrons:
                raise ValueError("sector.n_electrons disagrees with the reference")
        if ("sz" in sector
                and abs(_finite_real(sector["sz"], field="sector.sz") - sz) > 1e-12):
            raise ValueError("sector.sz disagrees with the reference")

    total = MV(n_qubits)
    for i in range(sites):
        for j in range(sites):
            coefficient = complex(matrix[i, j])
            if abs(coefficient) <= 1e-14:
                continue
            for spin in (SPIN_UP, SPIN_DOWN):
                p = spin_orbital(i, spin)
                q = spin_orbital(j, spin)
                total = total + coefficient * (
                    cdag_op(n_qubits, p) * c_op(n_qubits, q))
    for site, interaction in enumerate(onsite_u):
        up = spin_orbital(site, SPIN_UP)
        down = spin_orbital(site, SPIN_DOWN)
        total = total + interaction * (
            cdag_op(n_qubits, up) * c_op(n_qubits, up)
            * cdag_op(n_qubits, down) * c_op(n_qubits, down))
    if chemical_potential:
        for orbital in range(n_qubits):
            total = total - chemical_potential * (
                cdag_op(n_qubits, orbital) * c_op(n_qubits, orbital))

    if not total.is_hermitian(1e-11):
        raise ValueError("assembled effective Hamiltonian is not Hermitian")
    terms = {code: complex(coefficient.real)
             for code, coefficient in total.terms.items()
             if abs(coefficient.real) > 1e-14}
    hamiltonian = PauliSum(n_qubits, terms)

    bonds = []
    for i in range(sites):
        for j in range(i + 1, sites):
            if abs(matrix[i, j]) > 1e-14:
                bonds.append((i, j))
    name = payload.get("name", "effective-wannier-model")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    energy_unit = payload.get("energy_unit", "eV")
    if not isinstance(energy_unit, str) or not energy_unit.strip():
        raise ValueError("energy_unit must be a non-empty string")
    source = payload.get("source", {})
    if not isinstance(source, Mapping):
        raise TypeError("source must be a mapping")

    # ``schema`` here is the schema of the *input payload*; the metadata
    # contract's own version travels as ``metadata_schema``.
    metadata = model_metadata(
        FERMIONIC_LATTICE,
        spin_orbitals=n_qubits,
        n_spatial_orbitals=sites,
        n_electrons=n_electrons,
        sz=sz,
        spin_convention="interleaved",
        source_kind="effective_hamiltonian",
        schema=EFFECTIVE_HAMILTONIAN_SCHEMA,
        sites=sites,
        rows=1,
        cols=sites,
        n_orbitals=1,
        bonds=bonds,
        one_body=_serializable_complex_matrix(matrix),
        onsite_u=list(onsite_u),
        chemical_potential=chemical_potential,
        energy_unit=energy_unit,
        source=dict(source),
        reference_occupied_spin_orbitals=list(occupied),
    )
    return Model(name=name.strip(), n=n_qubits, hamiltonian=hamiltonian,
                 reference=reference, hva_layers=(), metadata=metadata)


def load_effective_hamiltonian(path: str | Path) -> Model:
    """Load :func:`effective_hamiltonian` from a UTF-8 JSON file."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} is not valid JSON: {error}") from error
    return effective_hamiltonian(payload)


__all__ = [
    "EFFECTIVE_HAMILTONIAN_SCHEMA",
    "effective_hamiltonian",
    "load_effective_hamiltonian",
]

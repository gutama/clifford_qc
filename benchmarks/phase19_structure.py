"""Outcome-free inputs, circuit manifests and allocation for Phase 19.

This module builds Hamiltonians and circuits only. It never prepares a state,
evaluates a mean/variance, diagonalizes a Hamiltonian, prices a device schedule,
or draws outcomes. The sampled comparison belongs to a later implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from clifford_qc.measurement.cliques import compile_clique_measurement_plan, _setting_resources
from clifford_qc.measurement.grouping import qwc_groups, shared_basis
from clifford_qc.models.fcidump import fcidump_model

ROOT = Path(__file__).resolve().parent.parent
BANKS = (
    ("h4", "h4_sto3g_r0.9"),
    ("lih", "lih_sto3g_r1.5949_cas4e4o"),
    ("beh2", "beh2_sto3g_r1.3264"),
)
CARDS = ("ion-like", "logical-alltoall", "superconducting-like")
PARENT_MERGE = "45966af20ca99f99e70daa0e075f456e48e1f71e"
CONFIG_PATH = "benchmarks/configs/phase19_energy_comparison.json"
RECORD_PATH = "benchmarks/reference_results/phase19_energy_comparison.json"
INPUTS = tuple(
    f"benchmarks/data/{stem}{suffix}"
    for _, stem in BANKS for suffix in (".FCIDUMP", ".provenance.json")
) + tuple(f"benchmarks/configs/device_cards/{name}.json" for name in CARDS) + (
    "benchmarks/reference_results/protocol_cost.json",
)
IMPLEMENTATIONS = (
    "clifford_qc/models/fcidump.py", "clifford_qc/fermion.py",
    "clifford_qc/multivector.py", "clifford_qc/pauli_kernel.py", "clifford_qc/ir.py",
    "clifford_qc/qasm3.py", "clifford_qc/measurement/cliques.py",
    "clifford_qc/measurement/bank.py", "clifford_qc/measurement/grouping.py",
    "clifford_qc/measurement/confidence.py", "clifford_qc/measurement/cost.py",
    "benchmarks/phase19_structure.py",
)
FLOAT_TOLERANCE = {"relative": 1e-13, "absolute": 1e-14}


def canonical_digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_binding(relative: str, root: Path = ROOT) -> dict:
    data = (root / relative).read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data,
                                     usedforsecurity=False).hexdigest(),
    }


def _gate_sequence_digest(ops) -> str:
    # Rotor angles are frozen explicitly in the manifest. Hash the gate/qubit
    # topology separately, avoiding a libm last bit in the topology hash.
    topology = [[op[0], *op[2:]] if op[0] == "rz" else list(op) for op in ops]
    return canonical_digest(topology)


def build_bank(bank_id: str, root: Path = ROOT) -> dict:
    """Rebuild the fixed functional, partitions, readouts and logical resources.

    QWC receives code-sorted words, so its stable largest-degree tie break is
    ascending code. Member order within each returned group is also canonical.
    Both protocols keep every nonzero word of the same assembled Hamiltonian.
    """
    stems = dict(BANKS)
    stem = stems[bank_id]
    model = fcidump_model(root / f"benchmarks/data/{stem}.FCIDUMP",
                          name=bank_id, integral_tolerance=1e-12)
    H = model.hamiltonian
    coefficients = {word.code: float(c.real) for word, c in H.items()}
    words = tuple(word for word, c in H.items() if word.code != 0 and c != 0)
    functional = [[code, coefficients[code].hex()] for code in sorted(coefficients)]
    clique = compile_clique_measurement_plan(H)
    clique_rows = []
    for setting in clique.settings:
        clique_rows.append({
            "word_codes": [word.code for word in setting.words],
            "max_abs_value": abs(setting.weight),
            "rotations": [{"word_code": op.word.code, "angle": float(op.angle)}
                          for op in setting.rotations],
            "readout": {"kind": "weighted_parity", "weight": setting.weight,
                        "qubits": list(setting.readout_qubits)},
            "gate_sequence_sha256": _gate_sequence_digest(setting.ops),
            "resources": asdict(setting.resources),
        })
    qwc_rows = []
    for group in qwc_groups(words):
        group = sorted(group, key=lambda word: word.code)
        ops = []
        for q, letter in sorted(shared_basis(group).items()):
            if letter == "Y":
                ops.append(("sdg", q))
            if letter in {"X", "Y"}:
                ops.append(("h", q))
        qwc_rows.append({
            "word_codes": [word.code for word in group],
            "max_abs_value": math.fsum(abs(coefficients[word.code]) for word in group),
            "rotations": [],
            "readout": {"kind": "weighted_sum_of_parities",
                        "word_qubits": [list(word.support()) for word in group]},
            "gate_sequence_sha256": _gate_sequence_digest(ops),
            "resources": asdict(_setting_resources(tuple(ops))),
        })
    return {
        "schema": "clifford_qc.phase19_structural_bank.v1",
        "id": bank_id, "fcidump": f"benchmarks/data/{stem}.FCIDUMP",
        "n_qubits": model.n, "n_electrons": model.metadata["n_electrons"],
        "ms2": model.metadata["ms2"], "mapping": "jw", "integral_tolerance": 1e-12,
        "reference_occupied_spin_orbitals": model.metadata["reference_occupied_spin_orbitals"],
        "coefficients_hex": functional, "functional_sha256": canonical_digest(functional),
        "nonidentity_terms": len(words),
        "arms": [{"name": "qwc", "settings": qwc_rows},
                 {"name": "clique", "settings": clique_rows}],
    }


def coefficient_range_schedule(scores: Sequence[float], total_shots: int) -> tuple[int, ...]:
    """Two-shot floor, then exact binary-rational largest-remainder allocation.

    Scores are the frozen setting bounds L_g; no sampled or oracle variance is
    consulted. Equal fractional remainders break on the frozen setting index.
    """
    if isinstance(total_shots, bool) or not isinstance(total_shots, int):
        raise TypeError("total_shots must be an integer")
    weights = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("scores must be real numbers")
        value = float(score)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("scores must be positive and finite")
        weights.append(Fraction.from_float(value))
    if not weights or total_shots < 2 * len(weights):
        raise ValueError("at least two shots per nonempty setting list are required")
    remaining = total_shots - 2 * len(weights)
    total_weight = sum(weights)
    quotas = [remaining * weight / total_weight for weight in weights]
    additions = [quota.numerator // quota.denominator for quota in quotas]
    order = sorted(range(len(weights)), key=lambda i: (-(quotas[i] - additions[i]), i))
    for i in order[:remaining - sum(additions)]:
        additions[i] += 1
    return tuple(2 + value for value in additions)

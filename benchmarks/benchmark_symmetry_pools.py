"""Bounded exact structural audit of symmetry-adapted molecular pools.

Compare the complete N/Sz-conserving determinant singles/doubles pool with
spin-free singles/doubles. Count both the initial I-versus-each-candidate 2x2
screen and the complete candidate bank, not just a selected basis. This is
exploratory, classical determinant-reference evaluation: no sampled accuracy,
shot saving, hardware cost or general timing improvement is established.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMED_SOURCES = (
    "benchmarks/benchmark_symmetry_pools.py", "clifford_qc/subspace/symmetry_pools.py",
    "clifford_qc/subspace/fermionic_generators.py", "clifford_qc/subspace/projection.py",
    "clifford_qc/subspace/linalg.py", "clifford_qc/reference_states.py", "clifford_qc/fermion.py",
    "clifford_qc/multivector.py", "clifford_qc/pauli_kernel.py",
    "clifford_qc/measurement/grouping.py", "clifford_qc/models/fcidump.py",
)


def _named_source_bindings():
    # Named sources for this exploratory audit, not a complete import closure.
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in NAMED_SOURCES}


# Hash the named sources before this process imports the package, so a CLI
# record binds the code that was loaded rather than whatever is on disk once
# the audit starts. An earlier import in the same process (a test session, a
# notebook) defeats this, and the record then says so.
_SOURCES_AT_IMPORT = _named_source_bindings()
_HASHED_BEFORE_IMPORT = "clifford_qc" not in sys.modules

from clifford_qc import clear_kernel_caches
from clifford_qc.models import fcidump_model, read_fcidump
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.subspace import MatrixElementBank, determinant_excitations, identity_generator
from clifford_qc.subspace.linalg import DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S
from clifford_qc.subspace.symmetry_pools import (
    closed_shell_occupied_spatial, point_group_filter, spin_adapted_excitations,
)

SOLVER_POLICY = {"tau_s": DEFAULT_TAU_S, "rel_tau": 0.0,
                 "max_condition": DEFAULT_MAX_CONDITION, "norm_floor": DEFAULT_NORM_FLOOR}


def _ledger(bank):
    # Every field read here is a whole-cache counter, whatever subset is
    # passed. The identity-only subset just keeps the per-subset fields that
    # resources() also computes (row supports, subset universe) cheap.
    resource = bank.resources([0])
    return {"pairs_built": resource["pairs_built"],
            "operator_rows": resource["resident_operator_rows"],
            "coefficient_occurrences": resource["coefficient_occurrences"],
            "words_including_identity": resource["resident_word_universe"],
            "qwc_settings": bank.qwc_group_count()}


def _git_provenance(root):
    unavailable = {"git_head": None, "dirty_tree": None, "git_provenance_verified": False}
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        # An exported tree nested inside another repository must not borrow
        # that repository's commit as its own implementation identity.
        if Path(git('rev-parse', '--show-toplevel')).resolve() != root.resolve():
            return unavailable
        return {"git_head": git('rev-parse', 'HEAD'),
                "dirty_tree": bool(git('status', '--porcelain')), "git_provenance_verified": True}
    except (subprocess.CalledProcessError, OSError):
        return unavailable


def build_report(path, *, orbital_irreps=None, hamiltonian_atol=0.0, loaded_sources=None):
    """Audit one bounded FCIDUMP.

    ``loaded_sources`` is a named-source snapshot taken before the package was
    imported (the CLI passes one). Without it, sources are first hashed here.
    """
    current = _named_source_bindings()
    sources = current if loaded_sources is None else dict(loaded_sources)
    if current != sources:
        raise RuntimeError("source changed after import")
    provenance = _git_provenance(ROOT)
    data = read_fcidump(path)
    if data.n_orbitals > 4 or data.n_electrons % 2 or data.ms2 != 0:
        raise ValueError("this complete-bank audit requires <=4 spatial orbitals and a closed-shell reference")
    model = fcidump_model(path)
    if model.metadata['source_sha256'] != data.source_sha256:
        raise RuntimeError("input changed during audit")
    spin_ordering = model.metadata['spin_convention']
    reference = DeterminantReference.from_program(model.reference)
    occupied = reference.occupied
    # Both arms excite the same determinant: read it, never assume a prefix.
    occupied_spatial = closed_shell_occupied_spatial(
        occupied, data.n_orbitals, spin_ordering=spin_ordering)
    raw = determinant_excitations(model.n, occupied)
    adapted = spin_adapted_excitations(data.n_orbitals, occupied_spatial,
                                       spin_ordering=spin_ordering)
    arms = [("number_sz", raw), ("spin_free", adapted)]
    if orbital_irreps is not None:
        symmetry = dict(hamiltonian=model.hamiltonian, hamiltonian_atol=hamiltonian_atol,
                        spin_ordering=spin_ordering)
        arms.extend([
            ("number_sz_point_group", point_group_filter(raw, orbital_irreps, **symmetry)),
            ("spin_free_point_group", spin_adapted_excitations(
                data.n_orbitals, occupied_spatial, orbital_irreps=orbital_irreps, **symmetry)),
        ])
    rows = []
    for name, pool in arms:
        clear_kernel_caches()
        started = time.perf_counter()
        bank = MatrixElementBank(reference, model.hamiltonian, [identity_generator(model.n)] + pool)
        bank.entry(0, 0)
        for j in range(1, len(bank)):
            bank.entry(0, j)
            bank.entry(j, j)
        initial = _ledger(bank)
        screened = time.perf_counter()
        result = bank.solve(**SOLVER_POLICY)
        complete = _ledger(bank)
        finished = time.perf_counter()
        rows.append({"arm": name, "generators_excluding_identity": len(pool),
                     "initial_screen": initial, "initial_screen_seconds": screened - started,
                     "complete_candidate_bank": complete,
                     "completion_seconds": finished - screened,
                     "total_seconds": finished - started,
                     "energy": result.ground_energy, "retained_rank": result.effective_rank})
    if _named_source_bindings() != sources:
        raise RuntimeError("source changed during audit")
    if read_fcidump(path).source_sha256 != data.source_sha256:
        raise RuntimeError("input changed during audit")
    if _git_provenance(ROOT) != provenance:
        raise RuntimeError("Git state changed during audit")
    return {"schema": "clifford_qc.symmetry_pool_audit.v1", "exploratory": True,
            "physical_shots": 0, "finite_shot_cost_claim": False,
            "input_sha256": data.source_sha256, "n_spatial": data.n_orbitals,
            "n_electrons": data.n_electrons, "spin_ordering": spin_ordering,
            "occupied_spatial": list(occupied_spatial),
            "orbital_irreps": None if orbital_irreps is None else [int(p) for p in orbital_irreps],
            "irrep_convention": "explicit_real_abelian_xor_characters",
            "hamiltonian_atol": float(hamiltonian_atol),
            "fcidump_orbsym_inferred": False,
            "screen_policy": "identity_plus_each_candidate_2x2",
            "solver_policy": SOLVER_POLICY, "energy_units": "hartree",
            "timing_policy": ("one_cold_kernel_cache_run_per_arm; initial_screen_seconds covers "
                              "screen pairs and their ledger, completion_seconds the remaining "
                              "pairs, solve and complete ledger; each ledger includes QWC grouping"),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "platform": platform.platform(), "processor": platform.processor(),
                            "threads": {key: os.environ.get(key) for key in
                                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
            **provenance, "named_source_sha256": sources,
            "named_sources_hashed_before_import": loaded_sources is not None,
            "rows": rows}


def _atol(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fcidump', type=Path, default=ROOT/'benchmarks/data/h4_sto3g_r0.9.FCIDUMP')
    parser.add_argument('--orbital-irreps', type=int, nargs='+', help='explicit spatial-orbital XOR characters')
    parser.add_argument('--hamiltonian-atol', type=_atol, default=0.0,
                        help='largest tolerated |coefficient| of a symmetry-forbidden Hamiltonian word')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("audit output already exists")
    record = build_report(args.fcidump, orbital_irreps=args.orbital_irreps,
                          hamiltonian_atol=args.hamiltonian_atol,
                          loaded_sources=_SOURCES_AT_IMPORT if _HASHED_BEFORE_IMPORT else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()

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

from clifford_qc import clear_kernel_caches
from clifford_qc.models import fcidump_model, read_fcidump
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.subspace import MatrixElementBank, determinant_excitations, identity_generator
from clifford_qc.subspace.linalg import DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S
from clifford_qc.subspace.symmetry_pools import point_group_filter, spin_adapted_excitations

SOLVER_POLICY = {"tau_s": DEFAULT_TAU_S, "rel_tau": 0.0,
                 "max_condition": DEFAULT_MAX_CONDITION, "norm_floor": DEFAULT_NORM_FLOOR}


def _ledger(bank, indices):
    resource = bank.resources(indices)
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
    except (subprocess.CalledProcessError, FileNotFoundError):
        return unavailable


def build_report(path, *, orbital_irreps=None):
    data = read_fcidump(path)
    if data.n_orbitals > 4 or data.n_electrons % 2 or data.ms2 != 0:
        raise ValueError("this complete-bank audit requires <=4 spatial orbitals and a closed-shell reference")
    model = fcidump_model(path)
    reference = DeterminantReference.from_program(model.reference)
    occupied = reference.occupied
    occupied_spatial = tuple(range(data.n_electrons // 2))
    raw = determinant_excitations(model.n, occupied)
    adapted = spin_adapted_excitations(data.n_orbitals, occupied_spatial)
    arms = [("number_sz", raw), ("spin_free", adapted)]
    if orbital_irreps is not None:
        arms.extend([
            ("number_sz_point_group", point_group_filter(raw, orbital_irreps, hamiltonian=model.hamiltonian)),
            ("spin_free_point_group", spin_adapted_excitations(
                data.n_orbitals, occupied_spatial, orbital_irreps=orbital_irreps,
                hamiltonian=model.hamiltonian)),
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
        initial = _ledger(bank, [0])
        initial_seconds = time.perf_counter() - started
        result = bank.solve(**SOLVER_POLICY)
        complete = _ledger(bank, None)
        rows.append({"arm": name, "generators_excluding_identity": len(pool),
                     "initial_screen": initial, "initial_screen_seconds": initial_seconds,
                     "complete_candidate_bank": complete,
                     "complete_bank_seconds": time.perf_counter() - started,
                     "energy": result.ground_energy, "retained_rank": result.effective_rank})
    # Named sources for this exploratory audit, not a complete import closure.
    sources = ("benchmarks/benchmark_symmetry_pools.py", "clifford_qc/subspace/symmetry_pools.py",
               "clifford_qc/subspace/fermionic_generators.py", "clifford_qc/subspace/projection.py",
               "clifford_qc/subspace/linalg.py",
               "clifford_qc/reference_states.py", "clifford_qc/fermion.py", "clifford_qc/multivector.py",
               "clifford_qc/pauli_kernel.py", "clifford_qc/measurement/grouping.py",
               "clifford_qc/models/fcidump.py")
    return {"schema": "clifford_qc.symmetry_pool_audit.v1", "exploratory": True,
            "physical_shots": 0, "finite_shot_cost_claim": False,
            "input_sha256": data.source_sha256, "n_spatial": data.n_orbitals,
            "n_electrons": data.n_electrons,
            "orbital_irreps": None if orbital_irreps is None else [int(p) for p in orbital_irreps],
            "irrep_convention": "explicit_real_abelian_xor_characters",
            "fcidump_orbsym_inferred": False,
            "screen_policy": "identity_plus_each_candidate_2x2",
            "solver_policy": SOLVER_POLICY, "energy_units": "hartree",
            "timing_policy": "one_cold_kernel_cache_run_per_arm_including_qwc_grouping",
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "platform": platform.platform(), "processor": platform.processor(),
                            "threads": {key: os.environ.get(key) for key in
                                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
            **_git_provenance(ROOT),
            "named_source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
            "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fcidump', type=Path, default=ROOT/'benchmarks/data/h4_sto3g_r0.9.FCIDUMP')
    parser.add_argument('--orbital-irreps', type=int, nargs='+', help='explicit spatial-orbital XOR characters')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("audit output already exists")
    record = build_report(args.fcidump, orbital_irreps=args.orbital_irreps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()

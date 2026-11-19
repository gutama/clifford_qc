"""Exploratory: symmetry-shift control variates on finite-shot Ritz values.

The banks here are molecular: their generators conserve particle number and
``S_z`` and their references lie in one sector, so
``sector_shift_family`` certifies and every correction term has exactly zero
mean. Three modes:

- ``exact``: no shots. The grouped Neyman shot variance of the Ritz functional
  ``B†(H - E)B``, regrouped into its own qubit-wise-commuting settings, before
  and after a shift fitted with each objective of ``fit_sector_shift``. The
  bank matrices and energy are checked unchanged by every shift.
- ``caches``: one fixed-size measurement per replica. The raw Ritz value is
  compared with the control-variate correction fitted from the same cache and
  with coefficients fitted exactly beforehand (an oracle a real run lacks).
- ``acquisition``: ``acquire_ritz`` over a fifteen-look schedule, uniform and
  Ritz-targeted, with and without control variates. A stopping rule only
  truncates the run it would otherwise have continued, so every run spends
  the whole schedule and each stopping target is applied to its ledger
  afterwards. The uncorrected uniform arm is the raw ledger of the corrected
  uniform run: the same seed draws the same shots.

This is not a declared study. Systems, seeds, schedule and targets were chosen
while the method was being developed, and no frozen record is rewritten. The
coverage reported for a stopped run uses the schedule-valid interval that
``acquire_ritz`` reports, over every look of the schedule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMED_SOURCES = (
    "benchmarks/explore_control_variates.py", "clifford_qc/subspace/symmetry_shifts.py",
    "clifford_qc/subspace/acquisition.py", "clifford_qc/subspace/symmetry_pools.py",
    "clifford_qc/subspace/linalg.py", "clifford_qc/measurement/functionals.py",
    "clifford_qc/measurement/cache.py", "clifford_qc/measurement/session.py",
    "clifford_qc/backends/finite_shot.py",
)


def _named_source_bindings():
    # Named sources for this exploratory run, not a complete import closure.
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in NAMED_SOURCES}


# Hashed before the package is imported, as in benchmark_symmetry_pools.py.
_SOURCES_AT_IMPORT = _named_source_bindings()
_HASHED_BEFORE_IMPORT = "clifford_qc" not in sys.modules

from clifford_qc.backends.finite_shot import FiniteShotBackend  # noqa: E402
from clifford_qc.dense_reference import density_from_statevector  # noqa: E402
from clifford_qc.matrix import to_matrix  # noqa: E402
from clifford_qc.measurement.session import SharedMeasurement  # noqa: E402
from clifford_qc.models import fcidump_model  # noqa: E402
from clifford_qc.reference_states import DeterminantReference  # noqa: E402
from clifford_qc.subspace import MatrixElementBank, identity_generator  # noqa: E402
from clifford_qc.subspace.acquisition import acquire_ritz  # noqa: E402
from clifford_qc.subspace.symmetry_pools import spin_adapted_excitations  # noqa: E402
from clifford_qc.subspace.symmetry_shifts import (  # noqa: E402
    OBJECTIVES, fit_sector_shift, grouped_shot_variance, ritz_control_variate,
    ritz_word_functional, sector_shift_family,
)

# Three spatial orbitals, two electrons: small enough for hundreds of replicas.
TRI = ('&FCI NORB=3, NELEC=2, MS2=0, ORBSYM=1,1,1, ISYM=1 /\n'
       '0.6 1 1 1 1\n0.5 2 2 2 2\n0.4 3 3 3 3\n0.1 2 1 2 1\n0.1 3 1 3 1\n'
       '0.12 3 2 3 2\n0.3 2 2 1 1\n0.3 3 3 1 1\n0.3 3 3 2 2\n0.05 2 1 1 1\n0.04 3 2 2 2\n'
       '-0.9 1 1 0 0\n-1.0 2 2 0 0\n0.3 3 3 0 0\n0.05 3 2 0 0\n0.07 2 1 0 0\n0.0 0 0 0 0\n')
FCIDUMPS = {"h4": "benchmarks/data/h4_sto3g_r0.9.FCIDUMP",
            "lih": "benchmarks/data/lih_sto3g_r1.5949_cas4e4o.FCIDUMP"}
SYSTEMS = ("tri", *FCIDUMPS)
CUMULATIVE = (100, 141, 200, 283, 400, 566, 800, 1131, 1600, 2263, 3200, 4525, 6400, 9051,
              12800)
ACQUISITION_RUNS = (("uniform_cv", "uniform", True), ("targeted_raw", "ritz_targeted", False),
                    ("targeted_cv", "ritz_targeted", True))
STATE: dict = {}


def build_system(name: str, *, prepare: bool = True) -> dict:
    """The bank, its exact energy and its certified shift family."""
    if name == "tri":
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tri.FCIDUMP"
            path.write_text(TRI)
            model = fcidump_model(path)
        n_spatial, electrons = 3, 2
        # A correlated singlet-sector state: the bank is not built on a determinant.
        rng = np.random.default_rng(5)
        psi = np.zeros(64, complex)
        for a in range(3):
            for b in range(3):
                bits = ["0"] * 6
                bits[2 * a] = bits[2 * b + 1] = "1"
                psi[int("".join(bits), 2)] += rng.normal() * (1.0 if a == b else 0.3)
        generators = [identity_generator(6)] + spin_adapted_excitations(3, (0,))
    else:
        model = fcidump_model(ROOT / FCIDUMPS[name])
        n_spatial, electrons = 4, 4
        generators = [identity_generator(8)] + spin_adapted_excitations(4, (0, 1))
        # The CISD state in this pool: the bank is not built on a determinant.
        determinant = DeterminantReference.from_program(model.reference)
        cisd = MatrixElementBank(determinant, model.hamiltonian, generators).solve()
        hartree_fock = np.zeros(256, complex)
        hartree_fock[int("".join("1" if q in determinant.occupied else "0"
                                 for q in range(8)), 2)] = 1
        psi = sum(complex(c) * (to_matrix(g.mv) @ hartree_fock)
                  for c, g in zip(cisd.ritz_vector(0), generators))
    psi = psi / np.linalg.norm(psi)
    hamiltonian = model.hamiltonian.to_mv()
    rho = density_from_statevector(psi)
    shared = SharedMeasurement(MatrixElementBank(rho, hamiltonian, generators))
    exact = shared.bank.solve()
    family = sector_shift_family(n_spatial, electrons, sz=0.0)
    oracle = fit_sector_shift(hamiltonian, family, objective="ritz", reference=rho,
                              generators=generators, ritz_coefficients=exact.ritz_vector(0),
                              energy=exact.ground_energy).coefficients
    system = {"shared": shared, "exact": float(exact.ground_energy), "family": family,
              "oracle": oracle, "hamiltonian": hamiltonian, "rho": rho,
              "generators": generators, "solved": exact, "n_spatial": n_spatial,
              "electrons": electrons}
    if prepare:
        # Compile every setting's outcome distribution once; replicas only reseed.
        system["backend"] = FiniteShotBackend(seed=0)
        system["backend"].prepare_grouped_state(shared.bank.reference, shared.groups)
    return system


EXACT_FAMILIES = (("N", {"one_body": False}), ("N+one_body", {"one_body": True}),
                  ("N+one_body+Sz", {"one_body": True, "sz": 0.0}))


def exact_table() -> dict:
    """Grouped Neyman shot variance of the Ritz functional, unshifted and shifted."""
    hamiltonian, rho, generators = STATE["hamiltonian"], STATE["rho"], STATE["generators"]
    solved = STATE["solved"]
    vector, energy = solved.ritz_vector(0), float(solved.ground_energy)
    reference = STATE["shared"].bank.matrices()
    table = {"unshifted": grouped_shot_variance(
        ritz_word_functional(hamiltonian, generators, vector, energy), rho)}
    for label, options in EXACT_FAMILIES:
        family = sector_shift_family(STATE["n_spatial"], STATE["electrons"], **options)
        for objective in OBJECTIVES:
            fit = fit_sector_shift(hamiltonian, family, objective=objective, reference=rho,
                                   generators=generators, ritz_coefficients=vector,
                                   energy=energy)
            shifted = MatrixElementBank(rho, fit.hamiltonian, generators)
            matrices = shifted.matrices()
            entry = grouped_shot_variance(
                ritz_word_functional(fit.hamiltonian, generators, vector, energy), rho)
            entry.update(
                neyman_ratio=table["unshifted"]["neyman"] / entry["neyman"],
                largest_matrix_change=max(float(np.max(np.abs(a - b)))
                                          for a, b in zip(matrices, reference)),
                energy_change=float(shifted.solve().ground_energy) - energy,
                fit_before=fit.before, fit_after=fit.after)
            table[f"{label}/{objective}"] = entry
    return table


def replica_seed(root: int, replica: int) -> int:
    return int(np.random.SeedSequence([root, replica]).generate_state(1, np.uint64)[0])


def cache_row(task: tuple[int, int, int]) -> dict:
    root, replica, shots = task
    shared, family, exact = STATE["shared"], STATE["family"], STATE["exact"]
    seed = replica_seed(root, replica)
    backend = STATE["backend"]
    backend.reseed(seed)
    cache = shared.measure(backend, shots)
    try:
        point = shared.solve(cache)
    except (ValueError, np.linalg.LinAlgError) as error:
        return {"replica": replica, "seed": seed, "failed": f"{type(error).__name__}: {error}"}
    fitted = ritz_control_variate(shared, cache, point, family)
    known = ritz_control_variate(shared, cache, point, family, coefficients=STATE["oracle"])
    return {"replica": replica, "seed": seed, "rank": int(point.effective_rank),
            "raw": [fitted.raw_energy - exact, fitted.raw_std_error],
            "fitted": [fitted.energy - exact, fitted.std_error],
            "oracle": [known.energy - exact, known.std_error],
            "unreadable": list(fitted.unreadable)}


def acquisition_row(task: tuple[int, int, int]) -> dict:
    root, replica, _ = task
    shared, family, exact = STATE["shared"], STATE["family"], STATE["exact"]
    groups = len(shared.groups)
    budgets = [(b - a) * groups for a, b in zip(CUMULATIVE, CUMULATIVE[1:])]
    seed = replica_seed(root, replica)
    row: dict = {"replica": replica, "seed": seed}
    for arm, allocation, corrected in ACQUISITION_RUNS:
        backend = STATE["backend"]
        backend.reseed(seed)
        try:
            run = acquire_ritz(shared, backend, pilot_shots_per_group=CUMULATIVE[0],
                               round_budgets=budgets, allocation=allocation,
                               control_variates=family if corrected else None)
        except ValueError as error:   # AcquisitionError included
            row[arm] = {"failed": f"{type(error).__name__}: {error}"}
            continue
        looks = run.looks
        entry = {"shots": [look["shots"] for look in looks],
                 "error": [look["energy"] - exact for look in looks],
                 "sigma": [look["std_error"] for look in looks],
                 "rank": [look["retained_rank"] for look in looks]}
        if corrected:
            entry["raw_error"] = [look["raw_energy"] - exact for look in looks]
            entry["raw_sigma"] = [look["raw_std_error"] for look in looks]
            entry["terms_used"] = [look["control_variates_used"] for look in looks]
        row[arm] = entry
    return row


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    scale = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / scale
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / scale
    return [centre - half, centre + half]


def _arm_summary(errors, sigmas, z) -> dict:
    errors, sigmas = np.asarray(errors), np.asarray(sigmas)
    covered = int(np.sum(np.abs(errors) <= z * sigmas))
    return {"rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_error": float(errors.mean()),
            "rms_sigma": float(np.sqrt(np.mean(sigmas ** 2))),
            "coverage": covered / len(errors), "coverage_wilson": wilson(covered, len(errors))}


def summarize_caches(rows: list[dict]) -> dict:
    ok = [row for row in rows if "failed" not in row]
    z = 1.959963984540054
    summary = {"replicas": len(rows), "failures": len(rows) - len(ok)}
    for arm in ("raw", "fitted", "oracle"):
        summary[arm] = _arm_summary([row[arm][0] for row in ok], [row[arm][1] for row in ok], z)
    for arm in ("fitted", "oracle"):
        summary[arm]["mse_ratio_vs_raw"] = (summary["raw"]["rmse"] / summary[arm]["rmse"]) ** 2
    summary["unreadable_terms"] = sorted({term for row in ok for term in row["unreadable"]})
    return summary


def _ledgers(row: dict) -> dict:
    uniform = row["uniform_cv"]
    return {"uniform_raw": {"shots": uniform["shots"], "error": uniform["raw_error"],
                            "sigma": uniform["raw_sigma"]},
            "uniform_cv": uniform, "targeted_raw": row["targeted_raw"],
            "targeted_cv": row["targeted_cv"]}


def summarize_acquisition(rows: list[dict], targets) -> dict:
    from statistics import NormalDist

    runs = [arm for arm, _, _ in ACQUISITION_RUNS]
    ok = [row for row in rows if not any("failed" in row[arm] for arm in runs)]
    looks = len(CUMULATIVE)
    schedule_z = NormalDist().inv_cdf(1 - 0.05 / (2 * looks))
    single_z = NormalDist().inv_cdf(0.975)
    ledgers = [_ledgers(row) for row in ok]
    arms = ("uniform_raw", "uniform_cv", "targeted_raw", "targeted_cv")
    summary: dict = {"replicas": len(rows), "failures": len(rows) - len(ok),
                     "schedule_z": schedule_z, "fixed": {}, "stopping": {}}
    for index in range(looks):
        cell = {arm: _arm_summary([ledger[arm]["error"][index] for ledger in ledgers],
                                  [ledger[arm]["sigma"][index] for ledger in ledgers], single_z)
                for arm in arms}
        for arm in arms:
            cell[arm]["mse_ratio_vs_uniform_raw"] = (cell["uniform_raw"]["rmse"]
                                                     / cell[arm]["rmse"]) ** 2
        summary["fixed"][str(CUMULATIVE[index])] = cell
    for target in targets:
        cell = {}
        for arm in arms:
            shots, errors, sigmas, met = [], [], [], 0
            for ledger in ledgers:
                entry = ledger[arm]
                stop = next((k for k, s in enumerate(entry["sigma"]) if s <= target),
                            looks - 1)
                met += entry["sigma"][stop] <= target
                shots.append(entry["shots"][stop])
                errors.append(entry["error"][stop])
                sigmas.append(entry["sigma"][stop])
            cell[arm] = {"met": met, "mean_shots": float(np.mean(shots)),
                         "schedule_valid": _arm_summary(errors, sigmas, schedule_z),
                         "naive": _arm_summary(errors, sigmas, single_z)}
        for arm in arms:
            cell[arm]["shot_ratio_uniform_raw_over_this"] = (
                cell["uniform_raw"]["mean_shots"] / cell[arm]["mean_shots"])
        summary["stopping"][repr(float(target))] = cell
    return summary


def _git_provenance() -> dict:
    unavailable = {"git_head": None, "dirty_tree": None, "git_provenance_verified": False}

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT.resolve():
            return unavailable
        return {"git_head": git("rev-parse", "HEAD"),
                "dirty_tree": bool(git("status", "--porcelain")),
                "git_provenance_verified": True}
    except (subprocess.CalledProcessError, OSError):
        return unavailable


def run(system: str, mode: str, *, root: int, replicas: int, shots: int | None,
        targets, workers: int) -> dict:
    started = time.time()
    STATE.clear()
    STATE.update(build_system(system, prepare=mode != "exact"))
    shared = STATE["shared"]
    if mode == "exact":
        rows, summary = [], exact_table()
        replicas = 0
        return _record(system, mode, root, replicas, shots, targets, summary, rows, started)
    work = cache_row if mode == "caches" else acquisition_row
    tasks = [(root, replica, shots or 0) for replica in range(replicas)]
    if workers > 1:
        # Forked workers inherit the bank and the compiled distributions.
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            rows = sorted(pool.imap_unordered(work, tasks), key=lambda row: row["replica"])
    else:
        rows = [work(task) for task in tasks]
    summary = (summarize_caches(rows) if mode == "caches"
               else summarize_acquisition(rows, targets))
    return _record(system, mode, root, replicas, shots, targets, summary, rows, started)


def _record(system, mode, root, replicas, shots, targets, summary, rows, started) -> dict:
    shared = STATE["shared"]
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = None
    return {
        "schema": "clifford_qc.explore_control_variates.v1",
        "exploratory": True, "system": system, "mode": mode, "seed_root": root,
        "replicas": replicas, "shots_per_setting": shots,
        "cumulative_shots_per_setting": list(CUMULATIVE) if mode == "acquisition" else None,
        "targets": list(targets) if mode == "acquisition" else None,
        "words": len(shared.words), "settings": len(shared.groups),
        "exact_energy": STATE["exact"], "family": [term.label for term in STATE["family"]],
        "oracle_coefficients": STATE["oracle"], "summary": summary, "rows": rows,
        "seconds": time.time() - started,
        "provenance": {**_git_provenance(),
                       "named_sources": _SOURCES_AT_IMPORT if _HASHED_BEFORE_IMPORT else None,
                       "python": platform.python_version(), "numpy": np.__version__,
                       "scipy": scipy_version, "platform": platform.platform()},
    }


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--mode", choices=("exact", "caches", "acquisition"), required=True)
    parser.add_argument("--seed-root", type=int, help="required with shots")
    parser.add_argument("--replicas", type=_positive, help="required with shots")
    parser.add_argument("--shots", type=_positive, help="shots per setting (caches mode)")
    parser.add_argument("--targets", type=float, nargs="+", default=[0.01, 0.005],
                        help="stopping standard errors (acquisition mode)")
    parser.add_argument("--workers", type=_positive, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "caches" and args.shots is None:
        parser.error("--shots is required in caches mode")
    if args.mode != "exact" and (args.seed_root is None or args.replicas is None):
        parser.error("--seed-root and --replicas are required unless --mode exact")
    if any(not math.isfinite(t) or t <= 0 for t in args.targets):
        parser.error("targets must be positive and finite")
    if args.output.exists():
        raise FileExistsError("output already exists")
    record = run(args.system, args.mode, root=args.seed_root, replicas=args.replicas,
                 shots=args.shots, targets=args.targets, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record["summary"]["stopping"] if args.mode == "acquisition"
                     else record["summary"], indent=1))


if __name__ == "__main__":
    main()

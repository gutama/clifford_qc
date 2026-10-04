"""Run the preregistered Phase 15 measured-residual cost preflight (Q18), once.

Prices, on each of the five licensed banks, the shots that estimate the
ground Ritz root's residual norm against the shots that estimate its energy,
at one matched standard error, exactly as
``benchmarks/configs/phase15_measured_residual_preflight.json`` froze it. See
``benchmarks/PHASE15_MEASURED_RESIDUAL_PREREGISTRATION.md``.

The producer refuses before it forms a second-moment row unless all of the
following hold:

* the preregistration gate passes, including its recomputation of every
  frozen number and the lineage of every bound file;
* the run is the declared one: no bank subset is written to the committed path;
* no record exists yet, since the preflight runs once;
* the working tree is clean;
* the environment is one the committed records declare.

For each bank it builds the retained block's ``S``, ``H`` and ``K`` rows and
the raw combined universe, which must reproduce the H-squared preflight's size
and SHA-256. It forms the energy functional (the existing ``ritz_functional``)
and the residual functional (the gate's ``residual_functional``) at the exact
reference means, and checks the latter against finite differences of the full
nonlinear pipeline along the two frozen directions. Under each declared
grouping protocol it partitions both measured universes, sums the exact
per-setting variances, and reads

    R = (G_U * V_sigma2) / (4 sigma^2 * G_SH * V_E)

with the gate's ``cost_ratio``. The status comes off the gate's ladder and the
verdict off its combination rule. The Neyman ratio, both sides at
variance-optimal allocation, is a diagnostic no clause reads.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python benchmarks/run_phase15_measured_residual_preflight.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

try:
    from benchmarks import check_phase15_measured_residual_preregistration as gate
    from benchmarks import check_phase15_preregistration as preflight_gate
    from benchmarks.run_phase15_h2_preflight import _digest
except ImportError:  # pragma: no cover - script execution
    import check_phase15_measured_residual_preregistration as gate
    import check_phase15_preregistration as preflight_gate
    from run_phase15_h2_preflight import _digest

SCHEMA = "clifford_qc.phase15_measured_residual_preflight.v1"
RECORD = gate.RECORD
# Deterministic-check tolerances, as the config's deterministic_checks state them.
VARIANCE_TOLERANCE = 1e-9
MEAN_TOLERANCE = 1e-9
ENERGY_VARIANCE_RELATIVE = 1e-9
CLAIM_BOUNDARY = (
    "This record reports the one measured-residual cost preflight "
    "benchmarks/configs/phase15_measured_residual_preflight.json froze. On each of the "
    "five licensed banks it records, under each declared grouping protocol, the "
    "single-assignment QWC settings and one-shot linearized variances of the ground Ritz "
    "root's energy and residual-norm estimators, their ratio R at one matched standard "
    "error, the status the frozen ladder gives, and the verdict. The ratios are "
    "first-order and asymptotic, read from exact reference variances with uniform shots "
    "per setting. They say nothing about finite-sample intervals, device time, other "
    "groupings or allocations, or any bank outside the five.")


# ------------------------------------------------------------------ pieces

def partition_problems(groups, universe: set[int]) -> list[str]:
    """A grouping must assign every measured word exactly once, in QWC settings."""
    from clifford_qc.measurement import shared_basis

    problems = []
    assigned = [word.code for group in groups for word in group]
    if len(assigned) != len(set(assigned)):
        problems.append("a word is assigned to more than one setting")
    if set(assigned) != set(universe) - {0}:
        problems.append("the settings do not cover exactly the measured universe")
    for group in groups:
        try:
            shared_basis(group)
        except ValueError:
            problems.append("a setting is not qubit-wise commuting")
            break
    return problems


def _norm(weights) -> float:
    return math.sqrt(sum(value * value for word, value in weights.items() if word != 0))


def price_protocol(reference, protocol: str, n: int, sh_words, combined_words,
                   energy_weights, residual_weights, variance: float) -> dict:
    """Both groupings under one protocol, their one-shot variances and R."""
    energy_groups = gate.group_words(sh_words, protocol, n)
    residual_groups = gate.group_words(combined_words, protocol, n)
    problems = (partition_problems(energy_groups, sh_words)
                + partition_problems(residual_groups, combined_words))
    try:
        energy_variances = gate.group_variances(reference, energy_groups, energy_weights)
        residual_variances = gate.group_variances(reference, residual_groups,
                                                  residual_weights)
        nonnegative = True
    except AssertionError:
        energy_variances, residual_variances, nonnegative = [], [], False
    energy_variance = float(sum(energy_variances))
    residual_variance = float(sum(residual_variances))
    energy_neyman = float(sum(math.sqrt(v) for v in energy_variances))
    residual_neyman = float(sum(math.sqrt(v) for v in residual_variances))
    priced = nonnegative and energy_variance > 0.0 and variance > 0.0
    ratio = (gate.cost_ratio(residual_settings=len(residual_groups),
                             residual_variance=residual_variance,
                             energy_settings=len(energy_groups),
                             energy_variance=energy_variance, variance=variance)
             if priced else None)
    neyman = (residual_neyman ** 2 / (4.0 * variance * energy_neyman ** 2)
              if priced and energy_neyman > 0.0 else None)
    return {
        "energy_settings": len(energy_groups),
        "energy_partitioned_words": sum(len(group) for group in energy_groups),
        "energy_variance_one_shot": energy_variance,
        "energy_neyman_sum": energy_neyman,
        "residual_settings": len(residual_groups),
        "residual_partitioned_words": sum(len(group) for group in residual_groups),
        "residual_variance_one_shot": residual_variance,
        "residual_neyman_sum": residual_neyman,
        "cost_ratio": ratio,
        "neyman_ratio": neyman,
        "partition_problems": problems,
        "variances_nonnegative": nonnegative,
    }


def evaluate_bank(config: dict, name: str, model, selected, frozen: dict,
                  validation_root: dict, *, progress=None) -> dict:
    """The rows, functionals, checks, prices and status of one bank."""
    from clifford_qc.measurement.functionals import ritz_functional
    from clifford_qc.subspace import SecondMomentBank
    from clifford_qc.subspace.second_moment import RESOLUTION

    def say(line):
        if progress is not None:
            progress(f"  {name}: {line}")

    started = time.perf_counter()
    grouping = config["grouping"]
    check = config["linearization"]["finite_difference"]
    threshold = float(config["statistic"]["max_ratio"])
    size = len(selected)
    order = list(range(size))

    bank = preflight_gate.first_moment_bank(model, selected)
    second = SecondMomentBank(bank)
    rows = gate.block_rows(bank, second, order)
    sh_words = set(bank.word_set())
    combined = gate.row_words(rows)
    say(f"rows built, raw |U_SH|={len(sh_words)} raw |U|={len(combined)}")

    means = gate.reference_means(bank.reference, combined)
    weights, variance = gate.residual_functional(rows, size, means)
    # The gate returns NumPy scalars; the record holds plain floats.
    residual_weights = {int(word): float(value) for word, value in weights.items()}
    result = bank.solve()
    energy = ritz_functional(bank, result.indices, result.ritz_vector(), result.ground_energy)
    energy_weights = dict(energy.coefficients)
    measured_residual = {w: v for w, v in residual_weights.items() if w != 0}
    scale = float(validation_root["cancellation_scale"])
    reference_mean = float(sum(value * means.get(word, 0.0)
                               for word, value in residual_weights.items()))
    say(f"functionals built, sigma^2={variance:.6g}")

    directions = gate.check_directions(residual_weights, sh_words, seed=int(check["seed"]))
    finite = {}
    for key in check["directions"]:
        row = gate.finite_difference_check(
            rows, size, means, residual_weights, directions[key], step=float(check["step"]),
            relative_tolerance=float(check["relative_tolerance"]),
            cancellation_scale=scale, resolution=RESOLUTION)
        finite[key] = {"numeric": float(row["numeric"]), "analytic": float(row["analytic"]),
                       "tolerance": float(row["tolerance"]), "passes": bool(row["passes"])}
    say("finite differences " + ", ".join(
        f"{key}={'pass' if value['passes'] else 'FAIL'}" for key, value in finite.items()))

    protocols = {}
    declared = grouping["declared_protocol_by_system"][name]
    alternative = grouping["alternative_protocol_by_system"][name]
    for role, protocol in (("declared", declared), ("alternative", alternative)):
        if protocol is None:
            continue
        priced = price_protocol(bank.reference, protocol, model.n, sh_words, combined,
                                energy_weights, measured_residual, variance)
        protocols[protocol] = {"role": role, **priced}
        say(f"{protocol}: G_SH={priced['energy_settings']} G_U={priced['residual_settings']} "
            f"R={priced['cost_ratio']}")

    committed_ve = frozen["mapping_axis_linearized_energy_variance"]
    declared_price = protocols[declared]
    checks = {
        "combined_universe_matches_preflight": (
            len(combined) == frozen["raw_combined_word_universe"]
            and _digest(combined) == frozen["raw_combined_words_sha256"]
            and 0 in combined),
        "energy_settings_reproduce_mapping_axis": (
            declared_price["energy_settings"] == frozen["energy_settings"]),
        "energy_variance_reproduces_mapping_axis": (
            committed_ve is None or math.isclose(
                declared_price["energy_variance_one_shot"], committed_ve,
                rel_tol=ENERGY_VARIANCE_RELATIVE, abs_tol=0.0)),
        "partitions_valid": all(not p["partition_problems"] for p in protocols.values()),
        "residual_functional_reproduces_variance": (
            abs(variance - float(validation_root["variance"]))
            <= VARIANCE_TOLERANCE * (1.0 + scale)),
        "residual_functional_mean_zero": bool(
            abs(reference_mean) <= MEAN_TOLERANCE * (1.0 + scale)),
        "linearization_matches_finite_differences": all(
            value["passes"] for value in finite.values()),
        "group_variances_nonnegative": all(
            p["variances_nonnegative"] for p in protocols.values()),
    }
    ratios = {p: protocols[p]["cost_ratio"] for p in protocols}
    if all(checks.values()) and all(value is not None for value in ratios.values()):
        status = gate.status_of(ratios[declared],
                                ratios[alternative] if alternative is not None else None,
                                threshold)
    else:
        status = "INVALID"
    residual_norm = math.sqrt(max(variance, 0.0))
    return {
        "n_qubits": model.n,
        "basis_size": size,
        "universes": {
            "raw_sh_word_universe": len(sh_words),
            "raw_combined_word_universe": len(combined),
            "measured_sh_words": len(sh_words - {0}),
            "measured_combined_words": len(combined - {0}),
            "raw_combined_words_sha256": _digest(combined),
        },
        "estimator": {
            "ground_energy": float(result.ground_energy),
            "variance": variance,
            "residual_norm": residual_norm,
            "matched_precision_half": residual_norm / 4.0,
            "cancellation_scale": scale,
            "residual_functional_words": len(measured_residual),
            "residual_functional_norm": _norm(residual_weights),
            "residual_identity_weight": float(residual_weights.get(0, 0.0)),
            "residual_reference_mean": reference_mean,
            "energy_functional_words": len(energy_weights),
            "energy_functional_norm": _norm(energy_weights),
        },
        "finite_differences": finite,
        "protocols": protocols,
        "mapping_axis_energy_variance": committed_ve,
        "deterministic_checks": checks,
        "status": status,
        "seconds": round(time.perf_counter() - started, 2),
    }


# ------------------------------------------------------------------ the run

def run_preflight(config: dict, *, config_bytes: bytes, banks=None, inputs=None,
                  validation: dict | None = None, structural: dict | None = None,
                  progress=None) -> dict:
    """The declared preflight as a record (unstamped). ``inputs`` is for tests."""
    inputs = preflight_gate.bank_inputs if inputs is None else inputs
    if validation is None:
        validation = json.loads(gate.VALIDATION.read_text(encoding="utf-8"))
    declared = list(config["banks"]["systems"])
    names = declared if banks is None else list(banks)
    started = time.perf_counter()
    record = {
        "schema": SCHEMA,
        "config_path": str(gate.CONFIG.relative_to(ROOT)),
        "config_digest": hashlib.sha256(config_bytes).hexdigest(),
        "claim_boundary": CLAIM_BOUNDARY,
        "preregistration": {
            "gate": "benchmarks/check_phase15_measured_residual_preregistration.py",
            "revision": config["revisions"][-1]["revision"],
            "config_claim_boundary_at_landing": config["claim_boundary"],
        },
        "quantum_advantage_claim": False,
        "evidence": config["evidence"],
        "statistic": config["statistic"],
        "grouping": config["grouping"],
        "finite_difference": config["linearization"]["finite_difference"],
        "banks": {},
    }
    for name in names:
        model, selected = inputs(name)
        frozen = config["measured_before_freezing"][name]
        measured = (structural or {}).get(name, frozen)
        if progress is not None:
            progress(f"{name}: n={model.n} M={len(selected)}")
        entry = evaluate_bank(config, name, model, selected, frozen,
                              validation["banks"][name]["roots"][0], progress=progress)
        entry["measured_at_execution"] = {k: measured[k] for k in gate.MEASURED_FIELDS}
        record["banks"][name] = entry
        if progress is not None:
            progress(f"{name}: {entry['status']} in {entry['seconds']} s")
    statuses = {name: record["banks"][name]["status"] for name in declared
                if name in record["banks"]}
    complete = len(statuses) == len(declared)
    verdict = gate.verdict_of(statuses.values()) if complete else "INCOMPLETE"
    record["decision"] = {
        "statuses": statuses,
        "verdict": verdict,
        "consequence": config["consequences"].get(verdict),
        "rule": "frozen in the config; see decision_rule",
    }
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    return record


def refusals(args, *, reduced: bool) -> list[str]:
    problems = []
    if args.out.resolve() == RECORD.resolve():
        if reduced:
            problems.append("a bank subset (--banks) is not the declared preflight and "
                            "may not write the committed record")
        if RECORD.exists() and not args.overwrite:
            problems.append(f"{RECORD.name} exists: the declared preflight runs once "
                            "(pass --overwrite only to regenerate it deliberately)")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RECORD,
                        help="record path (default: the committed record)")
    parser.add_argument("--banks", nargs="*", default=None,
                        help="bank subset; refused for the committed path")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate an existing committed record deliberately")
    args = parser.parse_args(argv)
    problems = refusals(args, reduced=args.banks is not None)
    if problems:
        for problem in problems:
            print(f"REFUSE {problem}")
        return 1

    from clifford_qc.reproducibility import execution_provenance, stamp_record

    config_bytes = gate.CONFIG.read_bytes()
    config = gate.load_config()
    problems = gate.static_problems(config)
    notes: list[str] = []
    computed: dict = {}
    if not problems:
        problems = gate.structural_problems(config, notes, computed)
    if problems:
        for problem in problems:
            print(f"REFUSE preregistration gate: {problem}")
        return 1
    for note in notes:
        print(note)
    provenance = execution_provenance()
    if args.out.resolve() == RECORD.resolve() and provenance["git_dirty"]:
        print("REFUSE the working tree is dirty; commit first so the record's "
              "provenance names the code that produced it")
        return 1
    # Raises before any row is formed when the environment is not a declared one.
    stamp_record({}, provenance)
    record = run_preflight(config, config_bytes=config_bytes, banks=args.banks,
                           structural=computed,
                           progress=lambda line: print(line, flush=True))
    stamped = stamp_record(record, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamped, indent=1) + "\n", encoding="utf-8")
    decision = record["decision"]
    print(f"\nverdict: {decision['verdict']}  statuses: {decision['statuses']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

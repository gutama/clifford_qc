"""Run the preregistered Phase 15 variance-extrapolation test (Q17), once.

Fits the prefix trajectory of each frozen mapping-axis basis exactly as
``benchmarks/configs/phase15_variance_extrapolation.json`` froze it. See
``benchmarks/PHASE15_EXTRAPOLATION_PREREGISTRATION.md``.

The producer refuses before it computes a prefix variance unless all of the
following hold:

* the preregistration gate passes, including its recomputation of the exact
  spectra, the final Ritz energies and the domain;
* the run is the declared one: no bank subset is written to the committed path;
* no record exists yet, since the test runs once;
* the working tree is clean;
* the environment is one the committed records declare.

For each bank it builds one ``SecondMomentBank`` over the full frozen basis.
Every prefix block is its top-left corner, so the trajectory costs one solve
per prefix. It records ``E_M``, ``sigma^2_M`` and the resolved flag for
``M = 1..M_f``. It fits the frozen window with the gate's own rule and reads
the status off the gate's own ladder. Then it runs the declared deterministic
checks:

* the final prefix reproduces the validation record's ground root;
* the energies are monotone;
* each window variance equals the matrix-free reconstructed residual;
* some exact eigenvalue lies within ``sigma_M`` of ``E_M`` at every prefix
  (Weinstein).

A failed check makes the bank INVALID. The verdict reads the required banks
only. Fits over other windows and the Temple bound are reported as
diagnostics.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python benchmarks/run_phase15_variance_extrapolation.py
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
    from benchmarks import check_phase15_extrapolation_preregistration as gate
    from benchmarks import check_phase15_preregistration as preflight_gate
    from benchmarks import run_second_moment_validation as validation_producer
except ImportError:  # pragma: no cover - script execution
    import check_phase15_extrapolation_preregistration as gate
    import check_phase15_preregistration as preflight_gate
    import run_second_moment_validation as validation_producer

SCHEMA = "clifford_qc.phase15_variance_extrapolation.v1"
RECORD = gate.RECORD
CHEMICAL_ACCURACY = 1.6e-3
MONOTONE_TOLERANCE = 1e-10
CLAIM_BOUNDARY = (
    "This record reports the one variance-extrapolation test "
    "benchmarks/configs/phase15_variance_extrapolation.json froze. On each frozen "
    "mapping-axis basis, it records the prefix trajectory of exact Ritz ground energies "
    "and variances, a straight line through its last three points read at zero "
    "variance, and that value's error against the exact sector ground energy beside the "
    "final Ritz energy's. The verdict reads the banks inside the declared domain only. "
    "It judges the rule on exact inputs; it says nothing about finite-shot variances, "
    "other windows, or bases outside the five.")


def _fit(points, *, trust_resolved: bool = False):
    return gate.extrapolate([r["variance"] for r in points], [r["energy"] for r in points],
                            resolved=[True if trust_resolved else r["resolved"]
                                      for r in points],
                            scales=[r["cancellation_scale"] for r in points])


def _window(rows, size):
    return _fit(rows[-size:])


def _all_resolved(rows):
    """A diagnostic line through every resolved prefix, when there are two."""
    points = [r for r in rows if r["resolved"]]
    return _fit(points, trust_resolved=True) if len(points) >= 2 else None


def trajectory(model, selected, spectrum):
    """Every prefix's Ritz ground energy and variance, and the dense window data."""
    from clifford_qc.subspace import SecondMomentBank
    from clifford_qc.subspace.second_moment import RESOLUTION

    bank = preflight_gate.first_moment_bank(model, selected)
    moments = SecondMomentBank(bank)
    moments.matrix()
    rows, results = [], []
    for size in range(1, len(selected) + 1):
        result = bank.solve(list(range(size)))
        residual = moments.residual(result, 0)
        floor = math.sqrt(RESOLUTION * residual.cancellation_scale)
        nearest = min(abs(value - residual.energy) for value in spectrum)
        rows.append({
            "prefix": size, "energy": residual.energy, "variance": residual.variance,
            "residual_norm": residual.residual_norm, "resolved": residual.resolved,
            "cancellation_scale": residual.cancellation_scale,
            "nearest_eigenvalue_distance": nearest,
            "weinstein_holds": nearest <= max(residual.residual_norm, floor) * (1 + 1e-9),
        })
        results.append(result)
    return bank, rows, results


def evaluate_bank(config: dict, name: str, model, selected, frozen: dict,
                  validation_root: dict) -> dict:
    """The trajectory, the frozen fit, the checks and the status of one bank."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    started = time.perf_counter()
    spectrum, _ = gate.sector_spectrum(model, count=2 ** model.n)
    ground, first = spectrum[0], spectrum[1]
    bank, rows, results = trajectory(model, selected, spectrum)
    window = int(config["fit"]["window"])
    factor = float(config["fit"]["improvement_factor"])
    dense = []
    for row, result in zip(rows[-window:], results[-window:]):
        reference = validation_producer.dense_residuals(
            model, selected, bank.hamiltonian, result)[0]
        dense.append({"prefix": row["prefix"], "dense_residual_norm": reference,
                      "variance_minus_dense": row["variance"] - reference ** 2})
    final = rows[-1]
    checks = {
        "final_prefix_matches_validation": (
            abs(final["energy"] - validation_root["energy"]) <= 1e-10
            and abs(final["variance"] - validation_root["variance"])
            <= RESOLUTION * final["cancellation_scale"]),
        "energies_monotone": all(b["energy"] <= a["energy"] + MONOTONE_TOLERANCE
                                 for a, b in zip(rows, rows[1:])),
        "window_variances_match_dense": all(
            abs(d["variance_minus_dense"]) <= RESOLUTION * r["cancellation_scale"]
            for d, r in zip(dense, rows[-window:])),
        "weinstein_interval": all(r["weinstein_holds"] for r in rows),
        "exact_matches_provenance": bool(frozen["exact_matches_provenance"]),
    }
    fit = _window(rows, window)
    final_error = final["energy"] - ground
    extrapolated_error = fit["intercept"] - ground if fit["extrapolable"] else None
    if all(checks.values()):
        status = gate.status_of(fit["extrapolable"], extrapolated_error, final_error, factor)
    else:
        status = "INVALID"
    # Chemical accuracy is a statement in hartree: FCIDUMP models carry their
    # source digest, lattice models do not.
    molecular = "source_sha256" in model.metadata
    diagnostics = {
        "window_2": _window(rows, 2),
        "window_4": _window(rows, 4) if len(rows) >= 4 else None,
        "all_resolved": _all_resolved(rows),
        "temple_lower_bound": (final["energy"] - final["variance"] / (first - final["energy"])
                               if final["energy"] < first else None),
        "reaches_chemical_accuracy_where_final_does_not": (
            molecular and extrapolated_error is not None
            and abs(extrapolated_error) <= CHEMICAL_ACCURACY
            and abs(final_error) > CHEMICAL_ACCURACY),
    }
    return {
        "role": "required" if name in config["required_banks"] else "diagnostic",
        "units": "hartree" if molecular else "model",
        "exact_spectrum": spectrum,
        "trajectory": rows,
        "window": {"prefixes": [r["prefix"] for r in rows[-window:]], "fit": fit,
                   "dense": dense},
        "final_error": final_error,
        "extrapolated_energy": fit.get("intercept") if fit["extrapolable"] else None,
        "extrapolated_error": extrapolated_error,
        "gain": (abs(final_error) / abs(extrapolated_error)
                 if extrapolated_error not in (None, 0.0) else None),
        "deterministic_checks": checks,
        "status": status,
        "diagnostics": diagnostics,
        "seconds": round(time.perf_counter() - started, 2),
    }


def run_extrapolation(config: dict, *, config_bytes: bytes, banks=None, inputs=None,
                      validation: dict | None = None, structural: dict | None = None,
                      progress=None) -> dict:
    """The declared test as a record (unstamped). ``inputs`` is for tests."""
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
            "gate": "benchmarks/check_phase15_extrapolation_preregistration.py",
            "revision": config["revisions"][-1]["revision"],
            "config_claim_boundary_at_landing": config["claim_boundary"],
        },
        "quantum_advantage_claim": False,
        "evidence": config["evidence"],
        "fit": config["fit"],
        "domain": config["domain"],
        "banks": {},
    }
    for name in names:
        model, selected = inputs(name)
        frozen = config["measured_before_freezing"][name]
        measured = (structural or {}).get(name, frozen)
        if progress is not None:
            progress(f"{name}: n={model.n} M={len(selected)}")
        entry = evaluate_bank(config, name, model, selected, frozen,
                              validation["banks"][name]["roots"][0])
        entry["measured_at_execution"] = {k: measured[k] for k in gate.MEASURED_FIELDS}
        record["banks"][name] = entry
        if progress is not None:
            progress(f"{name}: {entry['status']}")
    required = [n for n in config["required_banks"] if n in record["banks"]]
    statuses = {n: record["banks"][n]["status"] for n in required}
    complete = bool(required) and len(required) == len(config["required_banks"])
    verdict = gate.verdict_of(statuses.values()) if complete else "INCOMPLETE"
    record["decision"] = {
        "required_statuses": statuses,
        "diagnostic_statuses": {n: record["banks"][n]["status"]
                                for n in config["diagnostic_banks"] if n in record["banks"]},
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
            problems.append("a bank subset (--banks) is not the declared test and may "
                            "not write the committed record")
        if RECORD.exists() and not args.overwrite:
            problems.append(f"{RECORD.name} exists: the declared test runs once "
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
    stamp_record({}, provenance)
    record = run_extrapolation(config, config_bytes=config_bytes, banks=args.banks,
                               structural=computed,
                               progress=lambda line: print(line, flush=True))
    stamped = stamp_record(record, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamped, indent=1) + "\n", encoding="utf-8")
    decision = record["decision"]
    print(f"\nverdict: {decision['verdict']}  required: {decision['required_statuses']}  "
          f"diagnostic: {decision['diagnostic_statuses']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

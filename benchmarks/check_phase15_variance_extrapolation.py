"""Verify the Phase 15 variance-extrapolation record against its preregistration.

``run_phase15_variance_extrapolation.py`` writes
``benchmarks/reference_results/phase15_variance_extrapolation.json``. This
gate trusts none of the record's summary fields.

1. **Declaration.** The record names the config by digest, and the config is
   unchanged since the run. It carries its own claim boundary, quotes the
   config's, and makes no advantage claim.
2. **Completeness.** Every declared bank has one trajectory row per prefix
   ``1..M_f``, and the window is the last three.
3. **The freeze.** Each bank's ``measured_at_execution`` equals the config's
   ``measured_before_freezing``.
4. **Re-derivation.** Every fit (the window and the diagnostic ones) is
   refitted from the recorded trajectory with a closed-form regression written
   here, not the gate's. The errors, the gain, each status, the verdict and its
   quoted consequence follow under an independent statement of the rule. Every
   deterministic check is re-derived from the values it rests on: the
   monotone energies, Weinstein's interval against the recorded spectrum, the
   dense window residuals and the final prefix against the validation record.
5. **Recomputation.** Every bank is rebuilt, and its spectrum, trajectory and
   checks must reproduce (``--no-recompute`` skips this, ``--banks``
   restricts it).
6. **Order.** Through the preregistration gate, the config's last change must
   strictly precede the record's commit.

    python benchmarks/check_phase15_variance_extrapolation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

try:
    from benchmarks import check_phase15_extrapolation_preregistration as gate
    from benchmarks import run_phase15_variance_extrapolation as producer
except ImportError:  # pragma: no cover - script execution
    import check_phase15_extrapolation_preregistration as gate
    import run_phase15_variance_extrapolation as producer

DENIALS = ("holds no prefix variance", "no fit")


def _close(a, b, rel=1e-9, abs_=1e-12) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


# ------------------------------------------------------------------ the rule, restated

def derive_fit(points, *, trust_resolved: bool = False) -> dict:
    """Simple linear regression in closed form: slope, then intercept.

    Revision 1's distinguishability floor is the resolution times the largest
    cancellation scale among the points.
    """
    from clifford_qc.subspace.second_moment import RESOLUTION

    xs = [float(p["variance"]) for p in points]
    ys = [float(p["energy"]) for p in points]
    if not trust_resolved and not all(p["resolved"] for p in points):
        return {"extrapolable": False, "reason": "a window variance is not resolved"}
    floor = RESOLUTION * max(float(p["cancellation_scale"]) for p in points)
    if max(xs) - min(xs) <= floor:
        return {"extrapolable": False,
                "reason": "the window variances differ only by rounding"}
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = (sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
             / sum((x - xbar) ** 2 for x in xs))
    intercept = ybar - slope * xbar
    rms = math.sqrt(sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys)) / len(xs))
    out = {"intercept": intercept, "slope": slope, "residual_rms": rms}
    if slope <= 0.0:
        return {**out, "extrapolable": False, "reason": "the slope is not positive"}
    return {**out, "extrapolable": True, "reason": None}


def derive_status(config: dict, fit: dict, final_error: float, checks: dict) -> str:
    if not all(checks.values()):
        return "INVALID"
    if not fit["extrapolable"]:
        return "NOT_EXTRAPOLABLE"
    ground_distance = abs(fit["extrapolated_error"])
    if ground_distance <= float(config["fit"]["improvement_factor"]) * abs(final_error):
        return "IMPROVES"
    return "NO_GAIN" if ground_distance <= abs(final_error) else "WORSENS"


def derive_verdict(statuses: list[str]) -> str:
    if not statuses:
        return "INCOMPLETE"
    if "INVALID" in statuses:
        return "INVALID"
    improves = statuses.count("IMPROVES")
    if improves == len(statuses):
        return "GO"
    return "NO_GO" if improves == 0 else "CONDITIONAL"


def _fits_agree(recorded: dict | None, derived: dict | None) -> bool:
    if recorded is None or derived is None:
        return recorded is derived
    if recorded["extrapolable"] != derived["extrapolable"]:
        return False
    if "intercept" in derived:
        return (_close(recorded.get("intercept"), derived["intercept"], rel=1e-9, abs_=1e-10)
                and _close(recorded.get("slope"), derived["slope"], rel=1e-7, abs_=1e-10))
    return True


# ------------------------------------------------------------------ checks

def declaration_problems(config: dict, record: dict, config_bytes: bytes) -> list[str]:
    problems = []
    if record.get("schema") != producer.SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}")
    if record.get("config_digest") != hashlib.sha256(config_bytes).hexdigest():
        problems.append("the config changed after the run: its digest no longer matches")
    boundary = str(record.get("claim_boundary", ""))
    if boundary == config["claim_boundary"]:
        problems.append("the record inherits the config's claim boundary instead of "
                        "stating its own")
    if any(phrase in boundary for phrase in DENIALS):
        problems.append("the record's claim boundary denies the fits it reports")
    if record.get("preregistration", {}).get(
            "config_claim_boundary_at_landing") != config["claim_boundary"]:
        problems.append("the record does not quote the config's boundary at landing")
    if record.get("quantum_advantage_claim") is not False:
        problems.append("quantum_advantage_claim must be false")
    for key in ("fit", "domain"):
        if record.get(key) != config[key]:
            problems.append(f"the record's {key} is not the config's")
    return problems


def completeness_problems(config: dict, record: dict) -> list[str]:
    declared = config["banks"]["systems"]
    if sorted(record.get("banks", {})) != sorted(declared):
        return [f"banks {sorted(record.get('banks', {}))} are not the declared {sorted(declared)}"]
    problems = []
    window = int(config["fit"]["window"])
    for name in declared:
        entry = record["banks"][name]
        size = config["measured_before_freezing"][name]["basis_size"]
        prefixes = [row["prefix"] for row in entry["trajectory"]]
        if prefixes != list(range(1, size + 1)):
            problems.append(f"{name}: the trajectory is not prefixes 1..{size}")
        if entry["window"]["prefixes"] != prefixes[-window:]:
            problems.append(f"{name}: the window is not the last {window} prefixes")
        role = "required" if name in config["required_banks"] else "diagnostic"
        if entry["role"] != role:
            problems.append(f"{name}: role {entry['role']} is not the config's {role}")
    return problems


def freeze_problems(config: dict, record: dict) -> list[str]:
    problems = []
    for name, entry in record["banks"].items():
        frozen = config["measured_before_freezing"][name]
        measured = entry["measured_at_execution"]
        for key in gate._INT_FIELDS + gate._BOOL_FIELDS:
            if measured.get(key) != frozen[key]:
                problems.append(f"{name}: {key} at execution {measured.get(key)!r} is not "
                                f"the frozen {frozen[key]!r}")
        for key in gate._FLOAT_FIELDS:
            if not _close(measured.get(key, float("nan")), frozen[key], rel=1e-9, abs_=1e-10):
                problems.append(f"{name}: {key} at execution differs from the freeze")
    return problems


def derived_checks(entry: dict, validation_root: dict, resolution: float) -> dict:
    rows, spectrum = entry["trajectory"], entry["exact_spectrum"]
    final = rows[-1]
    weinstein = []
    for row in rows:
        nearest = min(abs(value - row["energy"]) for value in spectrum)
        floor = math.sqrt(resolution * row["cancellation_scale"])
        weinstein.append(nearest <= max(row["residual_norm"], floor) * (1 + 1e-9))
    window_rows = rows[-len(entry["window"]["dense"]):] if entry["window"]["dense"] else []
    return {
        "final_prefix_matches_validation": (
            abs(final["energy"] - validation_root["energy"]) <= 1e-10
            and abs(final["variance"] - validation_root["variance"])
            <= resolution * final["cancellation_scale"]),
        "energies_monotone": all(b["energy"] <= a["energy"] + producer.MONOTONE_TOLERANCE
                                 for a, b in zip(rows, rows[1:])),
        "window_variances_match_dense": all(
            abs(row["variance"] - d["dense_residual_norm"] ** 2)
            <= resolution * row["cancellation_scale"]
            for d, row in zip(entry["window"]["dense"], window_rows)),
        "weinstein_interval": all(weinstein),
        "exact_matches_provenance": bool(entry["measured_at_execution"]["exact_matches_provenance"]),
    }


def rederivation_problems(config: dict, record: dict, validation: dict) -> list[str]:
    from clifford_qc.subspace.second_moment import RESOLUTION

    problems = []
    window = int(config["fit"]["window"])
    statuses = {}
    for name, entry in record["banks"].items():
        rows = entry["trajectory"]
        ground = entry["exact_spectrum"][0]
        fit = derive_fit(rows[-window:])
        if not _fits_agree(entry["window"]["fit"], fit):
            problems.append(f"{name}: the window fit refits to {fit}")
        final_error = rows[-1]["energy"] - ground
        if not _close(entry["final_error"], final_error, abs_=1e-12):
            problems.append(f"{name}: final_error drifted")
        if fit["extrapolable"]:
            fit["extrapolated_error"] = fit["intercept"] - ground
            if not _close(entry["extrapolated_error"], fit["extrapolated_error"],
                          rel=1e-8, abs_=1e-10):
                problems.append(f"{name}: extrapolated_error drifted")
            if not _close(entry["extrapolated_energy"], fit["intercept"], abs_=1e-10):
                problems.append(f"{name}: extrapolated_energy drifted")
        elif entry["extrapolated_energy"] is not None or entry["extrapolated_error"] is not None:
            problems.append(f"{name}: an unextrapolable window reports an extrapolated energy")
        checks = derived_checks(entry, validation["banks"][name]["roots"][0], RESOLUTION)
        if entry["deterministic_checks"] != checks:
            problems.append(f"{name}: recorded deterministic checks are not what their "
                            "values give")
        status = derive_status(config, fit, final_error, checks)
        statuses[name] = status
        if entry["status"] != status:
            problems.append(f"{name}: recorded status {entry['status']} is not the rule's "
                            f"{status}")
        diagnostics = entry["diagnostics"]
        resolved = [r for r in rows if r["resolved"]]
        expected = {
            "window_2": derive_fit(rows[-2:]),
            "window_4": derive_fit(rows[-4:]) if len(rows) >= 4 else None,
            "all_resolved": derive_fit(resolved, trust_resolved=True) if len(resolved) >= 2 else None,
        }
        for key, value in expected.items():
            if not _fits_agree(diagnostics[key], value):
                problems.append(f"{name}: diagnostic {key} refits differently")
        final = rows[-1]
        first = entry["exact_spectrum"][1]
        temple = (final["energy"] - final["variance"] / (first - final["energy"])
                  if final["energy"] < first else None)
        if not _close(diagnostics["temple_lower_bound"], temple, abs_=1e-10):
            problems.append(f"{name}: the Temple bound drifted")
        reaches = (entry["units"] == "hartree" and fit["extrapolable"]
                   and abs(fit["extrapolated_error"]) <= producer.CHEMICAL_ACCURACY
                   and abs(final_error) > producer.CHEMICAL_ACCURACY)
        if diagnostics["reaches_chemical_accuracy_where_final_does_not"] != reaches:
            problems.append(f"{name}: the chemical-accuracy diagnostic is wrong")
    required = [n for n in config["required_banks"] if n in statuses]
    verdict = derive_verdict([statuses[n] for n in required])
    decision = record.get("decision", {})
    if decision.get("required_statuses") != {n: statuses[n] for n in required}:
        problems.append("the summary's required statuses are not the banks' own")
    if decision.get("diagnostic_statuses") != {
            n: statuses[n] for n in config["diagnostic_banks"] if n in statuses}:
        problems.append("the summary's diagnostic statuses are not the banks' own")
    if decision.get("verdict") != verdict:
        problems.append(f"recorded verdict {decision.get('verdict')} is not the rule's {verdict}")
    if decision.get("consequence") != config["consequences"].get(verdict):
        problems.append("the recorded consequence does not quote the config's for the verdict")
    return problems


def recompute_problems(config: dict, record: dict, validation: dict, *, banks=None,
                       inputs=None) -> list[str]:
    """Every bank rebuilt: spectrum, trajectory and checks must reproduce."""
    inputs = producer.preflight_gate.bank_inputs if inputs is None else inputs
    problems = []
    for name in (record["banks"] if banks is None else banks):
        entry = record["banks"][name]
        model, selected = inputs(name)
        got = producer.evaluate_bank(config, name, model, selected,
                                     config["measured_before_freezing"][name],
                                     validation["banks"][name]["roots"][0])
        if len(got["exact_spectrum"]) != len(entry["exact_spectrum"]) or not all(
                _close(a, b, abs_=1e-9) for a, b in zip(got["exact_spectrum"],
                                                        entry["exact_spectrum"])):
            problems.append(f"{name}: the exact spectrum recomputes differently")
        for mine, theirs in zip(got["trajectory"], entry["trajectory"]):
            scale = theirs["cancellation_scale"]
            if not (_close(mine["energy"], theirs["energy"], rel=1e-10, abs_=1e-12)
                    and abs(mine["variance"] - theirs["variance"]) <= 1e-12 * scale
                    and mine["resolved"] == theirs["resolved"]):
                problems.append(f"{name}: prefix {theirs['prefix']} recomputes to "
                                f"E={mine['energy']}, var={mine['variance']}")
                break
        if got["deterministic_checks"] != entry["deterministic_checks"]:
            problems.append(f"{name}: the checks recompute to {got['deterministic_checks']}")
        if got["status"] != entry["status"]:
            problems.append(f"{name}: the status recomputes to {got['status']}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=producer.RECORD)
    parser.add_argument("--no-recompute", action="store_true",
                        help="re-derive everything from the record without rebuilding a bank")
    parser.add_argument("--banks", nargs="*", default=None, help="recompute only these banks")
    args = parser.parse_args(argv)
    if not args.record.exists():
        print(f"FAIL missing record {args.record}")
        return 1
    config_bytes = gate.CONFIG.read_bytes()
    config = gate.load_config()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    validation = json.loads(gate.VALIDATION.read_text(encoding="utf-8"))
    problems = declaration_problems(config, record, config_bytes)
    problems += completeness_problems(config, record)
    if not problems:
        problems += freeze_problems(config, record)
        problems += rederivation_problems(config, record, validation)
        if not args.no_recompute:
            problems += recompute_problems(config, record, validation, banks=args.banks)
    if record.get("provenance", {}).get("git_dirty") is not False:
        problems.append("the record's provenance does not show a clean tree")
    notes: list[str] = []
    if args.record.resolve() == producer.RECORD.resolve():
        problems += gate.commit_order_problems(notes)
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    decision = record["decision"]
    print(f"OK Phase 15 extrapolation record re-derives under the frozen rule: verdict "
          f"{decision['verdict']}, required {decision['required_statuses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

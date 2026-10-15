"""Verify the folded-spectrum validation record.

``run_folded_spectrum_validation.py`` writes
``benchmarks/reference_results/folded_spectrum_validation.json``. This gate:

1. requires the record's preflight digest to match the committed preflight
   record, its banks to be exactly the ones that record licenses, and its
   evidence labels, tolerances and claim boundary to be the producer's;
2. restates every derived field from the recorded values: each root's
   residual norm, resolution flags and both intervals from its value and
   scale, the sector levels each interval holds, the dense agreement, the
   refinement inequality, the monotone and converged flags from the trace,
   every check, status, reading and the summary;
3. rebuilds every bank and compares it (``--banks`` restricts this,
   ``--no-recompute`` skips it). Counts and level indices must match exactly.
   Energies and levels must match to ``1e-9`` relative. A folded value or
   variance must match to its own resolution floor, since each is a difference
   of large terms whose rounding is the BLAS kernel's.

    python benchmarks/check_folded_spectrum_validation.py
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
    from benchmarks import check_phase15_certifying_screen as provenance_rules
    from benchmarks import check_phase15_preregistration as gate
    from benchmarks import run_folded_spectrum_validation as producer
    from benchmarks import run_second_moment_validation as second_moment
except ImportError:  # pragma: no cover - script execution
    import check_phase15_certifying_screen as provenance_rules
    import check_phase15_preregistration as gate
    import run_folded_spectrum_validation as producer
    import run_second_moment_validation as second_moment

RESOLUTION = 1e-12
TARGET_TOLERANCE = 1e-12
ROW_KEYS = frozenset({
    "root", "target", "folded_value", "energy", "second_moment", "variance",
    "residual_norm", "folded_scale", "variance_scale", "resolved", "variance_resolved",
    "target_interval", "weinstein_interval", "dense_energy", "dense_folded_value",
    "dense_variance", "target_levels", "weinstein_levels", "isolated_level", "matches_dense"})
REFINED_KEYS = ROW_KEYS | {"ritz_root", "ritz_variance", "ritz_residual_norm",
                           "ritz_variance_resolved", "refined_not_above_ritz"}


def _close(a, b, rel=1e-12, abs_=1e-15) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def _inside(levels, interval) -> list[int]:
    lo, hi = interval
    return [i for i, level in enumerate(levels) if lo <= level <= hi]


def declaration_problems(record: dict, preflight: dict, preflight_bytes: bytes) -> list[str]:
    problems = []
    for key, value in {
        "schema": producer.SCHEMA, "claim_boundary": producer.CLAIM_BOUNDARY,
        "quantum_advantage_claim": False, "evidence_schema": "clifford_qc.evidence.v1",
        "evidence": producer.EVIDENCE,
        "tolerances": {"variance_resolution": RESOLUTION, "energy": producer.ENERGY_TOLERANCE,
                       "level": producer.LEVEL_TOLERANCE,
                       "sector_leakage": producer.SECTOR_LEAKAGE_TOLERANCE},
    }.items():
        if record.get(key) != value:
            problems.append(f"record {key} differs from the producer's")
    if record.get("preflight", {}).get("sha256") != hashlib.sha256(preflight_bytes).hexdigest():
        problems.append("the preflight record changed after this validation")
    licensed = second_moment.licensed_banks(preflight)
    if sorted(record.get("banks", {})) != sorted(licensed):
        problems.append(f"banks {sorted(record.get('banks', {}))} are not the licensed "
                        f"{sorted(licensed)}")
    problems += provenance_rules.provenance_problems(record.get("provenance"))
    return problems


def _folded_identity(target, folded, energy, variance, scale) -> bool:
    """``mu = sigma^2 + (E - w)^2``, which every folded value obeys, to its floor."""
    return abs(folded - (variance + (energy - target) ** 2)) <= RESOLUTION * scale


def row_problems(row: dict, levels, label: str, *, refined: bool) -> list[str]:
    """One folded root's derived fields, restated from its values."""
    problems = []
    if set(row) != (REFINED_KEYS if refined else ROW_KEYS):
        return [f"{label}: fields differ from the record schema"]
    target_bound = math.sqrt(max(row["folded_value"], 0.0) + RESOLUTION * row["folded_scale"])
    weinstein_bound = math.sqrt(max(row["variance"], 0.0) + RESOLUTION * row["variance_scale"])
    target_interval = [row["target"] - target_bound, row["target"] + target_bound]
    weinstein_interval = [row["energy"] - weinstein_bound, row["energy"] + weinstein_bound]
    weinstein_levels = _inside(levels, weinstein_interval)
    expected = {
        "residual_norm": math.sqrt(max(row["variance"], 0.0)),
        "resolved": row["folded_value"] > RESOLUTION * row["folded_scale"],
        "variance_resolved": row["variance"] > RESOLUTION * row["variance_scale"],
        "target_levels": _inside(levels, target_interval),
        "weinstein_levels": weinstein_levels,
        "isolated_level": weinstein_levels[0] if len(weinstein_levels) == 1 else None,
        "matches_dense": bool(
            abs(row["energy"] - row["dense_energy"])
            <= producer.ENERGY_TOLERANCE * max(1.0, abs(row["energy"]))
            and abs(row["folded_value"] - row["dense_folded_value"])
            <= RESOLUTION * row["folded_scale"]
            and abs(row["variance"] - row["dense_variance"]) <= RESOLUTION * row["variance_scale"]),
    }
    if not _close(row["variance"], row["second_moment"] - row["energy"] ** 2,
                  rel=1e-12, abs_=4 * math.ulp(max(1.0, row["second_moment"]))):
        problems.append(f"{label}: variance is not <H^2> - <H>^2")
    if not _folded_identity(row["target"], row["folded_value"], row["energy"], row["variance"],
                            row["folded_scale"]):
        problems.append(f"{label}: folded value is not variance + (energy - target)^2")
    for key, value in expected.items():
        if isinstance(value, float):
            ok = isinstance(row[key], float) and _close(row[key], value)
        else:
            ok = row[key] == value and type(row[key]) is type(value)
        if not ok:
            problems.append(f"{label}: {key} does not derive")
    for key, interval in (("target_interval", target_interval),
                          ("weinstein_interval", weinstein_interval)):
        if len(row[key]) != 2 or not all(_close(a, b) for a, b in zip(row[key], interval)):
            problems.append(f"{label}: {key} does not derive")
    if refined:
        if not _close(row["ritz_residual_norm"], math.sqrt(max(row["ritz_variance"], 0.0))):
            problems.append(f"{label}: ritz_residual_norm does not derive")
        if row["refined_not_above_ritz"] is not bool(
                row["folded_value"] <= row["ritz_variance"] + RESOLUTION * row["folded_scale"]):
            problems.append(f"{label}: refined_not_above_ritz does not derive")
    return problems


def minimization_problems(block: dict, levels, label: str) -> tuple[list[str], bool]:
    problems = row_problems(block["root"], levels, f"{label} minimized", refined=False)
    trace = block["trace"]
    if block["iterations"] != len(trace) or not trace:
        return problems + [f"{label}: iterations differ from the trace"], False
    last, root = trace[-1], block["root"]
    if [last[0], last[1], last[2], last[3], last[4]] != [
            root["target"], root["folded_value"], root["energy"], root["variance"],
            root["folded_scale"]]:
        problems.append(f"{label}: the trace does not end at the reported root")
    if trace[0][0] != block["start"]:
        problems.append(f"{label}: the trace does not start at the start")
    for i, step in enumerate(trace):
        if len(step) != 5 or not _folded_identity(*step):
            problems.append(f"{label}: trace step {i} is not variance + (energy - target)^2")
    monotone = all(b[1] <= a[1] + RESOLUTION * b[4] for a, b in zip(trace, trace[1:]))
    converged = abs(last[2] - last[0]) <= TARGET_TOLERANCE * max(1.0, abs(last[0]))
    for a, b in zip(trace, trace[1:]):
        if b[0] != a[2]:
            problems.append(f"{label}: a step's target is not the previous energy")
            break
    if block["monotone"] is not monotone or block["converged"] is not converged:
        problems.append(f"{label}: monotone or converged does not derive")
    return problems, monotone and converged


def derived_problems(record: dict) -> list[str]:
    problems = []
    for name, entry in record["banks"].items():
        sector = entry["sector"]
        levels = sector["levels"]
        if any(b - a <= producer.LEVEL_TOLERANCE * max(1.0, abs(b))
               for a, b in zip(levels, levels[1:])):
            problems.append(f"{name}: sector levels are not distinct and ascending")
        roots = entry["roots"]
        if [row["ritz_root"] for row in roots] != list(range(len(roots))):
            problems.append(f"{name}: refined roots are not one per Ritz root")
        for row in roots:
            problems += row_problems(row, levels, f"{name} root {row['ritz_root']}",
                                     refined=True)
        issues, settled = minimization_problems(entry["minimization"], levels, name)
        problems += issues
        if roots and entry["minimization"]["start"] != roots[0]["target"]:
            problems.append(f"{name}: minimization does not start at the ground Ritz value")
        every = roots + [entry["minimization"]["root"]]
        checks = {
            "sector_premise": bool(
                sector["reference_image_sector_leakage"] <= producer.SECTOR_LEAKAGE_TOLERANCE
                and sector["hamiltonian_sector_leakage"] <= producer.SECTOR_LEAKAGE_TOLERANCE),
            "roots_match_dense": all(row["matches_dense"] for row in every),
            "refined_not_above_ritz": all(row["refined_not_above_ritz"] for row in roots),
            "target_intervals_contain_a_level": all(row["target_levels"] for row in every),
            "weinstein_intervals_contain_a_level": all(row["weinstein_levels"] for row in every),
            "minimization_monotone_and_converged": settled,
        }
        if entry["checks"] != checks:
            problems.append(f"{name}: recorded checks are not what their values give")
        status = "VALIDATED" if all(checks.values()) else "FAILED"
        if entry["status"] != status:
            problems.append(f"{name}: recorded status {entry['status']} is not {status}")
        ground, final = roots[0], entry["minimization"]["root"]
        reading = {
            "ground_ritz_residual_norm": ground["ritz_residual_norm"],
            "ground_refined_target_bound": math.sqrt(max(ground["folded_value"], 0.0)),
            "minimized_residual_norm": final["residual_norm"],
            "minimized_energy": final["energy"],
            "isolated_levels": sorted({row["isolated_level"] for row in roots
                                       if row["isolated_level"] is not None}),
        }
        if entry["reading"] != reading:
            problems.append(f"{name}: the reading does not derive")
    statuses = {name: entry["status"] for name, entry in record["banks"].items()}
    summary = record["summary"]
    if summary["bank_statuses"] != statuses:
        problems.append("the summary's statuses are not the banks' own")
    licensed = record["preflight"]["eligible_banks"]
    everything = (sorted(statuses) == sorted(licensed)
                  and all(s == "VALIDATED" for s in statuses.values()))
    if summary["all_validated"] is not everything:
        problems.append("all_validated contradicts the bank statuses")
    return problems


def _row_rebuild(mine: dict, theirs: dict, label: str) -> list[str]:
    """One rebuilt root against the recorded one, each value to its own floor."""
    problems = []
    folded_floor = RESOLUTION * theirs["folded_scale"]
    variance_floor = RESOLUTION * theirs["variance_scale"]
    for key in ("target", "energy", "second_moment", "dense_energy"):
        if not _close(mine[key], theirs[key], rel=1e-9, abs_=1e-12):
            problems.append(f"{label}.{key} does not rebuild")
    for key, floor in (("folded_value", folded_floor), ("dense_folded_value", folded_floor),
                       ("variance", variance_floor), ("dense_variance", variance_floor)):
        if not abs(mine[key] - theirs[key]) <= floor + 1e-9 * abs(theirs[key]):
            problems.append(f"{label}.{key} does not rebuild within its floor")
    for key in ("folded_scale", "variance_scale"):
        if not _close(mine[key], theirs[key], rel=1e-6):
            problems.append(f"{label}.{key} does not rebuild")
    for key in ("target_levels", "weinstein_levels", "isolated_level", "matches_dense",
                "refined_not_above_ritz", "ritz_root"):
        if key in theirs and mine[key] != theirs[key]:
            problems.append(f"{label}.{key} does not rebuild")
    if "ritz_variance" in theirs and not abs(mine["ritz_variance"] - theirs["ritz_variance"]) <= (
            variance_floor + 1e-9 * abs(theirs["ritz_variance"])):
        problems.append(f"{label}.ritz_variance does not rebuild within its floor")
    return problems


def _trace_rebuild(mine: list, theirs: list, label: str) -> list[str]:
    """Each recorded step re-solved at its own target, each value to its floor.

    The trace carries no variance scale, so the variance takes the folded
    scale, which bounds it, as its floor.
    """
    problems = []
    for i, (got, step) in enumerate(zip(mine, theirs)):
        target, folded, energy, variance, scale = step
        floor = RESOLUTION * scale
        if not (got[0] == target
                and _close(got[2], energy, rel=1e-9, abs_=1e-12)
                and abs(got[1] - folded) <= floor + 1e-9 * abs(folded)
                and abs(got[3] - variance) <= floor + 1e-9 * abs(variance)
                and _close(got[4], scale, rel=1e-6)):
            problems.append(f"{label}: trace step {i} does not rebuild at its target")
    return problems


def recompute_problems(record: dict, *, banks=None, inputs=None) -> list[str]:
    """Every bank rebuilt and compared."""
    inputs = gate.bank_inputs if inputs is None else inputs
    problems = []
    for name in (record["banks"] if banks is None else banks):
        entry = record["banks"][name]
        model, selected = inputs(name)
        trace = entry["minimization"]["trace"]
        got = json.loads(json.dumps(producer.validate_bank(
            model, selected, replay=[step[0] for step in trace]), allow_nan=False))
        problems += _trace_rebuild(got.pop("replay"), trace, f"{name} minimization")
        for key in ("n_qubits", "basis_size", "effective_rank", "checks", "status"):
            if got[key] != entry[key]:
                problems.append(f"{name}.{key} does not rebuild")
        if got["sector"]["dimension"] != entry["sector"]["dimension"] or len(
                got["sector"]["levels"]) != len(entry["sector"]["levels"]) or not all(
                _close(a, b, rel=1e-9, abs_=1e-12)
                for a, b in zip(got["sector"]["levels"], entry["sector"]["levels"])):
            problems.append(f"{name}: sector levels do not rebuild")
        if len(got["roots"]) != len(entry["roots"]):
            problems.append(f"{name}: {len(got['roots'])} roots rebuilt, "
                            f"{len(entry['roots'])} recorded")
            continue
        for mine, theirs in zip(got["roots"], entry["roots"]):
            problems += _row_rebuild(mine, theirs, f"{name} root {theirs['ritz_root']}")
        mine, theirs = got["minimization"], entry["minimization"]
        problems += _row_rebuild(mine["root"], theirs["root"], f"{name} minimized")
        if (mine["converged"], mine["monotone"]) != (theirs["converged"], theirs["monotone"]):
            problems.append(f"{name}: minimization flags do not rebuild")
        # The step count near a fixed point moves with the last bits of each solve.
        if abs(mine["iterations"] - theirs["iterations"]) > 3:
            problems.append(f"{name}: minimization iterations do not rebuild")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=producer.RECORD)
    parser.add_argument("--banks", nargs="*", default=None, help="recompute only these banks")
    parser.add_argument("--no-recompute", action="store_true",
                        help="restate checks and summary without rebuilding any bank")
    args = parser.parse_args(argv)
    if not args.record.exists():
        print(f"FAIL missing record {args.record}")
        return 1
    try:
        record = provenance_rules.load_record(args.record)
        preflight_bytes = producer.PREFLIGHT.read_bytes()
        preflight = json.loads(preflight_bytes)
        problems = declaration_problems(record, preflight, preflight_bytes)
        if not problems:
            problems += derived_problems(record)
            if not args.no_recompute:
                problems += recompute_problems(record, banks=args.banks)
        if not record["summary"]["all_validated"]:
            problems.append("the record does not validate every licensed bank")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"FAIL malformed folded-spectrum record: {exc}")
        return 1
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    readings = ", ".join(
        f"{name} {entry['reading']['ground_ritz_residual_norm']:.3e}->"
        f"{entry['reading']['minimized_residual_norm']:.3e}"
        for name, entry in record["banks"].items())
    print(f"OK folded-spectrum roots validate on every licensed bank; ground residual "
          f"Ritz->minimized: {readings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

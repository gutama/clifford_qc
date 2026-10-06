"""Run Q18-S2 once: the oracle allocation screen on certifying Hubbard 2x2 bases.

See ``benchmarks/PHASE15_CERTIFYING_SCREEN_DECLARATION.md``. For every prefix
``M = 9 ...`` of the two committed Hubbard 2x2 A-CASE trajectories, the
producer solves the ground Ritz root, reads its exact residual norm from
``SecondMomentBank``, decides whether its Weinstein interval certifies the
sector ground state, and prices ``sigma`` against the energy under both
declared QWC protocols with Q18's estimator and Q18-S1's Richardson validator.
The verdict reads only the Neyman ratios of certifying prefixes.

It refuses before forming a second-moment row unless the declaration gate
passes, the declaration is committed, no record exists, the working tree is
clean and the environment is one the committed records declare.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python benchmarks/run_phase15_certifying_screen.py
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
if str(ROOT) not in sys.path:  # pragma: no cover - script execution
    sys.path.insert(0, str(ROOT))

from benchmarks import check_phase15_certifying_screen_declaration as declaration
from benchmarks import phase15_certifying_screen as screen

RECORD = declaration.RECORD
SCHEMA = declaration.ARTIFACT_PATHS["schema"]
PREDECESSOR = HERE / "reference_results" / "phase15_residual_successor.json"
CLAIM_BOUNDARY = (
    "This record reports the one Q18-S2 screen its declaration froze. On every prefix "
    "from M = 9 of the two committed Hubbard 2x2 A-CASE trajectories it records the "
    "exact ground Ritz energy and residual norm, whether the Weinstein interval "
    "certifies the sector ground state, and, where the variance is resolved, Q18's "
    "uniform and Neyman cost ratios of the residual norm against the energy under both "
    "declared QWC protocols. The verdict reads only the Neyman ratios of certifying "
    "prefixes. Costs are first-order and asymptotic, computed from exact reference "
    "variances; the system was selected post hoc from Q18-S1's oracle diagnostic. No "
    "pilot is sampled, and no finite-shot estimator, coverage, device time or quantum "
    "advantage follows from this record.")
LINEAGE_PROTOCOL_FIELDS = (
    "energy_settings", "residual_settings", "energy_partitioned_words",
    "residual_partitioned_words", "energy_variance_one_shot", "residual_variance_one_shot",
    "energy_neyman_sum", "residual_neyman_sum", "cost_ratio", "neyman_ratio")
# Neyman sums add square roots of rounding-level setting variances, so they are
# rebuilt to the tolerance Q18's own checker gives them.
NEYMAN_FIELDS = ("energy_neyman_sum", "residual_neyman_sum", "neyman_ratio")


def _close(a, b, rel: float = declaration.FLOAT_TOLERANCE) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-12)


def lineage_entry(entry: dict, predecessor: dict) -> dict:
    """``M = 9`` against Q18-S1's committed entry for the same frozen bank."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    checks = {
        "ground_energy": _close(entry["ground_energy"],
                                predecessor["estimator"]["ground_energy"]),
        "variance": _close(entry["variance"], predecessor["estimator"]["variance"]),
        "universes": all(entry.get("universes", {}).get(key) == predecessor["universes"][key]
                         for key in ("raw_sh_word_universe", "raw_combined_word_universe",
                                     "measured_sh_words", "measured_combined_words")),
    }
    for protocol, theirs in predecessor["protocols"].items():
        mine = entry.get("protocols", {}).get(protocol)
        checks[f"{protocol}_prices"] = mine is not None and all(
            _close(mine[key], theirs[key],
                   declaration.NEYMAN_TOLERANCE if key in NEYMAN_FIELDS
                   else declaration.FLOAT_TOLERANCE)
            for key in LINEAGE_PROTOCOL_FIELDS) and all(
            len(mine[f"{side}_setting_variances"]) == len(theirs[f"{side}_setting_variances"])
            and all(abs(a - b) <= 1e-12 + declaration.FLOAT_TOLERANCE * abs(b)
                    for a, b in zip(mine[f"{side}_setting_variances"],
                                    theirs[f"{side}_setting_variances"]))
            for side in ("energy", "residual"))
    floor = RESOLUTION * float(entry["cancellation_scale"])
    for direction, theirs in predecessor["finite_differences"].items():
        mine = entry.get("finite_differences", {}).get(direction)
        checks[f"{direction}_endpoints"] = mine is not None and _close(
            mine["analytic"], theirs["analytic"]) and all(
            abs(a[key] - b[key]) <= floor and a["step"] == b["step"]
            for a, b in zip(mine["evaluations"], theirs["evaluations"])
            for key in ("plus", "minus")) and len(mine["evaluations"]) == len(
            theirs["evaluations"])
    return checks


def decide(config: dict, trajectories: dict, lineage: dict) -> dict:
    entries, statuses, domain = [], {}, []
    for method, block in trajectories.items():
        for size, entry in block["prefixes"].items():
            key = f"{method}/{size}"
            statuses[key] = entry["status"]
            if entry["in_domain"]:
                domain.append(key)
            entries.append(entry)
    verdict = "INVALID" if not lineage["passes"] else screen.verdict_of(entries)
    return {"domain": domain, "statuses": statuses, "verdict": verdict,
            "consequence": config["consequences"][verdict],
            "rule": "frozen in the config; see decision_rule"}


def run_screen(config: dict, *, config_bytes: bytes, model=None, inputs=None,
               predecessor: dict | None = None, progress=None) -> dict:
    """The declared screen as an unstamped record. Overrides are for tests."""
    started = time.perf_counter()
    model = declaration.system_model() if model is None else model
    inputs = declaration.trajectory_inputs(config, model) if inputs is None else inputs
    if predecessor is None:
        predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))["banks"][
            declaration.SYSTEM]
    spectrum = screen.sector_spectrum(model)
    premises = screen.sector_premises(model, [gens for gens, _ in inputs.values()])
    grouping = config["grouping"]
    protocols = (grouping["declared_protocol"], grouping["alternative_protocol"])
    trajectories = {}
    for method, (generators, pool) in inputs.items():
        spec = config["trajectories"]["rows"][method]
        if progress is not None:
            progress(f"{method}: pool {pool}, M <= {len(generators)}")
        say = (lambda line, m=method: progress(f"  {m} {line}")) if progress else None
        trajectories[method] = {
            "candidate_family": spec["candidate_family"],
            "pool_size": pool,
            "labels": spec["labels"],
            "prefixes": screen.evaluate_trajectory(
                config, model, generators, spec["energy_history"], spectrum,
                protocols=protocols, prefixes=spec["prefixes"], progress=say),
        }
    lineage = {method: lineage_entry(block["prefixes"][str(declaration.FROZEN_BANK_SIZE)],
                                     predecessor)
               for method, block in trajectories.items()}
    lineage["passes"] = all(all(checks.values()) for checks in lineage.values())
    record = {
        "schema": SCHEMA,
        "config_path": str(declaration.CONFIG.relative_to(ROOT)),
        "config_digest": hashlib.sha256(config_bytes).hexdigest(),
        "claim_boundary": CLAIM_BOUNDARY,
        "quantum_advantage_claim": False,
        "question": config["question"],
        "evidence": config["evidence"],
        "statistic": config["statistic"],
        "domain": config["domain"],
        "grouping": config["grouping"],
        "derivative_validation": config["derivative_validation"],
        "allocation_diagnostic": config["allocation_diagnostic"],
        "spectral_reference": spectrum,
        "premises": premises,
        "trajectories": trajectories,
        "lineage": lineage,
        "decision": decide(config, trajectories, lineage),
        "declaration": {
            "gate": "benchmarks/check_phase15_certifying_screen_declaration.py",
            "design_status": config["evidence"]["design_status"],
            "system_selection": config["evidence"]["system_selection"],
            "config_claim_boundary_at_landing": config["claim_boundary"],
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    return record


def refusals(args) -> list[str]:
    if args.out.resolve() != RECORD.resolve():
        return ["Q18-S2 writes only its declared record path"]
    if RECORD.exists():
        return ["Q18-S2 runs once; use its checker to rebuild"]
    return []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RECORD)
    args = parser.parse_args(argv)
    problems = refusals(args)
    notes: list[str] = []
    if not problems:
        config_bytes = declaration.CONFIG.read_bytes()
        config = declaration.load_config()
        problems += declaration.static_problems(config)
        problems += declaration.commit_order_problems(notes, require_config=True)
    if not problems:
        problems += declaration.structural_problems(config, notes)
    from clifford_qc.reproducibility import execution_provenance, stamp_record

    provenance = execution_provenance()
    if provenance["git_dirty"]:
        problems.append("commit the code before running the screen: the tree is dirty")
    for problem in problems:
        print(f"REFUSE {problem}")
    if problems:
        return 1
    for note in notes:
        print(note)
    stamp_record({}, provenance)  # environment guard before any second-moment row
    record = run_screen(config, config_bytes=config_bytes,
                        progress=lambda line: print(line, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamp_record(record, provenance), indent=1,
                                   allow_nan=False) + "\n", encoding="utf-8")
    decision = record["decision"]
    print(f"\nverdict: {decision['verdict']}  domain: {decision['domain']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

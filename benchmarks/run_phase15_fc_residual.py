"""Run Q18-S4 once: the measured residual's cost under fully commuting grouping.

Re-asks Q18-S1's uniform question with fully commuting settings, exactly as
``benchmarks/configs/phase15_fc_residual.json`` froze it, on the four 8-qubit
licensed banks. See ``benchmarks/PHASE15_FC_RESIDUAL_DECLARATION.md``.

The producer refuses before it forms a second-moment row unless the
declaration gate passes, including its recomputation of every frozen number;
the declaration is committed; no record exists yet; the run writes the
declared record path; the working tree is clean; and the environment is one
the committed records declare.

For each bank it builds Q18-S1's rows and functionals, validates the residual
functional with Q18-S1's Richardson rule, and checks both functionals against
Q18-S1's record. Under each fully commuting protocol it partitions both
measured universes, sums the exact per-setting variances, checks every one of
them against the reference statevector, and reads

    R = (G_U * V_sigma2) / (4 sigma^2 * G_SH * V_E)

with Q18's ``cost_ratio``. The status and verdict come off Q18's ladder and
combination rule, unchanged.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python benchmarks/run_phase15_fc_residual.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

from benchmarks import check_phase15_fc_residual_declaration as declaration
from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import check_phase15_preregistration as lineage
from benchmarks import check_phase15_residual_successor_declaration as q18s1
from benchmarks.phase15_fc_residual import evaluate_bank

RECORD = declaration.RECORD
SCHEMA = declaration.ARTIFACT_PATHS["schema"]
CLAIM_BOUNDARY = (
    "This post-hoc Q18-S4 record prices, on the four 8-qubit licensed banks, the shots "
    "that estimate the ground Ritz root's residual norm against the shots that estimate "
    "its energy, at one matched standard error, under single-assignment fully commuting "
    "settings with uniform shots: the shipped greedy colouring and the isotropic cover, "
    "each at block size n. The estimator, its linearization, the derivative validator, "
    "the statistic and the tenfold bar are Q18-S1's. Every Q18 and Q18-S1 ratio was seen "
    "before the declaration, so commit order protects execution identity, not inference. "
    "The ratios are first-order and asymptotic, read from exact variances on each bank's "
    "determinant reference, a stabilizer state. They price shots only: not the "
    "diagonalizing circuits fully commuting settings need, not device time, not "
    "finite-sample intervals, not another allocation, and not H2O or any bank outside "
    "the four.")


def run(config: dict, *, config_bytes: bytes, inputs=None, validation: dict | None = None,
        predecessor: dict | None = None, structural: dict | None = None,
        progress=None) -> dict:
    """The declared run as a record (unstamped). The keywords are for tests."""
    inputs = lineage.bank_inputs if inputs is None else inputs
    if validation is None:
        validation = json.loads(q18.VALIDATION.read_text(encoding="utf-8"))
    if predecessor is None:
        predecessor = json.loads(q18s1.RECORD.read_text(encoding="utf-8"))
    started = time.perf_counter()
    record = {
        "schema": SCHEMA,
        "config_path": str(declaration.CONFIG.relative_to(ROOT)),
        "config_digest": hashlib.sha256(config_bytes).hexdigest(),
        "claim_boundary": CLAIM_BOUNDARY,
        "quantum_advantage_claim": False,
        "evidence": config["evidence"],
        "statistic": config["statistic"],
        "grouping": config["grouping"],
        "derivative_validation": config["derivative_validation"],
        "excluded": config["banks"]["excluded"],
        "successor": config["successor"],
        "declaration": {"gate": "benchmarks/check_phase15_fc_residual_declaration.py",
                        "design_status": "post_hoc",
                        "config_claim_boundary_at_landing": config["claim_boundary"]},
        "banks": {},
    }
    for name in config["banks"]["systems"]:
        model, selected = inputs(name)
        frozen = config["measured_before_freezing"][name]
        if progress is not None:
            progress(f"{name}: n={model.n} M={len(selected)}")
        entry = evaluate_bank(config, name, model, selected, frozen,
                              validation["banks"][name]["roots"][0],
                              predecessor["banks"][name], progress=progress)
        entry["measured_at_execution"] = (structural or {}).get(name, frozen)
        entry["q18s1_uniform_qwc"] = {
            protocol: {"role": price["role"], "cost_ratio": price["cost_ratio"]}
            for protocol, price in predecessor["banks"][name]["protocols"].items()}
        record["banks"][name] = entry
        if progress is not None:
            progress(f"{name}: {entry['status']} in {entry['seconds']} s")
    statuses = {name: entry["status"] for name, entry in record["banks"].items()}
    verdict = q18.verdict_of(statuses.values())
    record["decision"] = {
        "statuses": statuses,
        "verdict": verdict,
        "consequence": config["consequences"][verdict],
        "rule": "Q18-S1's ladder and combination rule, frozen in the config",
    }
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    return record


def refusals(args) -> list[str]:
    if args.out.resolve() != RECORD.resolve():
        return ["Q18-S4 writes only its declared record path"]
    if RECORD.exists():
        return [f"{RECORD.name} exists: Q18-S4 runs once; use its checker to rebuild"]
    return []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RECORD,
                        help="record path; only the declared one is accepted")
    args = parser.parse_args(argv)
    problems = refusals(args)
    notes: list[str] = []
    structural: dict = {}
    if not problems:
        config_bytes = declaration.CONFIG.read_bytes()
        config = declaration.load_config()
        problems += [f"declaration gate: {p}" for p in declaration.static_problems(config)]
        problems += declaration.commit_order_problems(notes, require_config=True)
    if not problems:
        problems += [f"declaration gate: {p}"
                     for p in declaration.structural_problems(config, notes, structural)]
    from clifford_qc.reproducibility import execution_provenance, stamp_record

    provenance = execution_provenance()
    if provenance["git_dirty"]:
        problems.append("the working tree is dirty; commit first so the record's "
                        "provenance names the code that produced it")
    for problem in problems:
        print(f"REFUSE {problem}")
    if problems:
        return 1
    for note in notes:
        print(note)
    stamp_record({}, provenance)  # environment guard before any second-moment row
    record = run(config, config_bytes=config_bytes, structural=structural,
                 progress=lambda line: print(line, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamp_record(record, provenance), indent=1,
                                   allow_nan=False) + "\n", encoding="utf-8")
    decision = record["decision"]
    print(f"\npost-hoc fully commuting verdict: {decision['verdict']}  "
          f"statuses: {decision['statuses']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run Q18-S1 once from its separately committed post-hoc declaration.

The original INVALID record is immutable. The uniform-cost decision uses a
Richardson derivative check. Integer oracle allocation is diagnostic only;
the charged pilot is not sampled and licenses no finite-shot estimator.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from benchmarks import check_phase15_residual_successor_declaration as declaration
from benchmarks import run_phase15_measured_residual_preflight as original
from benchmarks.phase15_residual_successor import allocation_diagnostic, derivative_check

RECORD = declaration.RECORD
SCHEMA = "clifford_qc.phase15_residual_successor.v1"
CLAIM_BOUNDARY = (
    "This post-hoc Q18-S1 record re-executes the original five-bank uniform QWC cost "
    "question with fixed multi-step Richardson derivative validation. The original "
    "Q18 record remains INVALID. Its ratios and Hubbard step sweep were inspected "
    "before this design was declared; commit order does not make this a preregistered "
    "inference. Costs are first-order and asymptotic, computed from exact reference "
    "variances. Integer variance-optimal allocation charges an unperformed pilot and "
    "is an oracle diagnostic only. No finite-shot estimator, coverage, device time "
    "or quantum advantage is licensed by the allocation diagnostic."
)


def evaluate_bank(config, name, model, selected, frozen, validation_root, *, progress=None):
    entry = original.evaluate_bank(
        config, name, model, selected, frozen, validation_root, progress=progress,
        derivative_check=functools.partial(derivative_check, rule=config["derivative_validation"]),
        protocol_pricer=functools.partial(original.price_protocol, include_setting_variances=True))
    for price in entry["protocols"].values():
        if price["variances_nonnegative"]:
            price["allocation_diagnostic"] = allocation_diagnostic(
                price, entry["estimator"]["variance"], config["allocation_diagnostic"])
        else:
            price["allocation_diagnostic"] = None
    return entry


def run_successor(config, *, config_bytes, **kwargs):
    record = original.run_preflight(config, config_bytes=config_bytes,
                                    evaluator=evaluate_bank, **kwargs)
    record.update(schema=SCHEMA, config_path=str(declaration.CONFIG.relative_to(ROOT)),
                  claim_boundary=CLAIM_BOUNDARY, successor=config["successor"],
                  derivative_validation=config["derivative_validation"],
                  allocation_diagnostic=config["allocation_diagnostic"])
    record.pop("preregistration")
    record["declaration"] = {"gate": "benchmarks/check_phase15_residual_successor_declaration.py",
                             "design_status": "post_hoc",
                             "config_claim_boundary_at_landing": config["claim_boundary"]}
    return record


def refusals(args):
    if args.out.resolve() == declaration.original.RECORD.resolve():
        return ["the original INVALID record is immutable"]
    if args.out.resolve() != RECORD.resolve():
        return ["the successor writes only its declared record path"]
    if RECORD.exists():
        return ["the successor runs once; use its checker to rebuild"]
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RECORD)
    args = parser.parse_args(argv)
    problems = refusals(args)
    notes, structural = [], {}
    if not problems:
        config_bytes = declaration.CONFIG.read_bytes()
        config = declaration.load_config()
        problems += declaration.static_problems(config)
        problems += declaration.commit_order_problems(notes, require_config=True)
    if not problems:
        problems += declaration.original.structural_problems(config, notes, structural)
    from clifford_qc.reproducibility import execution_provenance, stamp_record
    provenance = execution_provenance()
    if provenance["git_dirty"]:
        problems.append("commit the code before running the successor: tree is dirty")
    for problem in problems:
        print(f"REFUSE {problem}")
    if problems:
        return 1
    for note in notes:
        print(note)
    stamp_record({}, provenance)  # environment guard before any second-moment row
    record = run_successor(config, config_bytes=config_bytes, structural=structural,
                           progress=lambda line: print(line, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamp_record(record, provenance), indent=1,
                                   allow_nan=False) + "\n", encoding="utf-8")
    print(f"post-hoc uniform verdict: {record['decision']['verdict']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

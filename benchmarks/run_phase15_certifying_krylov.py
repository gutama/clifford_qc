"""Run Q18-S3 once: certify on the ladder's Krylov prefixes, then allocate.

See ``benchmarks/PHASE15_CERTIFYING_KRYLOV_DECLARATION.md``. For every prefix
``M = 1 ... 9`` of the committed Hubbard 2x2 Krylov sequence the producer
solves the ground Ritz root, reads its residual norm from
``SecondMomentBank`` and establishes it twice more, at eighty digits and by
statevector. It applies the rank margin, validates the energy's and the
residual's linearizations at eighty digits and with Q18-S1's fixed ladder, and
prices ``sigma`` against the energy under both declared QWC protocols. The
oracle verdict reads only the Neyman ratios of certifying, rank-stable
prefixes. Only if it is OPEN does the sampled pilot-allocation stage run, on
the AFFORDABLE certifying prefixes, and decide.

It refuses before forming a second-moment row unless the declaration gate
passes, the declaration is committed, no record exists, the working tree is
clean and the environment is one the committed records declare.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python benchmarks/run_phase15_certifying_krylov.py
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

from benchmarks import check_phase15_certifying_krylov_declaration as declaration
from benchmarks import check_phase15_certifying_screen_declaration as q18s2
from benchmarks import phase15_certifying_krylov as krylov
from benchmarks import phase15_certifying_screen as screen
from benchmarks import run_phase15_certifying_screen as q18s2_producer

RECORD = declaration.RECORD
SCHEMA = declaration.ARTIFACT_PATHS["schema"]
SCREEN_RECORD = q18s2.RECORD
PREDECESSOR = q18s2_producer.PREDECESSOR
SPECTRUM_TOLERANCE = 1e-12
CLAIM_BOUNDARY = (
    "This record reports the one Q18-S3 run its declaration froze. On every prefix of the "
    "committed Hubbard 2x2 Krylov sequence {H^k|psi>, k <= 8} it records the exact ground "
    "Ritz energy and residual norm, their eighty-digit and statevector reproductions, the "
    "rank margin, whether the Weinstein interval certifies the sector ground state, and, "
    "where the block is rank-stable and the variance resolved, both validated "
    "linearizations and Q18's uniform and Neyman cost ratios under both declared QWC "
    "protocols. The oracle verdict reads only the Neyman ratios of certifying, rank-stable "
    "prefixes and is first-order and asymptotic, from exact reference variances. A sampled "
    "pilot-allocation stage is recorded only where that verdict is OPEN. The basis was "
    "selected post hoc after Q18-S2. Nothing here is a device time, a coverage claim beyond "
    "the declared simulation, another system or basis, or a quantum-advantage claim.")
EVIDENCE = {
    "certification_and_first_moments": {
        "tier": "exact_small_sector_simulation",
        "scope": "Exact reference pairings of the Hubbard 2x2 Krylov prefixes' S, H and K "
                 "blocks, the sector spectrum, and eighty-digit and statevector "
                 "reproductions of each ground Ritz energy and variance; this system and "
                 "basis only."},
    "oracle_cost_ratios": {
        "tier": "asymptotic_bound",
        "scope": "First-order (delta-method) uniform and Neyman ratios from exact "
                 "per-setting reference variances under single-assignment QWC; a lower "
                 "bound on any allocator only to first order, not a finite-sample cost."},
    "practical_allocation": {
        "tier": "finite_sample_empirical",
        "scope": "Seeded multinomial simulation of pilot-allocated campaigns from the exact "
                 "grouped outcome distributions of the reference, run only where the oracle "
                 "verdict is OPEN; no device noise, readout error or device time."},
}


# ------------------------------------------------------------------ lineage

def frozen_generators(model):
    """Q18-S1's frozen nine-generator Hubbard 2x2 bank."""
    from benchmarks.run_mapping_axis import _selected_generators
    from benchmarks.run_mapping_axis import load_config as load_mapping

    spec = next(s for s in load_mapping()["systems"] if s["key"] == q18s2.SYSTEM)
    return _selected_generators(model, spec)[1]


def frozen_bank_entry(model, spectrum, predecessor, generators) -> dict:
    """Q18-S1's frozen bank, priced again through Q18-S2's own path."""
    config = q18s2.load_config()
    size = len(generators)
    history = [math.nan] * (size - 1) + [predecessor["estimator"]["ground_energy"]]
    grouping = config["grouping"]
    return screen.evaluate_trajectory(
        config, model, generators, history, spectrum,
        protocols=(grouping["declared_protocol"], grouping["alternative_protocol"]),
        prefixes=[size])[str(size)]


def lineage_checks(model, spectrum, prefixes, predecessor, screen_record, ladder,
                   generators) -> dict:
    """The three ties to committed records the verdict depends on."""
    frozen = screen_record["spectral_reference"]
    entry = frozen_bank_entry(model, spectrum, predecessor, generators)
    final = prefixes[max(prefixes, key=int)]
    out = {
        "frozen_bank_entry": entry,
        "spectral_reference": {
            key: (abs(spectrum[key] - frozen[key]) <= SPECTRUM_TOLERANCE
                  if isinstance(frozen[key], float) else spectrum[key] == frozen[key])
            for key in sorted(frozen)},
        "frozen_bank": q18s2_producer.lineage_entry(entry, predecessor),
        "ladder_row": {
            "energy": abs(final["ground_energy"] - ladder["energy"])
            <= declaration.FLOAT_TOLERANCE,
            "retained_rank": final["rank"]["effective_rank"] == ladder["retained_rank"],
        },
    }
    out["passes"] = all(all(block.values()) for key, block in out.items()
                        if key != "frozen_bank_entry")
    return out


# ------------------------------------------------------------------ the run

def decide(config, prefixes, lineage, practical) -> dict:
    statuses = {size: entry["status"] for size, entry in prefixes.items()}
    domain = [size for size, entry in prefixes.items() if entry["in_domain"]]
    oracle = "INVALID" if not lineage["passes"] else krylov.oracle_verdict(prefixes.values())
    practical_statuses = {size: block["status"] for size, block in practical.items()}
    final = krylov.final_verdict(oracle, practical_statuses)
    return {"domain": domain, "statuses": statuses, "oracle_verdict": oracle,
            "practical_prefixes": sorted(practical, key=int),
            "practical_statuses": practical_statuses, "final_verdict": final,
            "consequence": config["consequences"][final],
            "rule": "frozen in the config; see decision_rule"}


def run(config: dict, *, config_bytes: bytes, model=None, predecessor=None,
        screen_record=None, ladder=None, frozen=None, progress=None) -> dict:
    """The declared run as an unstamped record. Overrides are for tests."""
    from clifford_qc.evidence import declare_evidence

    started = time.perf_counter()
    model = declaration.system_model() if model is None else model
    if predecessor is None:
        predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))["banks"][
            q18s2.SYSTEM]
    if screen_record is None:
        screen_record = json.loads(SCREEN_RECORD.read_text(encoding="utf-8"))
    ladder = declaration.ladder_row() if ladder is None else ladder
    spectrum = screen.sector_spectrum(model)
    bank = krylov.krylov_bank(model, max(config["basis"]["prefixes"]) - 1)
    generators = [bank.generator(i) for i in range(len(bank))]
    premises = screen.sector_premises(model, [generators])
    grouping = config["grouping"]
    protocols = (grouping["declared_protocol"], grouping["alternative_protocol"])
    prefixes = krylov.evaluate_prefixes(config, model, spectrum,
                                        prefixes=config["basis"]["prefixes"],
                                        protocols=protocols, progress=progress)
    frozen = frozen_generators(model) if frozen is None else frozen
    lineage = lineage_checks(model, spectrum, prefixes, predecessor, screen_record, ladder,
                             frozen)
    oracle = "INVALID" if not lineage["passes"] else krylov.oracle_verdict(prefixes.values())
    practical = {}
    if oracle == "OPEN":
        for size in krylov.practical_prefixes(prefixes):
            if progress is not None:
                progress(f"practical stage on M={size}")
            practical[size] = krylov.practical_stage(config, model, prefixes[size], spectrum,
                                                     progress=progress)
    record = {
        "schema": SCHEMA,
        "config_path": str(declaration.CONFIG.relative_to(ROOT)),
        "config_digest": hashlib.sha256(config_bytes).hexdigest(),
        "claim_boundary": CLAIM_BOUNDARY,
        "quantum_advantage_claim": False,
        "question": config["question"],
        "design": config["evidence"],
        "statistic": config["statistic"],
        "certification": config["certification"],
        "rank_margin": config["rank_margin"],
        "grouping": config["grouping"],
        "derivative_validation": config["derivative_validation"],
        "fixed_step_validation": config["fixed_step_validation"],
        "allocation_diagnostic": config["allocation_diagnostic"],
        "practical_allocation": config["practical_allocation"],
        "spectral_reference": spectrum,
        "premises": premises,
        "basis": {"labels": config["basis"]["labels"], "prefixes": prefixes},
        "lineage": lineage,
        "practical": practical,
        "decision": decide(config, prefixes, lineage, practical),
        "declaration": {
            "gate": "benchmarks/check_phase15_certifying_krylov_declaration.py",
            "design_status": config["evidence"]["design_status"],
            "basis_selection": config["evidence"]["basis_selection"],
            "config_claim_boundary_at_landing": config["claim_boundary"],
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    return declare_evidence(record, EVIDENCE)


def refusals(args) -> list[str]:
    if args.out.resolve() != RECORD.resolve():
        return ["Q18-S3 writes only its declared record path"]
    if RECORD.exists():
        return ["Q18-S3 runs once; use its checker to rebuild"]
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
        problems.append("commit the code before running Q18-S3: the tree is dirty")
    for problem in problems:
        print(f"REFUSE {problem}")
    if problems:
        return 1
    for note in notes:
        print(note)
    stamp_record({}, provenance)  # environment guard before any second-moment row
    record = run(config, config_bytes=config_bytes,
                 progress=lambda line: print(line, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamp_record(record, provenance), indent=1,
                                   allow_nan=False) + "\n", encoding="utf-8")
    decision = record["decision"]
    print(f"\noracle: {decision['oracle_verdict']}  final: {decision['final_verdict']}  "
          f"domain: {decision['domain']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

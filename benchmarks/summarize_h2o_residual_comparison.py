"""Generate the H2O comparison report exclusively from its observations."""
import argparse
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.metadata_io import read_json
from benchmarks.run_h2o_residual_comparison import ARMS, RECORD

REPORT = ROOT / "benchmarks/results/h2o_residual_comparison.md"


def first_target(worker):
    if worker["status"] == "ok":
        return next((r for r in worker["result"]["trajectory"] if r["target_reached"]), None)
    return None


def render(record):
    workers, repeats = record["workers"], record["config"]["repeats"]
    lines = ["# H2O spin-matched residual-growth comparison", "",
             "Exploratory exact simulation of one H2O STO-3G CAS(8e,6o) Hamiltonian: 225 determinants, singlet solver spans.",
             "All arms start with the complete singlet CAS(2e,2o) on orbitals [3,4].",
             f"Target: {1000*record['config']['accuracy_target_hartree']:.1f} mHa versus the full-sector energy.",
             f"Fresh processes: {len(workers)}; failed/timed out: {sum(w['status'] != 'ok' for w in workers)}.", "",
             "## First target prefix", "",
             "Medians over target-reaching runs; success denominators include all declared repeats.",
             "Time includes common numerical setup plus cumulative solver work; interpreter imports/startup are excluded.",
             "RSS is OS process high-water memory including imports/setup, not incremental solver memory.", "",
             "| Arm | Target repeats | Columns | H actions | Candidates scored | Time incl. setup (s) | Peak RSS (MiB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        hits = [(w, first_target(w)) for w in workers if w["arm"] == arm and first_target(w) is not None]
        if not hits:
            lines.append(f"| {arm} | 0/{repeats} | — | — | — | — | — |")
            continue
        values = [f"{median([r[key] for _, r in hits]):g}"
                  for key in ("dimension", "h_action_calls", "candidates_scored")]
        elapsed = median([w["result"]["setup"]["wall_seconds"] + r["wall_seconds"] for w, r in hits])
        rss = median([r["peak_rss_bytes"]/2**20 for _, r in hits])
        lines.append(f"| {arm} | {len(hits)}/{repeats} | " + " | ".join(values) + f" | {elapsed:.6f} | {rss:.3f} |")
    lines += ["", "## Support and tracked payload at first target", "",
              "Medians over target-reaching runs. V/HV and the screened-candidate H-action cache are partial array payloads, not total resident storage.",
              "Every arm additionally constructs fixed external-group frame copies; these, temporary candidates and other work arrays are outside these two counters. Use OS RSS for whole-process memory.", "",
              "| Arm | Determinant support | Basis amplitude nonzeros | V/HV (KiB) | Screened-candidate cache (KiB) |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        hits = [first_target(w) for w in workers if w["arm"] == arm and first_target(w) is not None]
        if not hits:
            lines.append(f"| {arm} | — | — | — | — |")
            continue
        values = [f"{median([r[key] for r in hits]):g}" for key in
                  ("determinant_support", "amplitude_nonzeros")]
        values += [f"{median([r[key] for r in hits])/1024:.3f}" for key in
                   ("retained_payload_bytes", "candidate_cache_payload_bytes")]
        lines.append(f"| {arm} | " + " | ".join(values) + " |")
    lines += ["", "## Retained-column frontiers", "",
              "Median error (mHa) at the largest observed prefix within each cap; no interpolation across CAS block jumps.",
              "Equal columns do not equalize support, cache memory or selection work.", "",
              "| Column cap | Residual packets | Spin-adapted selected CI | Adaptive active space |", "|---|---:|---:|---:|"]
    for cap in (8, 16, 32, 64, 105):
        values = []
        for arm in ARMS:
            errors = []
            for w in workers:
                if w["arm"] == arm and w["status"] == "ok":
                    prefixes = [r for r in w["result"]["trajectory"] if r["dimension"] <= cap]
                    if prefixes:
                        errors.append(1000*prefixes[-1]["error_hartree"])
            values.append(f"{median(errors):.6f}" if errors else "—")
        lines.append(f"| {cap} | " + " | ".join(values) + " |")
    lines += ["", "## Final readout", "",
              "| Arm | Completed repeats | Median error (mHa) | Stop reasons | Median full process time (s) |",
              "|---|---:|---:|---|---:|"]
    for arm in ARMS:
        group = [w for w in workers if w["arm"] == arm and w["status"] == "ok"]
        if not group:
            lines.append(f"| {arm} | 0/{repeats} | — | failed/timeout | — |")
            continue
        error = median([w["result"]["trajectory"][-1]["error_hartree"]*1000 for w in group])
        reasons = ", ".join(sorted({w["result"]["stop_reason"] for w in group}))
        elapsed = median([w["process_wall_seconds"] for w in group])
        lines.append(f"| {arm} | {len(group)}/{repeats} | {error:.9f} | {reasons} | {elapsed:.6f} |")
    for w in workers:
        if w["status"] != "ok":
            lines += ["", f"Failure: {w['arm']}, repeat {w['repeat']}, {w['status']}.", "```text", w["diagnostic"], "```"]
    lines += ["", "## Interpretation and scope", "",
              "Every arm preserves singlet symmetry. Selected CI selects deterministic configuration-state functions, not spin-contaminated determinants.",
              "Packets are refreshed after growth; active-space enlargement adds complete new singlet CAS blocks.",
              "Cached candidate H actions are charged when first built. Retained V and HV are reused; timed solvers build neither full H nor H².",
              "Dense S², the complete singlet frame, exact diagonals and operator compilation remain common small-sector setup costs. Setup constructs 225 S² columns.",
              "JSON retains array payloads, amplitude support and every prefix. Payload is not RSS; setup plus candidate caches must also be considered.",
              "The full reference energy is obtained after all measured solves. An independent occupation replay checks selection, Ritz energies, residuals, spin, support and action counts.",
              "These three-repeat tiny-system timings are descriptive. Column compression alone establishes no time/memory advantage; no optimized SHCI/DMRG, scalable spin adaptation, quantum preparation, shots or quantum advantage is established.", "",
              f"Declaration: `{record['declaration_commit']}`.", f"Execution: `{record['source_commit']}`.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(read_json(ROOT / RECORD))
    if args.check:
        if not REPORT.exists() or REPORT.read_text() != text:
            print("FAIL H2O report differs from observations")
            return 1
        print("OK H2O report matches observations")
    else:
        REPORT.write_text(text)
        print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

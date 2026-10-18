"""Generate/check the pilot report directly from its committed observation record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks/reference_results/orthogonal_active_space.json"
REPORT = ROOT / "benchmarks/results/orthogonal_active_space.md"


def render(record):
    rows = [
        "# Orthogonal active-space pilot",
        "",
        "Exact classical simulation on the four frozen molecular active-space inputs.",
        "All errors in the first table are against the independently checked full-sector ground energy.",
        "The classical reference is the complete four-determinant CAS(2e,2o), with other occupations frozen at HF.",
        "",
        "| Input | Classical error (mHa) | Packet error (mHa) | Equal-dimension determinant error (mHa) | Support-containing determinant error (mHa) | Corrections | Columns | Determinant support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    reached, initial_reached = [], []
    for case in record["cases"]:
        first, last = case["trajectory"][0], case["trajectory"][-1]
        a = last["arms"]
        rows.append(f"| {case['id']} | {first['arms']['orthogonal']['error'] * 1000:.6f}"
                    f" | {a['orthogonal']['error'] * 1000:.6f}"
                    f" | {a['dimension_matched_determinants']['error'] * 1000:.6f}"
                    f" | {a['support_matched_determinants']['error'] * 1000:.6f}"
                    f" | {last['retained_corrections']} / {last['requested_corrections']}"
                    f" | {a['orthogonal']['basis_dimension']} | {a['orthogonal']['determinant_support']} |")
        if a["orthogonal"]["target_reached"]:
            reached.append(case["id"])
        if first["arms"]["orthogonal"]["target_reached"]:
            initial_reached.append(case["id"])
    rows += [
        "",
        f"At the final declared budget, {len(reached)}/4 inputs meet 1.6 mHa: {', '.join(reached)}.",
        f"The classical reference already meets it on: {', '.join(initial_reached) or 'none'}.",
        "BeH2 has only three nonzero candidate packets; its four-packet request is explicitly capped.",
        "All nested packet energies decrease. Exact same-span mixed coordinates reproduce the orthogonal energies,",
        "and the determinant control containing each packet's support has no higher energy (to numerical tolerance).",
        "Packet gains over equal-dimension determinants are small and inconsistent across inputs.",
        "The packet and determinant controls have different amplitude support and selection work; column counts alone do not establish a computational advantage.",
        "",
        "## Illustrative centered matrix-noise diagnostic",
        "",
        "The table uses the largest frozen scale: 0.001 Ha Hamiltonian-entry noise and 0.001 dimensionless overlap-entry noise.",
        "RMSE is conditional on successful solves and measured around the **noiseless retained-span energy**, not the full-sector ground energy.",
        "Every scale/replica and every failed solve is retained in the JSON; unconditional target fractions count failures as failures.",
        "",
        "| Input | Known orthogonal overlap RMSE (mHa) | Mixed exact overlap RMSE (mHa) | Mixed noisy overlap RMSE (mHa) | Noisy-overlap failures / replicas |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in record["cases"]:
        summary = case["noise"]["scales"][-1]["summary"]
        values = [summary[name] for name in ("orthogonal_known_overlap", "mixed_known_overlap", "mixed_noisy_overlap")]
        errors = ["unresolved" if value["conditional_rmse"] is None else f"{value['conditional_rmse'] * 1000:.6f}" for value in values]
        rows.append(f"| {case['id']} | {' | '.join(errors)} | {values[-1]['failures']} / {values[-1]['replicas']} |")
    rows += [
        "",
        "Congruently transported noise gives the same result with either exact overlap representation, replica by replica.",
        "The added overlap-noise proxy does not demonstrate a substantial accuracy benefit for orthogonality on this frozen grid.",
        "These Gaussian matrix perturbations are not Pauli samples, hardware noise, shot counts or confidence certificates.",
        "A small estimation RMSE does not repair a retained subspace's energy bias.",
        "",
        "## Scope and provenance",
        "",
        "The experiment constructs packets classically from full-sector residuals and a full-sector diagonal.",
        "It stores explicit vectors, charges candidate selection and every projected Hamiltonian action, and reports payload bytes rather than peak RSS.",
        "Quantum preparation, physical transition measurements, hardware-accessible selection and total quantum cost remain unimplemented and unpriced.",
        "No superiority over established SHCI/DMRG implementations or quantum advantage is inferred.",
        "",
        f"- Declaration commit: `{record['declaration_commit']}`.",
        f"- Execution commit: `{record['source_commit']}`.",
        f"- Configuration SHA-256: `{record['config_sha256']}`.",
        "- The digest-bound [execution bundle](../provenance/orthogonal_active_space.bundle) preserves the original Git objects if publication changes commit IDs; the independent checker restores objects without changing branches.",
        "- Original observation bytes are preserved. A later verification fix handles numerical packet amplitudes crossing the serialization cutoff by comparing reconstructed vectors at the existing tolerance; it changes no result or campaign setting.",
        "- Sources: [design](../ORTHOGONAL_ACTIVE_SPACE_DECLARATION.md), [observations](../reference_results/orthogonal_active_space.json), and [independent checker](../check_orthogonal_active_space.py).",
        "",
    ]
    return "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    expected = render(json.loads(RECORD.read_text()))
    if args.check:
        if not args.output.exists() or args.output.read_text() != expected:
            print("FAIL orthogonal active-space report is stale")
            return 1
        print("VALID orthogonal active-space report")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

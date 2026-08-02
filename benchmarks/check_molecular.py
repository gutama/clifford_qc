"""Verify the molecular records are internally consistent and honestly labelled.

``run_molecular_pipeline.py`` writes one JSON per molecule plus a master
summary, and ``molecular_simulation_report.{md,tex}`` is derived from them. A
hand-edit to that record reached a submission-adjacent report and none of the
existing gates saw it: ``check_summaries.py`` covers ``benchmarks/reference_results``
and ``check_docs.py`` covers ``REPRODUCING.md``, so ``molecular_results/`` had
no checker at all.

The specific failure is worth naming, because every check below is derived from
it. Three fields were updated in place -- ``e_acase_adaptive``,
``error_adaptive_mha`` and ``subspace_size_m`` -- and their consistency
partners were not. The result claimed effective rank 11 inside a
6-dimensional subspace, an H2O energy that implied 1.27 mHa beside a reported
1.45 mHa, and a wavefunction whose leading amplitude could not have lowered the
reference energy by anything like the amount claimed. Each is arithmetic, not
physics, and each is catchable without rerunning anything.

Checks, per molecule:

  1. ``effective_rank <= subspace_size_m`` -- an M-dimensional subspace cannot
     have effective rank above M;
  2. ``error_*_mha`` agrees with ``1000 * (E - e_exact_fci)`` for every arm;
  3. every accuracy flag agrees with the error it describes;
  4. any run reporting chemical accuracy says which arm reached it and whether
     an oracle stop was used to get there;
  5. a ``sd_spans_full_sector`` row is flagged, so "exact to machine precision"
     is never read as accuracy when it is arithmetic;
  6. the Ritz vector is normalized and its leading amplitude is consistent with
     the correlation energy claimed (the Cauchy-Schwarz check below);
  7. ``hamiltonian_pauli_terms`` and ``element_word_universe`` are both present
     and are not confused for one another -- the first is the distinct-word
     count of H, the second the union over element operators, and reporting one
     under the other's name inflated the resource column by ~100x;
  8. the per-molecule JSON matches the master summary entry.

    python benchmarks/check_molecular.py

Exits nonzero on any inconsistency, naming the field and the arithmetic.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "molecular_results"
SUMMARY = RESULTS / "results_summary.json"

CHEMICAL_ACCURACY_MHA = 1.5936

# Energy/error agreement tolerance. The fields are written from the same float
# in one process, so anything above rounding is a transcription, not drift.
ERROR_TOL_MHA = 1e-3


def _fail(errors: list[str], molecule: str, message: str) -> None:
    errors.append(f"{molecule}: {message}")


def check_record(key: str, record: dict, errors: list[str]) -> None:
    m = record.get("subspace_size_m")
    rank = record.get("effective_rank")
    if m is None or rank is None:
        _fail(errors, key, "missing subspace_size_m or effective_rank")
    elif rank > m:
        _fail(errors, key,
              f"effective_rank {rank} exceeds basis size M={m}; an "
              f"M-dimensional subspace cannot have rank above M, so at least "
              f"one of the two fields is stale")

    exact = record.get("e_exact_fci")
    if exact is None:
        _fail(errors, key, "missing e_exact_fci")
        return

    # 2 + 3: every arm's error and accuracy flag must follow from its energy.
    arms = [
        ("adaptive", "e_acase_adaptive", "error_adaptive_mha",
         "adaptive_chemical_accuracy"),
        ("sd", "e_sd_complete", "error_sd_mha", "sd_chemical_accuracy"),
        ("cisd", "e_cisd", "error_cisd_mha", "cisd_chemical_accuracy"),
        ("ccsd", "e_ccsd", "error_ccsd_mha", "ccsd_chemical_accuracy"),
    ]
    # The complete-SD arm is optional per molecule: it reproduces PySCF CISD
    # exactly, so geometries added only to test CCSD skip it. A skipped arm
    # must be *declared* skipped rather than merely absent, or a run that
    # crashed halfway would look the same as one that opted out.
    sd_run = record.get("complete_sd_run")
    if sd_run is None:
        _fail(errors, key, "missing complete_sd_run; a skipped arm has to say "
                           "it was skipped, not just leave fields empty")
    for arm, e_field, err_field, flag_field in arms:
        energy = record.get(e_field)
        reported = record.get(err_field)
        if arm == "sd" and sd_run is False:
            if energy is not None or reported is not None:
                _fail(errors, key,
                      "complete_sd_run is False but the SD arm reported "
                      f"{e_field}/{err_field}; one of the two is wrong")
            continue
        if energy is None or reported is None:
            _fail(errors, key, f"missing {e_field} or {err_field}")
            continue
        computed = (energy - exact) * 1000.0
        if abs(computed - reported) > ERROR_TOL_MHA:
            _fail(errors, key,
                  f"{err_field} = {reported:.4f} mHa but {e_field} implies "
                  f"{computed:.4f} mHa (difference {computed - reported:+.4f}); "
                  f"the energy and its error come from different runs")
        if flag_field is not None:
            flag = record.get(flag_field)
            expected = abs(computed) <= CHEMICAL_ACCURACY_MHA
            if flag is not None and bool(flag) != expected:
                _fail(errors, key,
                      f"{flag_field} = {flag} but |error| = {abs(computed):.4f} "
                      f"mHa against a {CHEMICAL_ACCURACY_MHA} mHa threshold")

    # 4: a chemical-accuracy claim has to say how it was reached.
    if record.get("adaptive_chemical_accuracy"):
        if "adaptive_oracle_stop_used" not in record:
            _fail(errors, key,
                  "claims adaptive chemical accuracy without recording "
                  "adaptive_oracle_stop_used; a stop fed the exact energy "
                  "answers a different question than an unaided run")
        if not record.get("adaptive_stop_reason"):
            _fail(errors, key,
                  "claims adaptive chemical accuracy without an "
                  "adaptive_stop_reason")

    # 5: exactness that is arithmetic must be labelled as such.
    sd_size = record.get("complete_sd_basis_size")
    sector = record.get("sector_dimension")
    if sd_size is not None and sector is not None:
        spans = sd_size >= sector
        if bool(record.get("sd_spans_full_sector")) != spans:
            _fail(errors, key,
                  f"sd_spans_full_sector = {record.get('sd_spans_full_sector')} "
                  f"but SD basis {sd_size} against sector dimension {sector} "
                  f"says {spans}")

    # 6: the wavefunction has to be able to produce the energy.
    #
    # For a normalized Ritz vector the correlation weight outside the reference
    # is 1 - |c_0|^2. A state that is the reference to within 1e-6 in weight
    # cannot carry tens of mHa of correlation energy, and that mismatch is
    # exactly what a stale observables block looks like. This is a coarse
    # necessary condition, not the full Cauchy-Schwarz bound: it needs no
    # Hamiltonian, so it runs from the record alone.
    coeffs = record.get("ground_state_coefficients") or {}
    if coeffs:
        amps = [complex(v["real"], v["imag"]) for v in coeffs.values()]
        norm = math.sqrt(sum(abs(a) ** 2 for a in amps))
        leading = max(abs(a) for a in amps)
        if norm > 0:
            outside = max(0.0, 1.0 - (leading / norm) ** 2)
            e_rhf = record.get("e_rhf")
            e_ad = record.get("e_acase_adaptive")
            if e_rhf is not None and e_ad is not None:
                recovered_mha = abs(e_ad - e_rhf) * 1000.0
                # 1 mHa of correlation needs weight somewhere off the
                # reference; the constant is deliberately loose so only a
                # gross inconsistency trips it.
                if recovered_mha > 1.0 and outside < 1e-4:
                    _fail(errors, key,
                          f"Ritz vector is the reference to weight "
                          f"{outside:.2e} outside it, but the adaptive energy "
                          f"sits {recovered_mha:.2f} mHa below RHF; the "
                          f"wavefunction cannot have produced that energy")

    # 6b: a correlated state must show double occupancy below the closed-shell
    # value. Equality to many digits is the signature of an uncorrelated state.
    d = record.get("double_occupancy")
    d_hf = record.get("double_occupancy_uncorrelated")
    if d is not None and d_hf is not None:
        e_rhf = record.get("e_rhf")
        e_ad = record.get("e_acase_adaptive")
        if e_rhf is not None and e_ad is not None:
            if abs(e_ad - e_rhf) * 1000.0 > 1.0 and abs(d - d_hf) < 1e-7:
                _fail(errors, key,
                      f"double_occupancy {d:.9f} equals the uncorrelated value "
                      f"{d_hf:.9f} to 1e-7 while the energy claims correlation; "
                      f"the observables block is from a different run")

    # 7: the two word counts are distinct quantities and both must be present.
    for field in ("hamiltonian_pauli_terms", "element_word_universe"):
        if field not in record:
            _fail(errors, key, f"missing {field}")
    ham = record.get("hamiltonian_pauli_terms")
    universe = record.get("element_word_universe")
    if ham is not None and universe is not None and ham > universe:
        _fail(errors, key,
              f"hamiltonian_pauli_terms {ham} exceeds element_word_universe "
              f"{universe}; the element universe is a union over A_i'HA_j and "
              f"contains supp(H), so this says the two are swapped")

    # 9: §6 resource accounting. A basis size is not a compactness result on
    # its own -- the plan asks for bank build time and peak memory beside it,
    # and a 141-vector basis costing gigabytes is a result in its own right.
    required = ["adaptive_cached_operator_bytes", "adaptive_seconds",
                "adaptive_peak_rss_bytes", "adaptive_assemble_seconds",
                "adaptive_growth_seconds"]
    if record.get("complete_sd_run"):
        required.append("sd_peak_rss_bytes")
    for field in required:
        if field not in record:
            _fail(errors, key, f"missing §6 resource field {field}")

    # 9b: A-CASE's chemistry candidates are singles and doubles on the HF
    # reference, so its span is a *subspace* of the CISD space and Rayleigh-Ritz
    # cannot go below the CISD minimum -- **at matched multiplicity**.
    #
    # That qualifier is the whole content of this check, and leaving it out cost
    # a wrong conclusion. Neither A-CASE nor a plain determinant diagonalization
    # constrains spin, while CISD from a closed-shell reference means the
    # singlet. Where a high-spin state lies below the lowest singlet, A-CASE can
    # legitimately come out under the CISD energy without anything being wrong
    # with either. The spin check below is what makes this comparison
    # meaningful, so it runs first.
    # The comparison is against a CISD we computed, never against PySCF's.
    # ``e_cisd_determinant`` is the diagonalized SD determinant block; on rows
    # that ran the complete-SD arm, ``e_sd_complete`` is the same quantity by a
    # slower route, so those rows need no separate field.
    # 9a: the Ritz state has to be the multiplicity everything else targets.
    # These are closed-shell molecules, so <S^2> = 0. A-CASE's basis is
    # S_z-conserving but not spin-adapted, and at H2O 2.5x it converged to
    # <S^2> = 6 -- a quintet whose energy is not comparable to CCSD's singlet.
    # That row was removed; this stops the next one being read as a result.
    spin2 = record.get("total_spin_squared")
    comparable = True
    if spin2 is None:
        _fail(errors, key, "missing total_spin_squared; without it a Ritz "
                           "energy cannot be shown to describe the same state "
                           "the classical baselines do")
    elif abs(spin2) > 1.0:
        comparable = False
        _fail(errors, key,
              f"<S^2> = {spin2:.4f} on a closed-shell molecule: the Ritz state "
              f"is not a singlet, so its energy is not comparable to CCSD or "
              f"CISD. Drop the row or spin-adapt the basis")
    elif abs(spin2) > 0.05:
        print(f"  note: {key}: <S^2> = {spin2:.4f}, small but not negligible "
              f"spin contamination in the Ritz state", file=sys.stderr)

    e_ad = record.get("e_acase_adaptive")
    reference_cisd = record.get("e_cisd_determinant")
    source = "determinant CISD"
    if reference_cisd is None and record.get("complete_sd_run"):
        reference_cisd = record.get("e_sd_complete")
        source = "complete-SD"
    if e_ad is not None and reference_cisd is not None and comparable:
        if e_ad < reference_cisd - 1e-6:
            _fail(errors, key,
                  f"A-CASE {e_ad:.9f} lies below the {source} minimum "
                  f"{reference_cisd:.9f} by {(reference_cisd - e_ad) * 1000:.3f} "
                  f"mHa at matched multiplicity, but its basis is a subspace "
                  f"of that space and cannot")
    elif e_ad is not None:
        _fail(errors, key,
              "no CISD reference to check A-CASE against: the row has neither "
              "e_cisd_determinant nor a complete-SD arm")

    # A PySCF disagreement is a recorded fact about the external solver, not a
    # defect in the record -- but it has to be recorded rather than silent.
    agrees = record.get("cisd_pyscf_agrees")
    e_cisd = record.get("e_cisd")
    if agrees is not None and not agrees:
        det = record.get("e_cisd_determinant")
        if det is not None and e_cisd is not None:
            print(f"  note: {key}: PySCF CISD is {(e_cisd - det) * 1000:+.3f} "
                  f"mHa off the determinant CISD minimum; the record uses ours",
                  file=sys.stderr)

    # 10: an unconverged reference makes every correlated number built on it
    # meaningless rather than merely inaccurate, and stretched geometries are
    # exactly where that happens.
    for flag in ("rhf_converged", "cisd_converged", "ccsd_converged"):
        if flag not in record:
            _fail(errors, key, f"missing convergence flag {flag}")
        elif record[flag] is False:
            _fail(errors, key,
                  f"{flag} is False; the correlated baselines on this row are "
                  f"not trustworthy and the row should be dropped or labelled")

    for arm in ("adaptive", "sd"):
        if arm == "sd" and record.get("complete_sd_run") is False:
            continue
        peak = record.get(f"{arm}_peak_rss_bytes")
        delta = record.get(f"{arm}_peak_rss_delta_bytes")
        if peak is not None and delta is not None:
            if peak <= 0:
                _fail(errors, key, f"{arm}_peak_rss_bytes is {peak}; the "
                                   f"sampler never read a resident set size")
            elif delta > peak:
                _fail(errors, key,
                      f"{arm}_peak_rss_delta_bytes {delta} exceeds "
                      f"{arm}_peak_rss_bytes {peak}; the delta is measured "
                      f"against a baseline inside the peak")

    # The two bank timings are *nested*, not disjoint, and an earlier version of
    # this check got that wrong -- it summed them and tripped on all four
    # molecules at a consistent 1.4-1.8x, which is the signature of
    # double-counting rather than of bad data. `growth_seconds` brackets the
    # whole run_acase call (it matches the measured wall clock to the
    # millisecond), and `assemble_seconds` is the operator-product time spent
    # inside that window. So the invariant is containment, one level at a time.
    assemble = record.get("adaptive_assemble_seconds")
    growth = record.get("adaptive_growth_seconds")
    outer = record.get("adaptive_seconds")
    if assemble is not None and growth is not None and assemble > growth + 1.0:
        _fail(errors, key,
              f"adaptive assemble {assemble:.1f}s exceeds growth "
              f"{growth:.1f}s, but assembly happens inside the growth loop")
    if growth is not None and outer is not None and growth > outer + 1.0:
        _fail(errors, key,
              f"adaptive growth {growth:.1f}s exceeds the measured wall clock "
              f"{outer:.1f}s of the call containing it; the timings describe "
              f"different runs")


def main() -> int:
    if not SUMMARY.exists():
        print(f"missing {SUMMARY.relative_to(ROOT)}; run "
              f"`python run_molecular_pipeline.py`", file=sys.stderr)
        return 1

    summary = json.loads(SUMMARY.read_text())
    errors: list[str] = []

    for key, record in summary.items():
        check_record(key, record, errors)

        # 8: the per-molecule file and the summary entry are the same record.
        per_file = RESULTS / f"{key}_results.json"
        if not per_file.exists():
            _fail(errors, key, f"missing {per_file.name}")
            continue
        per = json.loads(per_file.read_text())
        differing = sorted(
            k for k in set(per) | set(record)
            # timings are wall-clock and legitimately differ between the two
            # writes only if the record was regenerated piecemeal, which is
            # itself the drift being checked -- so they are compared too.
            if per.get(k) != record.get(k)
        )
        if differing:
            _fail(errors, key,
                  f"{per_file.name} disagrees with the summary entry on: "
                  f"{', '.join(differing)}")

    if errors:
        print("molecular record inconsistencies:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(f"\n{len(errors)} problem(s). Regenerate with "
              f"`python run_molecular_pipeline.py` rather than editing the "
              f"JSON: the fields are consistency partners and updating one by "
              f"hand is what these checks exist to catch.", file=sys.stderr)
        return 1

    print(f"molecular records consistent ({len(summary)} molecules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

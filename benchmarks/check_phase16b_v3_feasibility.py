"""Verify the third Phase 16B record and that its verdict follows the frozen rule.

This gate does not resample. It re-derives every instance status, every
per-scheme verdict, and the overall verdict from the record's own per-cell
medians under the config's rule, and fails if the recorded verdict is not the
one that rule produces.

Beyond the v2 gate it enforces what this experiment added.

  * **The incumbent actually got its grouping.** A comparison that claims to
    price grouping and then charges the incumbent one setting per word is the
    v2 experiment wearing a new label, so the recorded setting counts must fall
    under the primary schemes, and must match the group counts the
    preregistration gate re-derived before the run.
  * **The setting model is the declared one.** Every arm's recorded setting
    count is recomputed from Section 3 of the design -- `2m` for `rt_unitary`,
    `2(m-1) + m*G_H` for `rt_hermitian`, `2(m-1) + m(m+1)/2*G_H` for
    `rt_trotter` -- against the `G_H` the record carries for that scheme.
  * **Covariance was carried, not assumed.** Every readout arm must record the
    construction's verification with zero mismatches, and on QWC and ungrouped
    settings the shot-for-shot literal check must have run, because that is the
    only place the declared per-qubit product readout is directly comparable.
  * **The verdict is the least favourable primary scheme.** Recomputed here,
    and `ungrouped` must not be among them: it is the v2 cost model this
    experiment exists to replace.

Everything the v2 gate checked still applies: digest, provenance, evidence
label, coverage, INCOMPLETE rejection, replicas and target read from the frozen
config rather than the record, admission recomputed from the per-size zero-noise
errors, and the Trotter substitution as a measured identity.

    python benchmarks/check_phase16b_v3_feasibility.py

Exits nonzero on any problem.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs" / "phase16b_v3_feasibility.json"
RECORD = ROOT / "reference_results" / "phase16b_v3_feasibility.json"
SCHEMA = "clifford_qc.phase16b_v3_feasibility.v1"
PENCIL_TOL = 1e-10
ROUND_OFF_FLOOR = 1e-12


def zero_noise_minimum(arms, arm_names):
    values = [info["zero_noise_error"]
              for arm_name in arm_names
              for info in arms.get(arm_name, {}).get("sizes", {}).values()
              if info.get("zero_noise_error") is not None]
    return min(values) if values else float("inf")


def distinct_verifications(arm):
    seen, out = set(), []
    for info in arm.get("sizes", {}).values():
        report = info.get("trotter_step_verification")
        if report is None:
            continue
        key = (report.get("shared_by_qubit_count"), report.get("verified_on_instance"))
        if key in seen:
            continue
        seen.add(key)
        out.append(report)
    return out


def declared_settings(arm_name, m, g_h):
    """Section 3 of the design, recomputed rather than read back."""
    if arm_name == "rt_unitary":
        return 2 * m
    if arm_name == "rt_hermitian":
        return 2 * (m - 1) + m * g_h
    if arm_name == "rt_trotter":
        return 2 * (m - 1) + (m * (m + 1) // 2) * g_h
    return None


def priceable_arms(config):
    """Arms the cost model prices; `exact_diag` is ground truth and is not one."""
    return sorted(name for name, spec in config["arms"].items()
                  if spec.get("priceable") is not False)


def expected_cells(config):
    """Every (arm, size, regularizer, budget) key the frozen config declares.

    The gate used to audit whatever cells the record happened to carry, which
    establishes that the present ones are well formed and nothing at all about
    the ones that are not. Deleting a single budget cell from a record left it
    passing as complete. The declared sweep is a Cartesian product and it is
    written down, so the comparison is against that product.
    """
    budgets = {f"1e{int(decade)}" for decade in config["cost_model"]["budget_grid_decades"]}
    regularizers = set(config["regularizers"])
    cells = set()
    for arm in priceable_arms(config):
        for size in config["basis_sizes_by_arm"][arm]:
            for regularizer in regularizers:
                for budget in budgets:
                    cells.add((arm, str(size), regularizer, budget))
    return cells


def recorded_cells(scheme_entry):
    cells = set()
    for arm, entry in scheme_entry.get("arms", {}).items():
        for regularizer, sizes in entry.get("budget", {}).items():
            for size, budgets in sizes.items():
                for budget in budgets:
                    cells.add((arm, str(size), regularizer, budget))
    return cells


def check_coverage(config, name, scheme, scheme_entry, problems):
    """The record must carry the whole declared sweep, not a subset of it."""
    declared = expected_cells(config)
    present = recorded_cells(scheme_entry)
    missing = sorted(declared - present)
    extra = sorted(present - declared)
    if missing:
        shown = ", ".join("/".join(cell) for cell in missing[:5])
        problems.append(f"{name}/{scheme}: {len(missing)} declared budget cells are absent from "
                        f"the record (e.g. {shown}); an incomplete sweep cannot pass the "
                        "canonical result gate")
    if extra:
        shown = ", ".join("/".join(cell) for cell in extra[:5])
        problems.append(f"{name}/{scheme}: {len(extra)} recorded cells are outside the declared "
                        f"sweep (e.g. {shown})")
    for arm in priceable_arms(config):
        sizes = {str(size) for size in config["basis_sizes_by_arm"][arm]}
        recorded = set(scheme_entry.get("arms", {}).get(arm, {}).get("sizes", {}))
        if sizes - recorded:
            problems.append(f"{name}/{scheme}/{arm}: no per-size record for basis sizes "
                            f"{sorted(sizes - recorded)}")


def check_conformance(record, config, problems):
    """Recompute whether the frozen continuity rule held, and hold the record to it.

    Derived from the per-instance comparison the record carries, never read back
    off its own summary flag: a record that failed the rule and said otherwise
    would pass a gate that trusted the flag. A run that failed it must say so.
    An explanation of the failure is not a pass, so the gate withholds the
    conforming stamp while the deviation stands.

    Returns the status so `main` can label its own result with it.
    """
    continuity = record.get("continuity_check")
    conformance = record.get("protocol_conformance")
    if conformance is None:
        problems.append("the record carries no protocol_conformance block, so it makes no "
                        "statement about whether it satisfied its own frozen continuity rule")
        return "unstated"
    if not continuity or not continuity.get("available"):
        if conformance.get("status") != "unverifiable":
            problems.append("no v2 comparison is available, so conformance is unverifiable, "
                            f"but the record claims {conformance.get('status')!r}")
        return conformance.get("status", "unverifiable")

    derived = all(row["v2_qualifying_arms"] == row["v3_ungrouped_qualifying_arms"]
                  for row in continuity["instances"].values())
    for name, row in continuity["instances"].items():
        if row.get("agree") != (row["v2_qualifying_arms"] == row["v3_ungrouped_qualifying_arms"]):
            problems.append(f"continuity_check/{name}: the recorded agree flag disagrees with "
                            "the qualifying arms recorded beside it")
    if bool(continuity.get("orderings_agree")) != derived:
        problems.append("continuity_check.orderings_agree disagrees with the per-instance "
                        "comparisons recorded beside it")
    if conformance.get("continuity_rule") != config["continuity_check"]["rule"]:
        problems.append("the record's stated continuity rule is not the frozen one")
    if conformance.get("continuity_rule_satisfied") is not derived:
        problems.append(f"the record states the continuity rule was satisfied="
                        f"{conformance.get('continuity_rule_satisfied')!r}, but the comparisons "
                        f"it carries give {derived!r}")
    expected_status = "conforming" if derived else "deviating"
    if conformance.get("status") != expected_status:
        problems.append(f"the record's protocol status is {conformance.get('status')!r} but its "
                        f"own continuity comparisons give {expected_status!r}")
    if not derived:
        deviation = conformance.get("deviation")
        if not isinstance(deviation, dict) or not deviation.get("cause"):
            problems.append("the continuity rule was not satisfied, so the record must carry a "
                            "deviation naming what failed and why; it does not")
    return expected_status


def main() -> int:
    if not RECORD.exists():
        print(f"FAIL missing record {RECORD}")
        return 1
    raw_config = CONFIG.read_bytes()
    config = json.loads(raw_config)
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    problems: list[str] = []

    if record.get("schema") != SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}, expected {SCHEMA!r}")
    if record.get("config_digest") != hashlib.sha256(raw_config).hexdigest():
        problems.append("config digest moved: the record was produced under a different config")
    if record.get("quantum_advantage_claim") is not False:
        problems.append("quantum_advantage_claim must be literally false")
    if record.get("evidence") != config["noise_model"]["evidence_label"]:
        problems.append(f"evidence must be {config['noise_model']['evidence_label']!r}, "
                        f"got {record.get('evidence')!r}")
    if record.get("cost_contract") != config["cost_model"]["contract"]:
        problems.append("cost contract does not match the frozen config")
    if record.get("covariance_treatment") != config["covariance_treatment"]["method"]:
        problems.append("the record's covariance treatment is not the declared one")
    if "provenance" not in record:
        problems.append("record is not stamped with execution provenance")
    for forbidden in ("phase_status", "phase_complete", "phase_completion"):
        if forbidden in record:
            problems.append(f"record carries a phase-completion claim ({forbidden})")
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1

    from run_phase16b_v3_feasibility import combine, instance_status, least_favourable

    target = float(config["target"]["value"])
    declared_replicas = int(config["noise_model"]["replicas"])
    if float(record.get("target", float("nan"))) != target:
        problems.append(f"record target {record.get('target')!r} does not match the frozen "
                        f"config target {target!r}")
    if int(record.get("replicas", -1)) != declared_replicas:
        problems.append(f"record replicas {record.get('replicas')!r} does not match the frozen "
                        f"config replicas {declared_replicas!r}")
    required = set(config["required_decision_instances"])
    incumbent = config["incumbent_arm"]
    readout_arms = set(config["covariance_treatment"]["applies_to"])
    primary = list(config["decision_rule"]["primary_schemes"])
    schemes = list(record.get("schemes", []))

    if "ungrouped" in primary:
        problems.append("the frozen config lists ungrouped as primary; this gate refuses a "
                        "verdict taken under the cost model the experiment replaces")
    for scheme in primary:
        if scheme not in schemes:
            problems.append(f"primary scheme {scheme} was not run")
    if set(schemes) - set(config["grouping_schemes"]):
        problems.append(f"the record runs undeclared schemes "
                        f"{sorted(set(schemes) - set(config['grouping_schemes']))}")
    # `ungrouped` is not a primary scheme, but the frozen continuity rule is
    # stated over it, so a record that skipped it cannot be held to that rule.
    unrun = sorted(set(config["grouping_schemes"]) - set(schemes))
    if unrun:
        problems.append(f"declared grouping schemes {unrun} were not run; the frozen continuity "
                        "rule is stated over the ungrouped scheme and cannot be checked without "
                        "every declared scheme present")

    present = {name for name, entry in record["instances"].items() if "skipped" not in entry}
    missing = sorted(required - present)
    if missing:
        problems.append(f"required instances absent from the record: {missing}; a partial run "
                        "cannot pass the canonical result gate")
    if record["decision"].get("verdict") == "INCOMPLETE":
        problems.append("recorded verdict is INCOMPLETE; the canonical result gate rejects "
                        "partial experiments unconditionally")

    declared_h = config["measured_before_freezing"]["hamiltonian_traceless_terms"]
    per_scheme_statuses = {scheme: {} for scheme in schemes}

    for name, entry in record["instances"].items():
        if "skipped" in entry:
            if name in required:
                problems.append(f"{name} is required but was skipped")
            continue
        checks = entry.get("checks", {})
        for flag in ("branch_condition", "enclosure_contains_spectrum"):
            if not checks.get(flag):
                problems.append(f"{name}: deterministic check {flag} failed")
        residual = checks.get("pencil_identity")
        if residual is None or residual > PENCIL_TOL:
            problems.append(f"{name}: pencil identity residual {residual} exceeds {PENCIL_TOL}")

        precondition = entry.get("reference_precondition", {})
        if name in required and not precondition.get("basis_state"):
            problems.append(f"{name}: the record does not show a computational-basis reference, "
                            "so the simulated-readout covariance treatment does not apply here")

        if name in required:
            admission = entry.get("admission")
            reference_scheme = (admission or {}).get("read_from_scheme")
            if not admission:
                problems.append(f"{name} is required but records no admission verification")
            elif reference_scheme not in entry.get("schemes", {}):
                problems.append(f"{name}: admission cites scheme {reference_scheme!r}, "
                                "which the record does not carry")
            else:
                arms = entry["schemes"][reference_scheme]["arms"]
                incumbent_seen = zero_noise_minimum(arms, [incumbent])
                candidate_seen = zero_noise_minimum(arms, config["exact_propagation_arms"])
                for label, derived, stored in (
                        ("incumbent", incumbent_seen,
                         admission.get("incumbent_zero_noise_error")),
                        ("candidate", candidate_seen,
                         admission.get("candidate_zero_noise_error"))):
                    if stored is None or not math.isclose(float(stored), derived,
                                                          rel_tol=1e-9, abs_tol=1e-15):
                        problems.append(
                            f"{name}: stored {label} zero-noise error {stored!r} disagrees with "
                            f"{derived!r} derived from the per-size records")
                derived_admits = incumbent_seen <= target and candidate_seen <= target
                if derived_admits != bool(admission.get("admits")):
                    problems.append(
                        f"{name}: admission flag is {admission.get('admits')!r} but the recorded "
                        f"zero-noise errors give {derived_admits!r}")
                if not derived_admits:
                    problems.append(
                        f"{name}: admission does not hold at execution (incumbent "
                        f"{incumbent_seen:.3e}, candidate {candidate_seen:.3e}, "
                        f"target {target:.1e})")

        absent = [scheme for scheme in schemes if scheme not in entry.get("schemes", {})]
        if absent and entry.get("status") != "INVALID":
            problems.append(f"{name}: the record declares schemes {schemes} but carries no "
                            f"results for {absent}")

        if entry.get("status") == "INVALID" or not entry.get("schemes"):
            for scheme in schemes:
                per_scheme_statuses[scheme][name] = "INVALID"
            if name in required:
                problems.append(f"{name} is required but records no per-scheme results")
            continue

        incumbent_settings = {}
        for scheme, scheme_entry in entry.get("schemes", {}).items():
            g_h = scheme_entry.get("hamiltonian_settings")
            if g_h is None:
                problems.append(f"{name}/{scheme}: no Hamiltonian setting count recorded")
                continue
            if entry["n_qubits"] == 8 and scheme in declared_h and g_h != declared_h[scheme]:
                problems.append(f"{name}/{scheme}: G_H is {g_h}, but the frozen config's "
                                f"measured table says {declared_h[scheme]}")
            arms = scheme_entry["arms"]
            for arm_name, arm in arms.items():
                for size_key, info in arm.get("sizes", {}).items():
                    expected = declared_settings(arm_name, int(size_key), g_h)
                    if expected is not None and info.get("settings") != expected:
                        problems.append(
                            f"{name}/{scheme}/{arm_name}@{size_key}: {info.get('settings')} "
                            f"settings, but the declared model gives {expected}")
                    if arm_name == incumbent:
                        incumbent_settings.setdefault(scheme, set()).add(info.get("settings"))
                    conditioning = info.get("moment_conditioning")
                    if conditioning is not None:
                        # H^p is Hermitian exactly; repeated MV products are not.
                        # The residual is relative to the operator's own scale.
                        if conditioning["hermiticity_residual"] > 1e-12:
                            problems.append(
                                f"{name}/{scheme}/{arm_name}@{size_key}: H^p carries imaginary "
                                f"Pauli coefficients at relative size "
                                f"{conditioning['hermiticity_residual']:.3e}")
                for report in distinct_verifications(arm):
                    if report["residual"] > report["tolerance"]:
                        problems.append(
                            f"{name}/{scheme}/{arm_name}: the dense Trotter step differs from "
                            f"trotter2_unitary by {report['residual']:.3e}, beyond its "
                            f"{report['tolerance']:.3e} tolerance")
                    allowed = max(report["reference_unitarity_defect"], ROUND_OFF_FLOOR)
                    if report["dense_unitarity_defect"] > allowed:
                        problems.append(
                            f"{name}/{scheme}/{arm_name}: the dense Trotter step's unitarity "
                            f"defect {report['dense_unitarity_defect']:.3e} exceeds the "
                            f"reference's {allowed:.3e}")
                for reg, sizes in arm.get("budget", {}).items():
                    for size_key, budgets in sizes.items():
                        for budget_key, stats in budgets.items():
                            where = f"{name}/{scheme}/{arm_name}/{reg}/{size_key}/{budget_key}"
                            if stats["replicas"] != declared_replicas:
                                problems.append(f"{where}: {stats['replicas']} replicas, "
                                                f"config declares {declared_replicas}")
                            if (stats["failures"] > declared_replicas // 2
                                    and stats["median"] < 1e30):
                                problems.append(f"{where}: {stats['failures']} failures but a "
                                                "finite median -- failed solves appear dropped")
                            budget = float(budget_key)
                            # A sampled setting is read a whole number of times,
                            # so the readout arms round; the priced arms do not.
                            if arm_name in readout_arms:
                                expected = float(max(1, round(budget / stats["settings"])))
                            else:
                                expected = budget / stats["settings"]
                            if not math.isclose(stats["shots_per_setting"], expected,
                                                rel_tol=1e-9):
                                problems.append(
                                    f"{where}: {stats['shots_per_setting']} shots per setting "
                                    f"against {stats['settings']} settings does not follow from "
                                    f"the {budget:.0e} budget ({expected} expected)")

            for label, report in scheme_entry.get("verifications", {}).items():
                if report.get("covariance_mismatches") != 0:
                    problems.append(f"{name}/{scheme}/{label}: "
                                    f"{report.get('covariance_mismatches')} within-class "
                                    "covariance identities failed")
                if report.get("covariance_pairs_checked") is None:
                    problems.append(f"{name}/{scheme}/{label}: no covariance verification")
                if scheme in ("qwc", "ungrouped") and not report.get("literal_words_checked"):
                    problems.append(f"{name}/{scheme}/{label}: a product-basis setting was not "
                                    "checked shot for shot against literal readouts")
            check_coverage(config, name, scheme, scheme_entry, problems)
            for arm_name in config["covariance_treatment"]["applies_to"]:
                if not any(key.startswith(f"{arm_name}@")
                           for key in scheme_entry.get("verifications", {})):
                    problems.append(f"{name}/{scheme}: {arm_name} is declared to use simulated "
                                    "group readouts but records no verification")

            audited = {key: value for key, value in scheme_entry.items() if key != "status"}
            status, evidence = instance_status(audited, config, target)
            if status != scheme_entry.get("status"):
                problems.append(f"{name}/{scheme}: recorded status "
                                f"{scheme_entry.get('status')!r} but the frozen rule gives "
                                f"{status!r}")
            per_scheme_statuses[scheme][name] = status
            if evidence.get("incumbent_censored") and status not in ("UNDETERMINED", "INVALID"):
                problems.append(f"{name}/{scheme}: the incumbent's cost is censored but the "
                                f"status is {status!r}; a missing control must not promote")

        base = incumbent_settings.get("ungrouped")
        for scheme in primary:
            grouped = incumbent_settings.get(scheme)
            if base and grouped and min(grouped) >= min(base):
                problems.append(f"{name}: the incumbent pays {min(grouped)} settings under "
                                f"{scheme} against {min(base)} ungrouped, so this record does "
                                "not price the grouping it claims to")

    verdicts = {}
    for scheme in schemes:
        verdicts[scheme] = combine(per_scheme_statuses[scheme], config)
        recorded = record["decision"]["per_scheme"].get(scheme, {}).get("verdict")
        if recorded != verdicts[scheme]:
            problems.append(f"{scheme}: recorded verdict {recorded!r} but the frozen rule "
                            f"gives {verdicts[scheme]!r}")
    primary_verdicts = {s: verdicts[s] for s in primary if s in verdicts}
    if not primary_verdicts:
        problems.append("no primary scheme produced a verdict")
        overall_scheme, overall = None, "INCOMPLETE"
    else:
        overall_scheme, overall = least_favourable(primary_verdicts)
    if overall != record["decision"].get("verdict"):
        problems.append(f"recorded verdict {record['decision'].get('verdict')!r} but the least "
                        f"favourable primary scheme gives {overall!r}")
    if overall_scheme != record["decision"].get("least_favourable_scheme"):
        problems.append(f"recorded least favourable scheme "
                        f"{record['decision'].get('least_favourable_scheme')!r} but the rule "
                        f"gives {overall_scheme!r}")

    conformance = check_conformance(record, config, problems)

    for scheme in schemes:
        marker = "primary" if scheme in primary else "baseline"
        print(f"  [{scheme}] {verdicts[scheme]:12} ({marker})  "
              + "  ".join(f"{k}={v}" for k, v in sorted(per_scheme_statuses[scheme].items())))
    print(f"  verdict: {overall} under {overall_scheme}")
    print(f"  protocol: {conformance}")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    if conformance != "conforming":
        # Everything the gate can check is sound, and the run still did not
        # satisfy a rule frozen before it ran. Stamping that "OK record is
        # complete" would launder the deviation, so the stamp says what it is.
        print(f"OK BUT {conformance.upper()}: the record is complete, the incumbent was priced "
              "with its grouping, covariance was verified, and the verdict follows the frozen "
              "decision rule -- but this run did NOT satisfy the frozen continuity rule, and "
              "its evidence is not that of a protocol-conforming run. See "
              "protocol_conformance.deviation in the record.")
        return 0
    print("OK record is complete, the incumbent was priced with its grouping, covariance was "
          "verified, and the verdict follows the frozen rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

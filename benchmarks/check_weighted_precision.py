"""Independent verifier for the weighted measurement & precision acquisition study.

Supports:
- ``--declaration-only``: validates the result-free declaration, historical
  digests, FCIDUMP hashes, seed uniqueness, schedule rules, and independent
  occupation-space anchors (without drawing any outcomes).
- ``--full-anchor``: additionally cross-checks native Pauli-bank word counts
  and exact expectation matrices against the independent occupation-space
  matrices.
- ``--report``: audits every committed Part A and Part B row, infinite-shot
  noisy bias anchor, shot billing, interval arithmetic, and primary criteria.
- ``--spot-replay``: replays designated replica-0 attempts from scratch and
  checks exact agreement with the committed record.
- ``--verify-history``: enforces strict Git chronology between declaration
  and result commits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_weighted_precision as producer  # noqa: E402
from benchmarks.experiment_artifacts import read_json  # noqa: E402
from benchmarks.noisy_readout_sim import (  # noqa: E402
    NoisyReadoutSimulator,
    ReadoutNoiseModel,
)
from clifford_qc.measurement import DeviceCard, NoiseGatePolicy  # noqa: E402
from clifford_qc.sampled_reference import PreparedSampledReference  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a: float, b: float, *, rel_tol: float = 1e-8, abs_tol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def z_score(level: float, family: int) -> float:
    return NormalDist().inv_cdf(1.0 - (1.0 - level) / (2.0 * family))


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_binomial_cdf(k: int, n: int, p: float) -> float:
    """Exact P(Binomial(n, p) <= k) using stdlib math.comb (no SciPy required)."""
    return float(sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1)))


def check_interval(
    entry: dict[str, Any],
    estimate: float,
    scale: float,
    delta: float,
    family: int,
    evidence: str,
) -> None:
    require(entry["evidence"] == evidence and close(entry["delta"], delta), "interval metadata mismatch")
    require(close(entry["estimate"], estimate), "interval center mismatch")
    radius = z_score(1.0 - delta, family) * scale
    require(
        close(entry["lower"], estimate - radius) and close(entry["upper"], estimate + radius),
        "interval radius mismatch",
    )


def validate_declaration_structure(d: dict[str, Any]) -> dict[str, Any]:
    """Validate internal declaration consistency, arm/case bindings, and seed uniqueness."""
    case_names = [c["name"] for c in d["cases"]]
    require(len(case_names) == len(set(case_names)) and len(case_names) >= 1, "duplicate or empty cases")
    case_set = set(case_names)

    a_keys = [a["key"] for a in d["part_a_arms"]]
    require(len(a_keys) == len(set(a_keys)), "duplicate Part A arms")
    for arm in d["part_a_arms"]:
        require(set(arm["cases"]).issubset(case_set), f"unknown case in arm {arm['key']}")
        if arm["grouping"] == "ritz_replan":
            require(arm["pilot_arm"] in a_keys, f"unknown pilot_arm in {arm['key']}")
            require(float(arm["overlap_weight"]) >= 0.0, "negative overlap_weight")

    b_keys = [a["key"] for a in d["part_b_arms"]]
    require(len(b_keys) == len(set(b_keys)), "duplicate Part B arms")
    reg_keys = [r["key"] for r in d["part_b_noise_regimes"]]
    require(len(reg_keys) == len(set(reg_keys)), "duplicate Part B noise regimes")

    card = DeviceCard.load(ROOT / d["device_card"])
    for regime in d["part_b_noise_regimes"]:
        ReadoutNoiseModel(
            p_1q=regime["p_1q"],
            p_2q=regime["p_2q"],
            routing_2q_multiplier=regime["routing_2q_multiplier"],
            p_readout=regime["p_readout"],
            label=regime["key"],
        )
        for tier_key in ("galic_tight_eps", "galic_moderate_eps"):
            NoiseGatePolicy(
                two_qubit_error=regime["p_2q"],
                max_relative_bias=regime[tier_key],
                routing_2q_multiplier=regime["routing_2q_multiplier"],
                device_card=card,
            )

    # Verify global seed uniqueness across every declared job and stream.
    jobs = producer.expected_job_keys(d)
    seen_seeds: set[int] = set()
    for part, ci, ai, reg_idx, rep in jobs:
        part_id = 0 if part == "A" else 1
        streams = (0, 1, 2) if (part == "A" and d["part_a_arms"][ai]["grouping"] == "ritz_replan") else (0, 1)
        for stream in streams:
            s = producer.derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, rep, stream)
            require(s not in seen_seeds, f"seed collision detected for {(part, ci, ai, reg_idx, rep, stream)}")
            seen_seeds.add(s)

    return {
        "cases": len(case_names),
        "part_a_jobs": sum(1 for j in jobs if j[0] == "A"),
        "part_b_jobs": sum(1 for j in jobs if j[0] == "B"),
        "total_jobs": len(jobs),
        "unique_seeds": len(seen_seeds),
    }


def anchors(d: dict[str, Any], *, full: bool = False) -> list[dict[str, Any]]:
    """Independently compute occupation-space anchors for every declared case."""
    from scipy.linalg import eigh

    from benchmarks.check_cross_domain_ratio import case_data
    from benchmarks.check_stretched_molecular_stop import independent_anchor

    systems: list[dict[str, Any]] = []
    for ci, case in enumerate(d["cases"]):
        if case["kind"] == "molecule_8q":
            indep = case_data({**case, "kind": "molecule"})
            s_mat, h_mat = indep["overlap"], indep["projected"]
            exact = float(eigh(h_mat, s_mat, eigvals_only=True)[0])
            diag = np.sqrt(np.real(np.diag(s_mat)))
            s_norm = s_mat / np.outer(diag, diag)
            spectrum = sorted(np.linalg.eigvalsh(0.5 * (s_norm + s_norm.conj().T)).tolist(), reverse=True)
            require(exact >= float(indep["ground"]) - 1e-9, f"{case['name']}: projected energy below singlet ground")
            require(len(spectrum) == int(case["expected_rank"]), f"{case['name']}: rank mismatch")
            require(spectrum[-1] > float(case["solver"]["tau_s"]), f"{case['name']}: weak mode below tau_s")
            anchor: dict[str, Any] = {
                "case": ci,
                "name": case["name"],
                "independent_target": exact,
                "exact_singlet_ground": float(indep["ground"]),
                "rank": len(spectrum),
                "overlap_spectrum": spectrum,
            }
        elif case["kind"] == "molecule_10q_stretched":
            _words, _h_occ, s_mat, h_mat, spec_asc, target, ground = independent_anchor(case)
            spectrum = sorted(spec_asc.tolist(), reverse=True)
            tau = float(case["solver"]["tau_s"])
            require(spectrum[-1] > tau and spectrum[-1] - tau < 0.001, "stress case does not straddle cutoff")
            require(len(spectrum) == int(case["expected_rank"]), "stretched rank mismatch")
            anchor = {
                "case": ci,
                "name": case["name"],
                "independent_target": float(target),
                "exact_singlet_ground": float(ground),
                "rank": len(spectrum),
                "overlap_spectrum": spectrum,
            }
        else:
            raise ValueError(f"unsupported case kind: {case['kind']!r}")

        if full:
            h, generators, state = producer.build_case_inputs(case)
            sim = NoisyReadoutSimulator(state, int(case["n"]), ReadoutNoiseModel.ideal())
            ref = PreparedSampledReference(int(case["n"]), f"{case['name']}-anchor", sim.draw)
            prov = producer.build_arm_provider(
                ref, h, generators, int(case["n"]), block_size=2, grouping="sorted_insertion"
            )
            require(len(prov.session.words) == int(case["expected_words"]), f"{case['name']}: word count mismatch")
            exps = sim.exact_word_expectations(prov.plan.settings)
            native_e, native_rank, native_spec = producer.exact_projected_from_expectations(
                prov, exps, case["solver"]
            )
            require(close(native_e, anchor["independent_target"], abs_tol=1e-9), f"{case['name']}: target mismatch")
            require(native_rank == anchor["rank"], f"{case['name']}: native rank mismatch")
            require(
                bool(np.allclose(native_spec, anchor["overlap_spectrum"], atol=1e-9, rtol=0.0)),
                f"{case['name']}: overlap spectrum mismatch",
            )
            anchor["words"] = len(prov.session.words)
        systems.append(anchor)
    return systems


def check_row(
    row: dict[str, Any],
    d: dict[str, Any],
    case: dict[str, Any],
    system: dict[str, Any],
) -> None:
    part = row["part"]
    part_id = 0 if part == "A" else 1
    ci, ai, reg_idx, rep = row["case"], row["arm_index"], row["regime_index"], row["replica"]
    require(
        row["acq_seed"] == producer.derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, rep, 0),
        "acq_seed mismatch",
    )
    require(
        row["boot_seed"] == producer.derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, rep, 1),
        "boot_seed mismatch",
    )
    is_replan = part == "A" and d["part_a_arms"][ai]["grouping"] == "ritz_replan"
    if is_replan:
        require(
            row["pilot_seed"] == producer.derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, rep, 2),
            "pilot_seed mismatch",
        )
        require(row["pilot_shots"] >= 0, "negative pilot_shots")
    else:
        require(row["pilot_seed"] is None and row["pilot_shots"] == 0, "unexpected pilot on non-replan arm")

    require(row["total_shots"] == row["pilot_shots"] + row["acquisition_shots"], "shot sum mismatch")
    require(row["device_seconds"] >= 0.0 and math.isfinite(row["device_seconds"]), "invalid device_seconds")

    if row["failure"] is not None:
        require(row["status"] == "failed" and row["estimate"] is None, "invalid failure row")
        return

    totals = case["cumulative_total_shots"]
    family = len(totals)
    looks = row["looks"]
    require(1 <= len(looks) <= family, "invalid look count")
    require(is_sha256(row["final_fingerprint"]), "invalid final_fingerprint")
    if is_replan:
        require(is_sha256(row["pilot_fingerprint"]), "invalid pilot_fingerprint")
    last = looks[-1]
    require(last["stopped"] and close(last["energy"], row["estimate"]) and close(last["std_error"], row["std_error"]), "terminal look mismatch")
    require(row["status"] in ("target_met", "schedule_exhausted", "deterministic"), "unknown status")

    # Verify rank guard metadata on every look.
    streak = 0
    target_se = float(d["target_std_error_Ha"])
    width_guard = part == "A" and bool(d["part_a_arms"][ai]["width_guard"])
    for idx, look in enumerate(looks):
        require(look["look"] == idx, "look index mismatch")
        guard = look.get("rank_guard")
        require(guard is not None and guard["required_rank"] == system["rank"], "missing/bad rank_guard")
        resolved = bool(guard["resolved"])
        streak = streak + 1 if resolved else 0
        rank_ok = streak >= int(d["rank_policy"]["consecutive_looks"])
        require(guard["consecutive_resolved"] == streak and guard["passed"] == rank_ok, "rank streak mismatch")
        se_ok = float(look["std_error"]) <= target_se
        width_ok = True
        if width_guard and se_ok and rank_ok:
            require("bootstrap_width_passed" in look, "missing width guard check")
            width_ok = bool(look["bootstrap_width_passed"])
        if idx < len(looks) - 1:
            require(not look["stopped"] and not (se_ok and rank_ok and width_ok), "missed earlier stop")

    if row["status"] == "target_met":
        require(float(last["std_error"]) <= target_se and bool(last["rank_guard"]["passed"]), "premature target_met")
    elif row["status"] == "schedule_exhausted":
        require(len(looks) == family, "premature schedule_exhausted")

    # Verify delta & bootstrap interval formulas.
    delta = float(d["delta"])
    naive = row["naive_interval"]
    scale = (naive["upper"] - naive["lower"]) / (2.0 * z_score(1.0 - delta, 1))
    require(math.isfinite(scale) and scale >= 0.0, "invalid delta scale")
    check_interval(naive, row["estimate"], scale, delta, 1, "asymptotic")
    check_interval(row["delta_interval"], row["estimate"], scale, delta, family, "asymptotic")
    if row["bootstrap_interval"] is not None:
        spread = float(last["bootstrap_spread"])
        check_interval(row["bootstrap_interval"], row["estimate"], spread, delta, family, "heuristic")
    else:
        require("bootstrap_failure" in last, "missing bootstrap_interval without bootstrap_failure")


def summarize(record: dict[str, Any], d: dict[str, Any], systems: list[dict[str, Any]]) -> dict[str, Any]:
    target = float(d["target_std_error_Ha"])
    part_a_summary: list[dict[str, Any]] = []
    for ci, case in enumerate(d["cases"]):
        exact = float(systems[ci]["independent_target"])
        rank = int(systems[ci]["rank"])
        for arm in d["part_a_arms"]:
            if case["name"] not in arm["cases"]:
                continue
            rows = [r for r in record["rows"] if r["part"] == "A" and r["case"] == ci and r["arm"] == arm["key"]]
            n = len(rows)
            succ = [r for r in rows if r["failure"] is None]
            met = [r for r in succ if r["status"] == "target_met"]
            errors = [r["estimate"] - exact for r in succ]
            true_1s = sum(abs(r["estimate"] - exact) <= target for r in met)
            false_1s = len(met) - true_1s
            within_2s = sum(abs(r["estimate"] - exact) <= 2.0 * target for r in met)
            boot_cov = sum(
                r["bootstrap_interval"] is not None
                and r["bootstrap_interval"]["lower"] <= exact <= r["bootstrap_interval"]["upper"]
                for r in succ
            )
            delta_cov = sum(
                r["delta_interval"] is not None
                and r["delta_interval"]["lower"] <= exact <= r["delta_interval"]["upper"]
                for r in succ
            )
            part_a_summary.append(
                {
                    "case": case["name"],
                    "arm": arm["key"],
                    "attempts": n,
                    "target_met": len(met),
                    "true_precision_stops": true_1s,
                    "false_precision_stops": false_1s,
                    "within_2sigma_stops": within_2s,
                    "reduced_rank_stops": sum(r["looks"][-1]["retained_rank"] < rank for r in met),
                    "exhausted": sum(r["status"] == "schedule_exhausted" for r in rows),
                    "failed": n - len(succ),
                    "median_total_shots": float(np.median([r["total_shots"] for r in rows])),
                    "mean_total_shots": float(np.mean([r["total_shots"] for r in rows])),
                    "rmse_mHa": float(np.sqrt(np.mean(np.square(errors))) * 1000.0) if errors else None,
                    "bias_mHa": float(np.mean(errors) * 1000.0) if errors else None,
                    "boot95_covered": boot_cov,
                    "boot95_p_lower": exact_binomial_cdf(boot_cov, n, 0.95),
                    "delta95_covered": delta_cov,
                    "median_r_hat_ritz": (
                        float(np.median([r["r_hat_ritz"] for r in succ if r["r_hat_ritz"] is not None]))
                        if any(r["r_hat_ritz"] is not None for r in succ)
                        else None
                    ),
                    "median_r_emp_ritz": (
                        float(np.median([r["r_emp_ritz"] for r in succ if r["r_emp_ritz"] is not None]))
                        if any(r["r_emp_ritz"] is not None for r in succ)
                        else None
                    ),
                }
            )

    anchor_by_key = {
        (a["case"], a["regime"], a["arm"]): a for a in record.get("noisy_anchors", [])
    }
    part_b_summary: list[dict[str, Any]] = []
    for ci, case in enumerate(d["cases"]):
        if "B" not in case["parts"]:
            continue
        exact = float(systems[ci]["independent_target"])
        for regime in d["part_b_noise_regimes"]:
            for arm in d["part_b_arms"]:
                rows = [
                    r
                    for r in record["rows"]
                    if r["part"] == "B"
                    and r["case"] == ci
                    and r["regime"] == regime["key"]
                    and r["arm"] == arm["key"]
                ]
                n = len(rows)
                succ = [r for r in rows if r["failure"] is None]
                met = [r for r in succ if r["status"] == "target_met"]
                errors = [r["estimate"] - exact for r in succ]
                true_1s = sum(abs(r["estimate"] - exact) <= target for r in met)
                false_1s = len(met) - true_1s
                na = anchor_by_key.get((ci, regime["key"], arm["key"]), {})
                part_b_summary.append(
                    {
                        "case": case["name"],
                        "regime": regime["key"],
                        "arm": arm["key"],
                        "attempts": n,
                        "target_met": len(met),
                        "true_precision_stops": true_1s,
                        "false_precision_stops": false_1s,
                        "exact_inf_bias_mHa": (
                            None if "bias_Ha" not in na else float(na["bias_Ha"] * 1000.0)
                        ),
                        "emp_bias_mHa": float(np.mean(errors) * 1000.0) if errors else None,
                        "rmse_mHa": float(np.sqrt(np.mean(np.square(errors))) * 1000.0) if errors else None,
                        "median_total_shots": float(np.median([r["total_shots"] for r in rows])),
                        "median_device_seconds": float(np.median([r["device_seconds"] for r in rows])),
                        "cz_max": na.get("cz_max"),
                    }
                )
    return {"part_a": part_a_summary, "part_b": part_b_summary}


def check_record(record: dict[str, Any], *, verify_history: bool = False, full: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    d = producer.declaration()
    validate_declaration_structure(d)
    require(record["schema"] == producer.SCHEMA and record["declaration"] == d, "declaration mismatch")
    require(record["environment"] == d["environment"], "environment mismatch")
    commit = record["declaration_commit"]
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "invalid declaration_commit")
    decl_bytes = (ROOT / producer.DECLARATION).read_bytes()
    require(
        record["declaration_sha256"] == producer.digest(decl_bytes)
        and producer.git("show", f"{commit}:{producer.DECLARATION}") == decl_bytes,
        "declaration commit binding mismatch",
    )
    require(
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{producer.RESULT}"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        != 0,
        "result already existed at declaration commit",
    )
    require(record["source_sha256"] == producer.manifest(commit, d), "source manifest mismatch")
    if verify_history:
        final = producer.git("log", "-1", "--format=%H", "--", producer.RESULT).decode().strip()
        require(
            bool(final)
            and final != commit
            and subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, final], cwd=ROOT
            ).returncode
            == 0,
            "result is not in a separate descendant commit",
        )
        require(
            producer.git("show", f"{final}:{producer.RESULT}") == (ROOT / producer.RESULT).read_bytes(),
            "uncommitted result bytes",
        )

    systems = anchors(d, full=full)
    for case, anchor, actual in zip(d["cases"], systems, record["systems"], strict=True):
        require(
            close(actual["exact_energy_Ha"], anchor["independent_target"], abs_tol=1e-9)
            and actual["exact_rank"] == anchor["rank"]
            and actual["words"] == int(case["expected_words"]),
            f"system anchor mismatch for {case['name']}",
        )
        require(
            bool(np.allclose(actual["overlap_spectrum"], anchor["overlap_spectrum"], atol=1e-9, rtol=0.0)),
            f"overlap spectrum mismatch for {case['name']}",
        )

    expected = set(producer.expected_job_keys(d))
    seen: set[tuple[str, int, int, int, int]] = set()
    for row in record["rows"]:
        key = (row["part"], row["case"], row["arm_index"], row["regime_index"], row["replica"])
        require(key in expected and key not in seen, f"unexpected or duplicate row: {key}")
        seen.add(key)
        check_row(row, d, d["cases"][row["case"]], systems[row["case"]])
    require(seen == expected, "incomplete record")
    return d, systems


def spot_replay(record: dict[str, Any], d: dict[str, Any]) -> int:
    """Replay replica 0 of beh2_8q Part A (si_fc, ritz_si_fc) & Part B (depol2q_p010_r3 si_fc_unconstrained)."""
    case = d["cases"][0]
    n = int(case["n"])
    h, generators, state = producer.build_case_inputs(case)
    replayed = 0

    ideal_sim = NoisyReadoutSimulator(state, n, ReadoutNoiseModel.ideal())
    ideal_ref = PreparedSampledReference(n, f"{case['name']}-ideal", ideal_sim.draw)
    si_fc_idx = next(i for i, a in enumerate(d["part_a_arms"]) if a["key"] == "si_fc")
    ritz_idx = next(i for i, a in enumerate(d["part_a_arms"]) if a["key"] == "ritz_si_fc")

    prov_si_fc = producer.build_arm_provider(
        ideal_ref, h, generators, n, block_size=n, grouping="sorted_insertion"
    )
    for ai, is_replan in ((si_fc_idx, False), (ritz_idx, True)):
        recorded = next(
            r
            for r in record["rows"]
            if r["part"] == "A" and r["case"] == 0 and r["arm_index"] == ai and r["replica"] == 0
        )
        fresh = producer.run_single_attempt(
            prov_si_fc,
            case,
            d,
            acq_seed=recorded["acq_seed"],
            boot_seed=recorded["boot_seed"],
            pilot_seed=recorded["pilot_seed"],
            replan_spec=d["part_a_arms"][ai] if is_replan else None,
        )
        require(
            fresh["status"] == recorded["status"]
            and fresh["total_shots"] == recorded["total_shots"]
            and fresh["final_fingerprint"] == recorded["final_fingerprint"]
            and (
                recorded["estimate"] is None
                or close(fresh["estimate"], recorded["estimate"], abs_tol=1e-9)
            ),
            f"spot replay mismatch for Part A arm {d['part_a_arms'][ai]['key']}",
        )
        replayed += 1

    reg_idx = next(i for i, r in enumerate(d["part_b_noise_regimes"]) if r["key"] == "depol2q_p010_r3")
    regime = d["part_b_noise_regimes"][reg_idx]
    b_idx = next(i for i, a in enumerate(d["part_b_arms"]) if a["key"] == "si_fc_unconstrained")
    nsim = NoisyReadoutSimulator(
        state,
        n,
        ReadoutNoiseModel(
            p_1q=regime["p_1q"],
            p_2q=regime["p_2q"],
            routing_2q_multiplier=regime["routing_2q_multiplier"],
            p_readout=regime["p_readout"],
            label=regime["key"],
        ),
    )
    nref = PreparedSampledReference(n, f"{case['name']}-{regime['key']}", nsim.draw)
    prov_b = producer.build_arm_provider(
        nref, h, generators, n, block_size=n, grouping="sorted_insertion"
    )
    recorded_b = next(
        r
        for r in record["rows"]
        if r["part"] == "B"
        and r["case"] == 0
        and r["regime_index"] == reg_idx
        and r["arm_index"] == b_idx
        and r["replica"] == 0
    )
    fresh_b = producer.run_single_attempt(
        prov_b, case, d, acq_seed=recorded_b["acq_seed"], boot_seed=recorded_b["boot_seed"]
    )
    require(
        fresh_b["status"] == recorded_b["status"]
        and fresh_b["total_shots"] == recorded_b["total_shots"]
        and fresh_b["final_fingerprint"] == recorded_b["final_fingerprint"]
        and (
            recorded_b["estimate"] is None
            or close(fresh_b["estimate"], recorded_b["estimate"], abs_tol=1e-9)
        ),
        "spot replay mismatch for Part B depol2q_p010_r3 si_fc_unconstrained",
    )
    replayed += 1
    return replayed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration-only", action="store_true")
    parser.add_argument("--full-anchor", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--spot-replay", action="store_true")
    parser.add_argument("--verify-history", action="store_true")
    parser.add_argument("--record", type=Path, default=ROOT / producer.RESULT)
    args = parser.parse_args(argv)

    if args.declaration_only:
        d = producer.declaration()
        stats = validate_declaration_structure(d)
        systems = anchors(d, full=args.full_anchor)
        print(
            json.dumps(
                {
                    "status": "declaration_ok",
                    "result_exists": (ROOT / producer.RESULT).exists(),
                    "stats": stats,
                    "anchors": systems,
                },
                indent=2,
            )
        )
        return 0

    record = read_json(args.record)
    d, systems = check_record(record, verify_history=args.verify_history, full=args.full_anchor)
    out: dict[str, Any] = {
        "status": "record_ok",
        "rows_verified": len(record["rows"]),
        "noisy_anchors_verified": len(record.get("noisy_anchors", [])),
    }
    if args.spot_replay:
        out["spot_replays_verified"] = spot_replay(record, d)
    if args.report:
        out["summary"] = summarize(record, d, systems)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute the frozen Phase 19 fixed-reference energy measurement campaign.

Default invocation is outcome-free preflight. After landing and validating this
implementation, --execute consumes the one local execution claim, samples every
frozen endpoint/audit once, and writes a new record without overwriting one.
Tests call run_bank only with undeclared toy fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

from benchmarks import check_phase19_energy_comparison as independent
from benchmarks import phase19_structure as structure
from benchmarks.check_phase19_preregistration import load_config, read_json, static_problems
from clifford_qc.measurement.confidence import empirical_bernstein_radius
from clifford_qc.measurement.cost import (
    DeviceCard, SettingResources, cost_schedule, inflate_shots_for_fidelity, setting_fidelity,
)
from clifford_qc.reproducibility import execution_provenance, stamp_record


def statevector_circuit(n, ops, index):
    """Bit-index pair updates, independent of the checker's dense tensor oracle."""
    state = np.zeros(1 << n, dtype=complex)
    state[index] = 1
    indices = np.arange(1 << n)
    for op in ops:
        if op[0] == "cx":
            _, control, target = op
            mask = 1 << (n - 1 - target)
            left = indices[((indices & (1 << (n - 1 - control))) != 0) &
                           ((indices & mask) == 0)]
            right = left | mask
            state[left], state[right] = state[right].copy(), state[left].copy()
            continue
        if op[0] == "rz":
            _, angle, q = op
            matrix = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
        else:
            name, q = op
            matrix = {"h": np.array([[1, 1], [1, -1]]) / math.sqrt(2),
                      "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j])}[name]
        mask = 1 << (n - 1 - q)
        left = indices[(indices & mask) == 0]
        right = left | mask
        a, b = state[left].copy(), state[right].copy()
        state[left] = matrix[0, 0] * a + matrix[0, 1] * b
        state[right] = matrix[1, 0] * a + matrix[1, 1] * b
    return state


def moments(weights, values):
    anchor = float(values[np.flatnonzero(weights)[0]])
    differences = values - anchor
    mean_difference = math.fsum(float(p) * float(y) for p, y in zip(weights, differences))
    mean = anchor + mean_difference
    variance = math.fsum(float(p) * (float(y) - mean_difference) ** 2
                         for p, y in zip(weights, differences))
    return mean, variance


def sample_statistics(histogram, values, shots):
    """Centered sample moments retain within-shot QWC covariance."""
    indices = [outcome for outcome, _ in histogram]
    counts = [count for _, count in histogram]
    if sum(counts) != shots or shots < 2:
        raise ValueError("histogram does not match its setting shot count")
    anchor = float(values[indices[0]])
    differences = [float(values[index]) - anchor for index in indices]
    shift = math.fsum(count * delta for count, delta in zip(counts, differences)) / shots
    mean = anchor + shift
    variance = math.fsum(count * (delta - shift) ** 2
                         for count, delta in zip(counts, differences)) / (shots - 1)
    return mean, variance


def sample_histogram(probabilities, root, indices, shots):
    sequence = np.random.SeedSequence(root, spawn_key=indices)
    counts = np.random.Generator(np.random.PCG64(sequence)).multinomial(shots, probabilities)
    return [[index, int(count)] for index, count in enumerate(counts) if count]


def device_reports(arm, endpoint, cards, n, epsilon):
    resources = [SettingResources.from_mapping(item["resources"]) for item in arm["settings"]]
    reports = []
    for card in cards:
        fidelities = [setting_fidelity(card, item, n) for item in resources]
        admissible = all(value >= card.fidelity_floor for value in fidelities)
        raw, ledger = None, None
        if endpoint is not None:
            raw = inflate_shots_for_fidelity(card, resources, endpoint["shots"], n)
            ledger = cost_schedule(card, resources, raw, n_qubits=n,
                                   evidence_tier="exact", epsilon=epsilon)
        reports.append({"card": card.name, "card_sha256": card.sha256,
            "admissible": admissible, "setting_fidelities": fidelities,
            "status": "inadmissible" if not admissible else
                      "right_censored" if endpoint is None else "priced",
            "raw_shots": raw, "total_raw_shots": None if raw is None else sum(raw),
            "C_time_epsilon_us": None if ledger is None else ledger["accuracy"]["C_time_epsilon_us"],
            "epsilon_hartree": epsilon})
    return reports


def shot_decision(qwc, clique, config):
    factor = config["acceptance"]["material_shot_reduction_factor"]
    maximum = config["protocol"]["effective_shot_endpoints"][-1]
    if qwc is not None and clique is not None:
        return "MATERIAL_REDUCTION" if factor * clique <= qwc else "NO_MATERIAL_REDUCTION"
    if qwc is not None:
        return "NO_MATERIAL_REDUCTION"
    if clique is not None and factor * clique <= maximum:
        return "MATERIAL_REDUCTION_CENSORED_BASELINE"
    return "INDETERMINATE"


def run_bank(bank, config, cards, system_index):
    """Sample one bank. The CLI supplies frozen banks; unit tests supply toys."""
    started = time.perf_counter()
    protocol = config["protocol"]
    oracle, dense_laws = independent.dense_oracle(bank, protocol["probability_policy"])
    blocking = independent.oracle_failures(oracle)
    record = {"structural_bank": bank, "oracle": oracle, "arms": [],
              "blocking_failures": blocking}
    if blocking:
        return record, time.perf_counter() - started, 0.0
    coefficients = {code: float.fromhex(value) for code, value in bank["coefficients_hex"]}
    offset = coefficients.get(0, 0.0)
    reference = oracle["reference"]["energy_hartree"]
    prepared = []
    for ai, arm in enumerate(bank["arms"]):
        laws, diagonals = [], []
        for si, (setting, ops) in enumerate(zip(arm["settings"], independent.measurement_ops(bank, arm))):
            state = statevector_circuit(bank["n_qubits"], ops, independent.reference_index(bank))
            raw = np.abs(state) ** 2
            diagonal = independent.diagonal_contribution(bank, arm, setting)
            blocking += independent.law_failures(raw, dense_laws[ai][si], arm["name"], si)
            try:
                probabilities, correction = independent.corrected_probabilities(raw,
                    protocol["probability_policy"])
                mean, variance = moments(probabilities, diagonal)
                law = {"raw_probabilities": independent.json_probabilities(raw),
                       "probabilities": probabilities.tolist(), "correction": correction,
                       "mean": mean, "variance": variance, "failure": None}
            except ValueError as exc:
                law = {"raw_probabilities": independent.json_probabilities(raw),
                       "probabilities": None, "correction": None,
                       "mean": None, "variance": None, "failure": str(exc)}
                blocking.append(f"{arm['name']}/{si}: {exc}")
            laws.append(law)
            diagonals.append(diagonal)
        error = (None if any(law["mean"] is None for law in laws) else
                 abs(math.fsum([offset, *(law["mean"] for law in laws)]) - reference))
        if error is not None and error > protocol["numerical_bias_allowance_hartree"]:
            blocking.append(f"{arm['name']}: reference law mean exceeds numerical allowance")
        prepared.append((laws, diagonals, error))
    compilation_seconds = time.perf_counter() - started
    if blocking:
        record["arms"] = [independent.blocked_arm(arm["name"], laws, error)
                          for arm, (laws, _, error) in zip(bank["arms"], prepared)]
        return record, compilation_seconds, 0.0
    sampling_started = time.perf_counter()
    for ai, (arm, (laws, diagonals, law_error)) in enumerate(zip(bank["arms"], prepared)):
        bounds = [item["max_abs_value"] for item in arm["settings"]]
        delta = protocol["confidence"]["delta"] / (protocol["confidence"]["family"] *
            protocol["confidence"]["rounds"] * len(bounds))
        endpoints = []
        for ei, budget in enumerate(protocol["effective_shot_endpoints"]):
            shots = list(structure.coefficient_range_schedule(bounds, budget))
            histograms = [sample_histogram(law["probabilities"], protocol["seeds"]["headline"],
                (system_index, ai, ei, si), count) for si, (law, count) in enumerate(zip(laws, shots))]
            stats = [sample_statistics(hist, diagonal, count)
                     for hist, diagonal, count in zip(histograms, diagonals, shots)]
            radii = [empirical_bernstein_radius(variance, count, delta, 2 * bound)
                     for (_, variance), count, bound in zip(stats, shots, bounds)]
            estimate = math.fsum([offset, *(mean for mean, _ in stats)])
            stochastic = math.fsum(radii)
            endpoints.append({"effective_shots": budget, "shots": shots,
                "histograms": histograms, "setting_means": [mean for mean, _ in stats],
                "setting_sample_variances": [variance for _, variance in stats],
                "setting_radii": radii, "estimate": estimate,
                "exact_reference_error": abs(estimate - reference),
                "stochastic_radius": stochastic,
                "total_radius": stochastic + protocol["numerical_bias_allowance_hartree"]})
        certified = next((item for item in endpoints
                          if item["total_radius"] <= protocol["accuracy_hartree"]), None)
        audit_shots = list(structure.coefficient_range_schedule(bounds,
            protocol["covariance_audit"]["effective_shots_per_replica"]))
        predicted = math.fsum(law["variance"] / count for law, count in zip(laws, audit_shots))
        estimates = []
        audit_digest = hashlib.sha256()
        for ri in range(protocol["covariance_audit"]["replicas"]):
            histograms = [sample_histogram(law["probabilities"], protocol["seeds"]["covariance_audit"],
                (system_index, ai, ri, si), count) for si, (law, count) in enumerate(zip(laws, audit_shots))]
            audit_digest.update(json.dumps(histograms, separators=(",", ":")).encode())
            estimates.append(math.fsum([offset, *(sample_statistics(hist, diagonal, count)[0]
                for hist, diagonal, count in zip(histograms, diagonals, audit_shots))]))
        centered = np.asarray(estimates) - estimates[0]
        average = math.fsum(float(value) for value in centered) / len(estimates)
        empirical = math.fsum((float(value) - average) ** 2 for value in centered) / (len(estimates) - 1)
        ratio = None if predicted == 0 else empirical / predicted
        low, high = protocol["covariance_audit"]["acceptable_empirical_to_predicted_interval"]
        passed = empirical == 0 if predicted == 0 else low <= ratio <= high
        if not passed:
            blocking.append(f"{arm['name']}: covariance audit failed")
        audit = {"shots": audit_shots, "estimates": estimates, "predicted_variance": predicted,
                 "empirical_variance": empirical, "ratio": ratio, "passed": passed,
                 "histograms_sha256": audit_digest.hexdigest()}
        record["arms"].append({"name": arm["name"], "laws": laws,
            "exact_law_reference_error": law_error, "sampling_status": "complete", "endpoints": endpoints,
            "certified_endpoint": None if certified is None else certified["effective_shots"],
            "right_censored": certified is None, "covariance_audit": audit,
            "device_reports": device_reports(arm, certified, cards, bank["n_qubits"],
                                             protocol["accuracy_hartree"])})
    return record, compilation_seconds, time.perf_counter() - sampling_started


def assemble_record(config, banks, compilation_seconds, sampling_seconds, provenance):
    invalid = any(bank["blocking_failures"] for bank in banks)
    headline = sum(sum(sum(endpoint["shots"]) for endpoint in arm["endpoints"])
                   for bank in banks for arm in bank["arms"])
    audit = sum(sum(arm["covariance_audit"]["shots"]) * len(arm["covariance_audit"]["estimates"])
                for bank in banks for arm in bank["arms"] if arm["covariance_audit"] is not None)
    return {"schema": independent.SCHEMA,
            "declaration_sha256": independent.EXPECTED_CONFIG_SHA256,
            "banks": banks, "verdict": "INVALID" if invalid else "VALID",
            "shot_decisions": [None if invalid else shot_decision(
                bank["arms"][0]["certified_endpoint"], bank["arms"][1]["certified_endpoint"], config)
                for bank in banks],
            "accounting": {"headline_effective_shots": headline, "audit_effective_shots": audit,
                           "compilation_seconds": compilation_seconds,
                           "sampling_seconds": sampling_seconds},
            "provenance": provenance}


def execution_preflight(root=ROOT):
    config = load_config(root / structure.CONFIG_PATH)
    problems = static_problems(config, root)
    if problems:
        raise ValueError("; ".join(problems))
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True,
                              check=True).stdout.strip()
    if git("rev-parse", "--is-shallow-repository") != "false":
        raise ValueError("execution requires full git history")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("execution requires a clean committed tree")
    commit = git("rev-parse", "HEAD")
    if commit == independent.MERGE:
        raise ValueError("execution must follow the preregistration merge")
    git("merge-base", "--is-ancestor", independent.MERGE, commit)
    if (root / structure.RECORD_PATH).exists() or git("log", "--all", "--format=%H", "--",
                                                   structure.RECORD_PATH):
        raise ValueError("the Phase 19 campaign record already exists in this history")
    sources = [structure.file_binding(path, root) for path in independent.SOURCES]
    for item in sources:
        if git("rev-parse", f"{commit}:{item['path']}") != item["git_blob_sha1"]:
            raise ValueError(f"uncommitted execution source: {item['path']}")
    provenance = execution_provenance()
    # The actual source revision wins over CI/environment SHA aliases.
    provenance.update(git_sha=commit, git_commit=commit, git_dirty=False,
                      preregistration_merge_commit=independent.MERGE, source_files=sources)
    reference = read_json(root / config["lineage"]["environment_reference"],
                          result_free=False)["provenance"]
    problems = independent.environment_problems(provenance, reference)
    if problems:
        raise ValueError("; ".join(problems))
    common = Path(git("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = root / common
    claim = common / f"phase19-{independent.EXPECTED_CONFIG_SHA256}.execution.json"
    if claim.exists():
        raise ValueError(f"the one execution was already claimed: {claim}")
    return config, provenance, claim


def claim_execution(path, provenance):
    """Exclusive, persistent claim shared by all worktrees, including failures."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"status": "claimed", "provenance": provenance}, handle, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the one frozen execution")
    args = parser.parse_args(argv)
    claim = None
    claimed = False
    records = []
    try:
        if not args.execute:
            config = load_config()
            problems = static_problems(config)
            if problems:
                raise ValueError("; ".join(problems))
            print("OK: outcome-free Phase 19 preflight; use --execute from the later clean pinned tree")
            return 0
        config, provenance, claim = execution_preflight()
        # Environment stamping is checked before consuming the claim or any outcomes.
        provenance = stamp_record({}, provenance)["provenance"]
        claim_execution(claim, provenance)
        claimed = True
        cards = [DeviceCard.from_dict(read_json(ROOT / item["path"]))
                 for item in config["protocol"]["device_cards"]]
        compilation, sampling = 0.0, 0.0
        for index, descriptor in enumerate(config["bank_manifests"]):
            bank = read_json(ROOT / descriptor["path"])
            record, built, sampled = run_bank(bank, config, cards, index)
            records.append(record)
            compilation += built
            sampling += sampled
        record = assemble_record(config, records, compilation, sampling, provenance)
        with independent.RECORD.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
        claim.write_text(json.dumps({"status": "completed", "record": structure.RECORD_PATH,
                                     "provenance": provenance}, allow_nan=False), encoding="utf-8")
        print(f"Wrote {structure.RECORD_PATH}: {record['verdict']}")
        return 0 if record["verdict"] == "VALID" else 1
    except Exception as exc:
        # Unexpected implementation errors also consume the execution. Preserve
        # their diagnostics rather than leaving a claimed run indistinguishable
        # from one still running. KeyboardInterrupt retains the original claim.
        if claimed:
            # Never release a consumed claim or substitute a seed after failure.
            failure = json.loads(claim.read_text(encoding="utf-8"))
            failure.update(status="failed", error=str(exc), completed_banks=records)
            claim.write_text(json.dumps(failure, allow_nan=False), encoding="utf-8")
        print(f"Phase 19 execution refused/failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

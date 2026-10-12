"""Independently gate the one frozen Phase 19 measurement record.

No producer functions are imported. Explicit tensor gates reconstruct the
operator and reference laws; stored joint histograms and seeded audit replays
reconstruct statistics, confidence crossings, decisions and card prices.
Without a committed record this command checks only the declaration.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

from benchmarks import phase19_structure as structure
from benchmarks.check_phase19_preregistration import (
    EXPECTED_CONFIG_SHA256, _compare, load_config, read_json, static_problems,
    temporal_problems,
)
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.cost import DeviceCard
from clifford_qc.qasm3 import lower_rotor
from clifford_qc.reproducibility import sampling_stream_mismatch

SCHEMA = "clifford_qc.phase19_energy_comparison.v1"
MERGE = "5f0df477dbcccc5102256a6f1e85ebce923f522a"
RECORD = ROOT / structure.RECORD_PATH
SOURCES = (
    "benchmarks/run_phase19_energy_comparison.py",
    "benchmarks/check_phase19_energy_comparison.py",
    "benchmarks/check_phase19_preregistration.py",
)
MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
}


def measurement_ops(bank, arm):
    """Decode frozen rotor/readout ledgers, checking their topology hashes."""
    n = bank["n_qubits"]
    output = []
    for setting in arm["settings"]:
        ops = []
        for rotor in setting["rotations"]:
            ops.extend(lower_rotor(PauliWord(n, rotor["word_code"]), rotor["angle"]))
        if arm["name"] == "clique":
            basis = {q: PauliWord(n, setting["word_codes"][0]).letter(q)
                     for q in setting["readout"]["qubits"]}
        else:
            basis = {}
            for code in setting["word_codes"]:
                word = PauliWord(n, code)
                for q in word.support():
                    letter = word.letter(q)
                    if q in basis and basis[q] != letter:
                        raise ValueError("QWC readout contains conflicting letters")
                    basis[q] = letter
        for q, letter in sorted(basis.items()):
            if letter == "Y":
                ops.append(("sdg", q))
            if letter in {"X", "Y"}:
                ops.append(("h", q))
        if structure._gate_sequence_digest(ops) != setting["gate_sequence_sha256"]:
            raise ValueError("frozen measurement gate topology drifted")
        output.append(tuple(ops))
    return tuple(output)


def tensor_pauli(label):
    out = np.ones((1, 1), dtype=complex)
    for letter in label:
        out = np.kron(out, MATS[letter])
    return out


def dense_unitary(n, ops):
    """Explicit tensor-factor gate action on every computational basis column."""
    size = 1 << n
    unitary = np.eye(size, dtype=complex)
    for op in ops:
        if op[0] == "cx":
            _, control, target = op
            permutation = [col ^ (1 << (n - 1 - target))
                           if col & (1 << (n - 1 - control)) else col
                           for col in range(size)]
            unitary = unitary[permutation, :]
        else:
            if op[0] == "rz":
                _, angle, q = op
                gate = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
            else:
                name, q = op
                gate = MATS[name]
            tensor = unitary.reshape(1 << q, 2, 1 << (n - q - 1), size)
            unitary = np.einsum("ij,ajbk->aibk", gate, tensor).reshape(size, size)
    return unitary


def reference_index(bank):
    return sum(1 << (bank["n_qubits"] - 1 - q)
               for q in bank["reference_occupied_spin_orbitals"])


def rational_reference(bank):
    """Exact binary-rational HF functional; off-diagonal words have zero mean."""
    n = bank["n_qubits"]
    occupied = set(bank["reference_occupied_spin_orbitals"])
    energy = Fraction(0)
    for code, coefficient in bank["coefficients_hex"]:
        label = PauliWord(n, code).label
        if any(letter in "XY" for letter in label):
            continue
        sign = (-1) ** sum(label[q] == "Z" for q in occupied)
        energy += sign * Fraction.from_float(float.fromhex(coefficient))
    return {"numerator": energy.numerator, "denominator": energy.denominator,
            "energy_hartree": float(energy)}


def diagonal_contribution(bank, arm, setting):
    """Enumerate the scalar readout, keeping all QWC words in the same shot."""
    n = bank["n_qubits"]
    coefficients = {code: float.fromhex(value) for code, value in bank["coefficients_hex"]}
    values = []
    for outcome in range(1 << n):
        def parity(support):
            return (-1) ** sum(bool(outcome & (1 << (n - 1 - q))) for q in support)
        if arm["name"] == "clique":
            value = setting["readout"]["weight"] * parity(setting["readout"]["qubits"])
        else:
            value = math.fsum(coefficients[code] * parity(support)
                              for code, support in zip(setting["word_codes"],
                                  setting["readout"]["word_qubits"]))
        values.append(value)
    return np.asarray(values)


def corrected_probabilities(raw):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("probabilities must be a finite nonempty vector")
    tolerance = 1e-12  # frozen protocol.probability_policy
    total = math.fsum(float(value) for value in raw)
    if raw.min() < -tolerance or raw.max() > 1 + tolerance or abs(total - 1) > tolerance:
        raise ValueError("raw circuit law fails the frozen probability policy")
    clipped_mass = math.fsum(-float(value) for value in raw if value < 0)
    clipped = np.maximum(raw, 0)
    clipped_total = math.fsum(float(value) for value in clipped)
    corrected = clipped / clipped_total  # no nonzero-probability cutoff
    return corrected, {"raw_normalization_error": abs(total - 1),
                       "clipped_mass": clipped_mass,
                       "normalization_change": abs(clipped_total - 1)}


def finite_or_none(value):
    # Nonfinite diagnostics remain visible as nulls rather than invalid JSON.
    return value if math.isfinite(value) else None


def dense_oracle(bank, *, laws=True):
    """Reconstruct the functional gate by gate; laws=False never touches the reference state."""
    n = bank["n_qubits"]
    size = 1 << n
    coefficients = {code: float.fromhex(value) for code, value in bank["coefficients_hex"]}
    hamiltonian = sum((value * tensor_pauli(PauliWord(n, code).label)
                       for code, value in coefficients.items()),
                      start=np.zeros((size, size), dtype=complex))
    offset = coefficients.get(0, 0.0)
    diagnostics = []
    arm_laws = []
    for arm in bank["arms"]:
        reconstructed = offset * np.eye(size, dtype=complex)
        unitarity = []
        raw_laws = []
        for setting, ops in zip(arm["settings"], measurement_ops(bank, arm)):
            unitary = dense_unitary(n, ops)
            unitarity.append(np.max(np.abs(unitary.conj().T @ unitary - np.eye(size))))
            diagonal = diagonal_contribution(bank, arm, setting)
            reconstructed += unitary.conj().T @ (diagonal[:, None] * unitary)
            if laws:
                raw_laws.append(np.abs(unitary[:, reference_index(bank)]) ** 2)
        diagnostics.append({"name": arm["name"],
            "max_operator_entry_error": finite_or_none(
                float(np.max(np.abs(reconstructed - hamiltonian)))),
            # np.max propagates NaN; the builtin max would silently drop it.
            "max_unitarity_entry_error": finite_or_none(float(np.max(unitarity, initial=0.0)))})
        arm_laws.append(raw_laws)
    if not laws:
        return {"arms": diagnostics}, None
    return {"reference": rational_reference(bank), "arms": diagnostics}, arm_laws


def within(value, tolerance):
    """Missing or NaN values fail; a bare `value > tolerance` test would pass NaN."""
    return value is not None and value <= tolerance


def oracle_failures(oracle):
    problems = []
    for row in oracle["arms"]:
        if not within(row["max_operator_entry_error"], 1e-10):
            problems.append(f"{row['name']}: operator reconstruction exceeds 1e-10")
        if not within(row["max_unitarity_entry_error"], 2e-12):
            problems.append(f"{row['name']}: unitarity exceeds 2e-12")
    return problems


def law_failures(raw, dense_raw, name, setting_index):
    """A sampler's circuit law must also match the independent gate oracle."""
    if raw.shape != dense_raw.shape:
        raise ValueError("raw law shape drift")
    if not np.isfinite(raw).all() or not within(float(np.max(np.abs(raw - dense_raw))), 1e-12):
        return [f"{name}/{setting_index}: circuit law disagrees with dense oracle"]
    return []


def json_probabilities(raw):
    return [finite_or_none(float(value)) for value in raw]


def blocked_arm(name, laws, error):
    return {"name": name, "laws": laws, "exact_law_reference_error": error,
            "sampling_status": "blocked", "endpoints": [], "certified_endpoint": None,
            "right_censored": None, "covariance_audit": None, "device_reports": []}


def independent_schedule(scores, budget):
    if type(budget) is not int or not scores or budget < 2 * len(scores):
        raise ValueError("a fixed schedule needs an integer budget of two shots per setting")
    if any(type(score) not in (int, float) or not math.isfinite(score) or score <= 0
           for score in scores):
        raise ValueError("schedule scores must be positive and finite")
    weights = list(map(Fraction.from_float, scores))
    quotas = [(budget - 2 * len(weights)) * weight / sum(weights) for weight in weights]
    whole = [int(value) for value in quotas]
    order = sorted(range(len(weights)), key=lambda i: (whole[i] - quotas[i], i))
    for index in order[:budget - 2 * len(weights) - sum(whole)]:
        whole[index] += 1
    return [value + 2 for value in whole]


def rational_counts(histogram, values, n):
    """Validate a sparse histogram and pair each count with its exact value."""
    if not isinstance(histogram, list):
        raise ValueError("histogram must be a sparse array")
    previous = -1
    counts = []
    for row in histogram:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("invalid sparse histogram row")
        outcome, count = row
        if (type(outcome) is not int or type(count) is not int or
                not previous < outcome < len(values) or count <= 0):
            raise ValueError("invalid, unsorted or duplicated histogram outcome/count")
        previous = outcome
        counts.append((count, Fraction.from_float(float(values[outcome]))))
    if sum(count for count, _ in counts) != n or n < 2:
        raise ValueError("histogram disagrees with the fixed schedule")
    return counts


def rational_mean(counts, n):
    return sum((count * value for count, value in counts), start=Fraction(0)) / n


def histogram_mean(histogram, values, n):
    """Audit replicas use only the mean; the exact variance is the costly part."""
    return float(rational_mean(rational_counts(histogram, values, n), n))


def histogram_statistics(histogram, values, n):
    """Exact rational centering avoids subtracting two large raw moments."""
    counts = rational_counts(histogram, values, n)
    mean = rational_mean(counts, n)
    variance = sum((count * (value - mean) ** 2 for count, value in counts),
                   start=Fraction(0)) / (n - 1)
    return float(mean), float(variance)


def radius(variance, n, delta, bound):
    log = math.log(3 / delta)
    return math.sqrt(2 * variance * log / n) + 6 * bound * log / n


def law_moments(probabilities, values):
    positive = np.flatnonzero(probabilities)
    anchor = float(values[positive[0]])
    difference = math.fsum(float(p) * (float(y) - anchor)
                           for p, y in zip(probabilities, values))
    mean = anchor + difference
    variance = math.fsum(float(p) * ((float(y) - anchor) - difference) ** 2
                         for p, y in zip(probabilities, values))
    return mean, variance


def seeded_histogram(probabilities, root, indices, shots):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(root, spawn_key=indices)))
    counts = rng.multinomial(shots, probabilities)
    return [[i, int(count)] for i, count in enumerate(counts) if count]


def covariance_result(estimates, predicted, protocol, shots):
    anchor = estimates[0]
    mean_difference = math.fsum(value - anchor for value in estimates) / len(estimates)
    empirical = math.fsum(((value - anchor) - mean_difference) ** 2
                         for value in estimates) / (len(estimates) - 1)
    ratio = None if predicted == 0 else empirical / predicted
    low, high = protocol["covariance_audit"]["acceptable_empirical_to_predicted_interval"]
    passed = empirical == 0 if predicted == 0 else low <= ratio <= high
    return {"shots": shots, "estimates": estimates, "predicted_variance": predicted,
            "empirical_variance": empirical, "ratio": ratio, "passed": passed}


def decision(qwc, clique, config):
    maximum = config["protocol"]["effective_shot_endpoints"][-1]
    factor = config["acceptance"]["material_shot_reduction_factor"]
    if qwc is None:
        return ("MATERIAL_REDUCTION_CENSORED_BASELINE" if clique is not None
                and clique <= maximum / factor else "INDETERMINATE")
    if clique is None:
        return "NO_MATERIAL_REDUCTION"
    return "MATERIAL_REDUCTION" if clique <= qwc / factor else "NO_MATERIAL_REDUCTION"


def independent_device_reports(settings, endpoint, cards, n, epsilon):
    """Reprice cards directly, without calling the production cost routines."""
    reports = []
    for card in cards:
        fidelities = []
        durations = []
        for item in settings:
            def routed(count, multiplier):
                rational = Fraction(str(multiplier))
                return -((-count * rational.numerator) // rational.denominator)
            fidelities.append((1 - card.eps_1q) ** item["n_1q"] *
                (1 - card.eps_2q) ** routed(item["n_2q"], card.routing_2q_multiplier) *
                (1 - card.eps_readout) ** n)
            durations.append(card.t_prep_us + card.t_readout_us + card.t_reset_us +
                card.t_1q_us * item["d_1q"] +
                card.t_2q_us * routed(item["d_2q"], card.routing_depth_multiplier))
        admissible = all(value >= card.fidelity_floor for value in fidelities)
        raw = None if endpoint is None else [math.ceil(count / fidelity ** 2)
            for count, fidelity in zip(endpoint["shots"], fidelities)]
        reports.append({"card": card.name, "card_sha256": card.sha256,
            "admissible": admissible, "setting_fidelities": fidelities,
            "status": "inadmissible" if not admissible else
                      "right_censored" if endpoint is None else "priced",
            "raw_shots": raw, "total_raw_shots": None if raw is None else sum(raw),
            "C_time_epsilon_us": None if endpoint is None or not admissible else
                                 math.fsum(count * time for count, time in zip(raw, durations)),
            "epsilon_hartree": epsilon})
    return reports


def _numbers_compare(expected, actual, path="record"):
    """Closed/type-strict structure; tolerate only recomputed float leaves."""
    if isinstance(expected, float) and type(actual) is float:
        return [] if math.isfinite(actual) and math.isclose(expected, actual,
            rel_tol=1e-11, abs_tol=2e-12) else [f"{path}: number differs"]
    if type(expected) is not type(actual):
        return [f"{path}: type differs"]
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return [f"{path}: key set differs"]
        return [problem for key in expected for problem in
                _numbers_compare(expected[key], actual[key], f"{path}.{key}")]
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: length differs"]
        return [problem for i, (left, right) in enumerate(zip(expected, actual))
                for problem in _numbers_compare(left, right, f"{path}[{i}]")]
    return [] if expected == actual else [f"{path}: value differs"]


def check_bank(record, bank, config, cards, system_index):
    """Independent full seeded replay on one bank; accepts undeclared toy inputs."""
    protocol = config["protocol"]
    problems = _compare(bank, record["structural_bank"], bank["id"])

    def finish(expected):
        # The structural bank was compared strictly above; every other leaf,
        # histograms included, is walked exactly once here.
        rest = {key: value for key, value in record.items() if key != "structural_bank"}
        return problems + _numbers_compare(expected, rest), blocking

    oracle, raw_laws = dense_oracle(bank)
    blocking = oracle_failures(oracle)
    if blocking:
        return finish({"oracle": oracle, "arms": [], "blocking_failures": blocking})
    expected_arms = []
    coefficients = dict(bank["coefficients_hex"])
    offset = float.fromhex(coefficients.get(0, "0x0.0p+0"))
    reference = oracle["reference"]["energy_hartree"]
    prepared = []
    for ai, arm in enumerate(bank["arms"]):
        stored = record["arms"][ai]
        values = [diagonal_contribution(bank, arm, item) for item in arm["settings"]]
        laws = []
        means, variances = [], []
        for si, (raw, observed) in enumerate(zip(raw_laws[ai], stored["laws"])):
            observed_raw = np.asarray(observed["raw_probabilities"], dtype=float)
            blocking += law_failures(observed_raw, raw, arm["name"], si)
            try:
                probabilities, correction = corrected_probabilities(observed_raw)
                mean, variance = law_moments(probabilities, values[si])
                law = {"raw_probabilities": json_probabilities(observed_raw),
                       "probabilities": probabilities.tolist(), "correction": correction,
                       "mean": mean, "variance": variance, "failure": None}
            except ValueError as exc:
                mean, variance = None, None
                law = {"raw_probabilities": json_probabilities(observed_raw),
                       "probabilities": None, "correction": None,
                       "mean": None, "variance": None, "failure": str(exc)}
                blocking.append(f"{arm['name']}/{si}: {exc}")
            laws.append(law)
            means.append(mean)
            variances.append(variance)
        if len(laws) != len(arm["settings"]):
            raise ValueError("missing setting laws")
        law_error = None if any(value is None for value in means) else abs(math.fsum([offset, *means]) - reference)
        if law_error is not None and not within(law_error,
                                                protocol["numerical_bias_allowance_hartree"]):
            blocking.append(f"{arm['name']}: reference law mean exceeds numerical allowance")
        prepared.append((laws, values, variances, law_error))
    if blocking:
        return finish({"oracle": oracle,
                       "arms": [blocked_arm(arm["name"], laws, error)
                                for arm, (laws, _, _, error) in zip(bank["arms"], prepared)],
                       "blocking_failures": blocking})
    for ai, (arm, (laws, values, variances, law_error)) in enumerate(zip(bank["arms"], prepared)):
        stored = record["arms"][ai]
        scores = [item["max_abs_value"] for item in arm["settings"]]
        endpoints = []
        delta = protocol["confidence"]["delta"] / (protocol["confidence"]["family"] *
            protocol["confidence"]["rounds"] * len(scores))
        for ei, budget in enumerate(protocol["effective_shot_endpoints"]):
            shots = independent_schedule(scores, budget)
            histograms = [seeded_histogram(law["probabilities"], protocol["seeds"]["headline"],
                (system_index, ai, ei, si), count) for si, (law, count) in enumerate(zip(laws, shots))]
            observed_histograms = stored["endpoints"][ei]["histograms"]
            if len(observed_histograms) != len(shots):
                raise ValueError("histogram setting count differs")
            stats = [histogram_statistics(hist, value, count)
                     for hist, value, count in zip(observed_histograms, values, shots)]
            radii = [radius(variance, count, delta, bound)
                     for (_, variance), count, bound in zip(stats, shots, scores)]
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
        audit_shots = independent_schedule(scores,
            protocol["covariance_audit"]["effective_shots_per_replica"])
        predicted = math.fsum(var / count for var, count in zip(variances, audit_shots))
        estimates = []
        audit_digest = hashlib.sha256()
        for ri in range(protocol["covariance_audit"]["replicas"]):
            histograms = [seeded_histogram(law["probabilities"],
                protocol["seeds"]["covariance_audit"], (system_index, ai, ri, si), count)
                for si, (law, count) in enumerate(zip(laws, audit_shots))]
            audit_digest.update(json.dumps(histograms, separators=(",", ":")).encode())
            estimates.append(math.fsum([offset, *(histogram_mean(hist, value, count)
                for hist, value, count in zip(histograms, values, audit_shots))]))
        audit = covariance_result(estimates, predicted, protocol, audit_shots)
        audit["histograms_sha256"] = audit_digest.hexdigest()
        if not audit["passed"]:
            blocking.append(f"{arm['name']}: covariance audit failed")
        expected_arms.append({"name": arm["name"], "laws": laws,
            "exact_law_reference_error": law_error, "sampling_status": "complete", "endpoints": endpoints,
            "certified_endpoint": None if certified is None else certified["effective_shots"],
            "right_censored": certified is None, "covariance_audit": audit,
            "device_reports": independent_device_reports(
                [item["resources"] for item in arm["settings"]], certified, cards,
                bank["n_qubits"], protocol["accuracy_hartree"])})
    return finish({"oracle": oracle, "arms": expected_arms, "blocking_failures": blocking})


def environment_problems(provenance, reference):
    problems = []
    if str(provenance.get("python", "")).split(".")[:2] != reference["python"].split(".")[:2]:
        problems.append("Python minor differs from the frozen environment")
    for name, version in reference["dependencies"].items():
        actual = provenance.get("dependencies", {}).get(name)
        if version is not None and actual is not None and version != actual:
            problems.append(f"{name} differs from the frozen environment")
    if provenance.get("dependencies", {}).get("numpy") != reference["dependencies"]["numpy"]:
        problems.append("the frozen NumPy sampling version is required")
    return problems


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True,
                          check=True).stdout.strip()


def provenance_problems(record, config, root=ROOT):
    p = record["provenance"]
    problems = []
    if p.get("git_dirty") is not False or p.get("preregistration_merge_commit") != MERGE:
        problems.append("record requires the clean execution and frozen merge identities")
    if p.get("git_sha") != p.get("git_commit"):
        problems.append("execution SHA aliases disagree")
    reference = read_json(root / config["lineage"]["environment_reference"],
                          result_free=False)["provenance"]
    problems += environment_problems(p, reference)
    current = [structure.file_binding(path, root) for path in SOURCES]
    problems += _compare(current, p.get("source_files"), "source_files")
    for binding in current:
        blob = git(root, "rev-parse", f"{p['git_commit']}:{binding['path']}")
        if blob != binding["git_blob_sha1"]:
            problems.append(f"execution source drift: {binding['path']}")
    records = git(root, "log", "--reverse", "--format=%H", "HEAD", "--",
                  structure.RECORD_PATH).splitlines()
    if records:
        changed = git(root, "diff-tree", "--no-commit-id", "--name-only", "-r",
                      records[0]).splitlines()
        if changed != [structure.RECORD_PATH]:
            problems.append("the first record must be committed alone after execution")
    # The preregistration checker pins input/manifests, and its ancestry gate
    # requires execution strictly before the first committed record.
    problems += temporal_problems(config, root)
    return problems


def check_record(record, config, banks, cards, *, check_provenance=True):
    problems = []
    failures = []
    if record["schema"] != SCHEMA or record["declaration_sha256"] != EXPECTED_CONFIG_SHA256:
        problems.append("record schema/declaration binding differs")
    if set(record) != {"schema", "declaration_sha256", "banks", "shot_decisions", "verdict",
                       "accounting", "provenance"}:
        problems.append("record key set differs")
    if len(record["banks"]) != len(banks):
        return problems + ["bank count differs"]
    for index, (stored, bank) in enumerate(zip(record["banks"], banks)):
        bank_problems, blocking = check_bank(stored, bank, config, cards, index)
        problems += [f"{bank['id']}: {problem}" for problem in bank_problems]
        failures += blocking
    expected_verdict = "INVALID" if failures else "VALID"
    if record["verdict"] != expected_verdict:
        problems.append("campaign verdict differs")
    expected_decisions = [None if failures else decision(
        cell["arms"][0]["certified_endpoint"], cell["arms"][1]["certified_endpoint"], config)
        for cell in record["banks"]]
    problems += _numbers_compare(expected_decisions, record["shot_decisions"], "shot_decisions")
    headline = sum(sum(sum(endpoint["shots"]) for endpoint in arm["endpoints"])
                   for cell in record["banks"] for arm in cell["arms"])
    audit = sum(sum(arm["covariance_audit"]["shots"]) * len(arm["covariance_audit"]["estimates"])
                for cell in record["banks"] for arm in cell["arms"]
                if arm["covariance_audit"] is not None)
    accounting = record["accounting"]
    if set(accounting) != {"headline_effective_shots", "audit_effective_shots",
                           "compilation_seconds", "sampling_seconds"}:
        problems.append("accounting key set differs")
    for key, value in (("headline_effective_shots", headline), ("audit_effective_shots", audit)):
        problems += _numbers_compare(value, accounting.get(key), key)
    for key in ("compilation_seconds", "sampling_seconds"):
        value = accounting.get(key)
        if type(value) is not float or not math.isfinite(value) or value < 0:
            problems.append(f"invalid classical timing: {key}")
    if check_provenance:
        problems += provenance_problems(record, config)
    return problems


def main():
    try:
        config = load_config()
        problems = static_problems(config)
        if problems:
            raise ValueError("; ".join(problems))
        if not RECORD.exists():
            print("OK: Phase 19 declaration intact; campaign record absent; no outcomes evaluated")
            return 0
        record = read_json(RECORD, result_free=False)
        # Refuse a replay under an incompatible sampling stream before drawing;
        # provenance_problems separately binds the record to the frozen reference.
        problems = sampling_stream_mismatch(record)
        if problems:
            raise ValueError("; ".join(problems))
        banks = [read_json(ROOT / item["path"]) for item in config["bank_manifests"]]
        cards = [DeviceCard.from_dict(read_json(ROOT / item["path"]))
                 for item in config["protocol"]["device_cards"]]
        problems = check_record(record, config, banks, cards)
        if problems:
            raise ValueError("; ".join(problems[:30]))
        print(f"OK: independent Phase 19 reconstruction, replay and provenance; {record['verdict']}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as exc:
        print(f"Phase 19 record failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

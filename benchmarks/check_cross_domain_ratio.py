"""Independently check cross-domain ratio evidence without drawing outcomes.

Scientific assessment uses tensor-Pauli TFIM matrices and occupation-sign
molecular Hamiltonians, spin, and pair transfers. Producer setup, summaries,
verdicts, and current-process experiment admission are deliberately not used.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.phase17_ceo_oracle import _fermionic_action, hamiltonian, ladder, multiply
from benchmarks.check_orthogonal_active_space import independent_spin_squared
from clifford_qc.multivector import MV
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.subspace.sampled import SampledProjectedProvider

DECLARATION = "docs/restart/cross_domain_ratio_declaration.json"
RESULT = "docs/restart/cross_domain_ratio_results.json.gz"
BENCHMARK_SOURCES = (
    "benchmarks/run_cross_domain_ratio.py", "benchmarks/check_cross_domain_ratio.py",
    "benchmarks/run_sampled_ratio.py", "benchmarks/phase17_ceo_oracle.py",
    "benchmarks/check_orthogonal_active_space.py",
)
ARMS = ("ratio", "bank")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_record(path):
    path = Path(path)
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def close(left, right, label, *, atol=1e-9, rtol=1e-10):
    a, b = np.asarray(left), np.asarray(right)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=rtol), label)


def complex_array(value, label):
    values = np.asarray(value, dtype=float)
    require(values.ndim >= 2 and values.shape[-1] == 2 and np.isfinite(values).all(), label)
    return values[..., 0] + 1j * values[..., 1]


def tensor_word(n, code):
    local = (np.eye(2), np.array([[0, 1], [1, 0]]),
             np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]))
    result = np.ones((1, 1), complex)
    for q in range(n):
        result = np.kron(result, local[(code >> (2*q)) & 3])
    return result


def _pair_matrix(words, n, source, target):
    lookup = {word: i for i, word in enumerate(words)}
    sequence = [(2*target, True), (2*target+1, True),
                (2*source+1, False), (2*source, False)]
    transfer = np.zeros((len(words), len(words)), complex)
    for column, word in enumerate(words):
        action = _fermionic_action(word, sequence, n)
        if action is not None:
            new_word, sign = action
            require(new_word in lookup, "pair transfer escaped fixed sector")
            transfer[lookup[new_word], column] += sign
    return transfer - transfer.conj().T


def _pair_coefficients(n, source, target):
    sequence = [(2*target, True), (2*target+1, True),
                (2*source+1, False), (2*source, False)]
    terms = {0: 1.}
    for q, creation in sequence:
        terms = multiply(terms, ladder(n, q, creation, fermionic=True), n)
    # Pauli words are Hermitian; conjugating their scalar coefficients is the
    # adjoint of the independently multiplied fermionic transfer operator.
    return {code: value-value.conjugate() for code, value in terms.items()
            if abs(value-value.conjugate()) > 1e-12}


@lru_cache(maxsize=12)
def independent_case(case_json):
    case = json.loads(case_json)
    n = case["n"]
    if case["kind"] == "tfim":
        words = list(range(2**n))
        coefficients = {3 << (2*q) | 3 << (2*(q+1)): -case["J"] for q in range(n-1)}
        coefficients.update({1 << (2*q): -case["h"] for q in range(n)})
        h = sum((value*tensor_word(n, code) for code, value in coefficients.items()),
                np.zeros((2**n, 2**n), complex))
        state = np.ones(1, complex)
        for q in range(n):
            theta = case["theta"] + .1*(q % 2)
            phi = (-1)**q * case["phi"]
            state = np.kron(state, [math.cos(theta/2), np.exp(1j*phi)*math.sin(theta/2)])
        matrices = [np.eye(len(words)), h/n, tensor_word(n, 3)]
        generator_terms = [{0: 1.}, {code: value/n for code, value in coefficients.items()}, {3: 1.}]
        labels = ("I", "H/n", "Z0")
        ground = float(np.linalg.eigvalsh(h)[0])
        spin = None
    else:
        words = [word for word in range(2**n)
                 if word.bit_count() == case["n_electrons"]
                 and sum((1 if q % 2 == 0 else -1)*((word >> (n-1-q)) & 1)
                         for q in range(n))/2 == case["sz"]]
        h, coefficients = hamiltonian(ROOT/case["fcidump"], words)
        spin = independent_spin_squared(words, n)
        state = np.zeros(len(words), complex)
        state[words.index(int(case["reference_bits"][0], 2))] = math.cos(case["theta"])
        state[words.index(int(case["reference_bits"][1], 2))] = np.exp(1j*case["phi"])*math.sin(case["theta"])
        matrices, generator_terms = [np.eye(len(words), dtype=complex)], [{0: 1.}]
        for target in case["pair_targets"]:
            generator = _pair_matrix(words, n, case["pair_source"], target)
            matrices.append((np.eye(len(words))+generator)/math.sqrt(2))
            terms = _pair_coefficients(n, case["pair_source"], target)
            terms[0] = terms.get(0, 0) + 1
            generator_terms.append({code: value/math.sqrt(2) for code, value in terms.items()})
        labels = ("I", "(I+G12)/sqrt(2)", "(I+G13)/sqrt(2)")
        spin_values, spin_vectors = np.linalg.eigh(spin)
        mask = np.abs(spin_values-case["target_spin"]*(case["target_spin"]+1)) < 1e-9
        require(mask.any(), "declared spin sector is empty")
        frame = spin_vectors[:, mask]
        ground = float(np.linalg.eigvalsh(frame.conj().T@h@frame)[0])
    basis = np.column_stack([matrix@state for matrix in matrices])
    if spin is not None:
        target = case["target_spin"]*(case["target_spin"]+1)
        close(spin@basis, target*basis, "reference/correction span escaped target spin")
    s, projected = basis.conj().T@basis, basis.conj().T@h@basis
    close(s, s.conj().T, "independent overlap is not Hermitian")
    close(projected, projected.conj().T, "independent Hamiltonian is not Hermitian")
    return {"words": words, "hamiltonian": h, "state": state, "basis": basis,
            "overlap": s, "projected": projected, "ground": ground,
            "h_terms": coefficients, "generator_terms": generator_terms, "labels": labels}


def case_data(case):
    return independent_case(json.dumps(case, sort_keys=True))


def _refuse_samples(*args):
    raise AssertionError("record checking must not draw fresh outcomes")


def _pauli_restriction(operator, words):
    """Independent letter action, without native Pauli products or restrictions."""
    lookup = {word: i for i, word in enumerate(words)}
    matrix = np.zeros((len(words), len(words)), complex)
    for column, word in enumerate(words):
        escaped = {}
        for code, coefficient in operator.terms.items():
            target, value = word, coefficient
            for q in range(operator.n):
                letter, mask = (code >> (2*q)) & 3, 1 << (operator.n-1-q)
                occupied = bool(word & mask)
                if letter in (1, 2):
                    target ^= mask
                if letter == 2:
                    value *= -1j if occupied else 1j
                elif letter == 3 and occupied:
                    value *= -1
            if target in lookup:
                matrix[lookup[target], column] += value
            else:
                escaped[target] = escaped.get(target, 0)+value
        require(all(abs(value) < 1e-10 for value in escaped.values()), "native Hamiltonian leaks particle/spin sector")
    return matrix


@lru_cache(maxsize=12)
def _structural_provider(case_json):
    case = json.loads(case_json)
    data = independent_case(case_json)
    keys = ("kind", "n", "theta", "phi") + (("reference_bits",) if case["kind"] == "molecule" else ())
    preparation = json.dumps({key: case[key] for key in keys}, sort_keys=True)
    reference = PreparedSampledReference(case["n"], preparation, _refuse_samples)
    h = (fcidump_model(ROOT/case["fcidump"]).hamiltonian.to_mv() if case["kind"] == "molecule"
         else MV(case["n"], data["h_terms"]))
    close(_pauli_restriction(h, data["words"]), data["hamiltonian"], "native/independent Hamiltonian")
    generators = [(label, MV(case["n"], terms)) for label, terms
                  in zip(data["labels"], data["generator_terms"])]
    return SampledProjectedProvider(reference, h, generators)


def structural_provider(case):
    return _structural_provider(json.dumps(case, sort_keys=True))


def array_bound(n):
    return 128*12*4**n + 16*4**n*2**n + 8*16*4**n + 2**20


def _depths(gates):
    # Independent scheduling of one-/two-qubit type-homogeneous layers, on one
    # dependency clock. Commuting CZ operations still consume disjoint layers.
    occupied, kinds = {}, {}
    for name, qubits in gates:
        require(name in ("H", "S", "SDG", "CZ"), "unsupported readout gate")
        require(len(qubits) == (2 if name == "CZ" else 1), "readout gate arity")
        layer = max((occupied.get(q, 0) for q in qubits), default=0)+1
        kind = len(qubits)
        while layer in kinds and kinds[layer] != kind:
            layer += 1
        kinds[layer] = kind
        for q in qubits:
            occupied[q] = layer
    return sum(value == 1 for value in kinds.values()), sum(value == 2 for value in kinds.values())


def settings_payload(settings):
    payload = []
    for setting in settings:
        d1, d2 = _depths(setting.clifford.gates)
        payload.append({"assigned_word_codes": list(setting.assigned_word_codes),
                        "readouts": {str(code): [int(sign), list(positions)]
                                     for code, (sign, positions) in setting.readouts.items()},
                        "gates": [[name, list(qubits)] for name, qubits in setting.clifford.gates],
                        "d_1q": d1, "d_2q": d2})
    return payload


def preparation_stats(case):
    if case["kind"] == "tfim":
        return {"one_qubit_gates": 2*case["n"], "two_qubit_gates": 0,
                "one_qubit_depth": 2, "two_qubit_depth": 0}
    return {"one_qubit_gates": 6, "two_qubit_gates": 3,
            "one_qubit_depth": 3, "two_qubit_depth": 3}


def device_seconds(settings, counts, case, config):
    card, prep = config["device_scenario_us"], preparation_stats(case)
    cost = 0.
    for setting, count in zip(settings, counts):
        d1, d2 = _depths(setting.clifford.gates)
        cost += count*(case["n"]*card["preparation_per_qubit"] + card["readout_reset"]
                       + (d1+prep["one_qubit_depth"])*card["one_qubit_layer"]
                       + (d2+prep["two_qubit_depth"])*card["two_qubit_layer"])
    return cost*1e-6


def binomial_lower(successes, attempts, alpha):
    if successes == 0:
        return 0.
    lower, upper = 0., 1.
    for _ in range(64):
        p = (lower+upper)/2
        tail = math.fsum(math.comb(attempts, k)*p**k*(1-p)**(attempts-k)
                         for k in range(successes, attempts+1))
        if tail < alpha:
            lower = p
        else:
            upper = p
    return (lower+upper)/2


def independent_summaries(rows, config):
    summaries = []
    family = len(config["cases"])*len(config["total_shot_budgets"])*len(ARMS)
    for case in config["cases"]:
        for budget in config["total_shot_budgets"]:
            for arm in ARMS:
                group = [row for row in rows if (row["case"], row["budget"], row["arm"])
                         == (case["name"], budget, arm)]
                successes = sum(row["status"] == "resolved" and row["estimation_error"]
                                <= case["target_estimation_abs_error"] for row in group)
                summary = {"case": case["name"], "role": case["role"], "kind": case["kind"], "unit": case["unit"],
                           "target_estimation_abs_error": case["target_estimation_abs_error"],
                           "budget": budget, "arm": arm, "attempts": len(group), "successes": successes,
                           "success_probability": successes/len(group),
                           "success_probability_lower": binomial_lower(successes, len(group),
                                config["success_probability_alpha"]/family)}
                for status, field in (("pilot_failed", "pilot_failures"),
                                      ("denominator_unresolved", "denominator_failures"),
                                      ("resource_failed", "resource_failures"),
                                      ("worker_failed", "worker_failures")):
                    summary[field] = sum(row["status"] == status for row in group)
                summary["other_failures"] = sum(row["status"] in ("evaluation_failed", "assessment_failed") for row in group)
                summary["incomplete_costs"] = sum(not row.get("costs_complete", True) for row in group)
                summary["resolved"] = sum(row["status"] == "resolved" for row in group)
                summary["resolved_interval_misses"] = sum(row["status"] == "resolved" and row.get("interval_contains_truth") is False for row in group)
                for source, field in (("end_to_end_seconds", "median_end_to_end_seconds"),
                                      ("device_seconds", "median_device_seconds"),
                                      ("classical_seconds", "median_classical_seconds"),
                                      ("total_shots", "median_total_shots")):
                    values = [row[source] for row in group if row.get(source) is not None]
                    summary[field] = statistics.median(values) if values else None
                for source, field in (("estimation_error", "median_resolved_estimation_error"),
                                      ("retained_space_error", "median_resolved_retained_space_error"),
                                      ("ground_error", "median_resolved_ground_error")):
                    values = [row[source] for row in group if row["status"] == "resolved" and row.get(source) is not None]
                    summary[field] = statistics.median(values) if values else None
                summaries.append(summary)
    return summaries


def independent_verdict(summaries, config):
    comparisons = []
    for case in config["cases"]:
        eligible = {}
        for arm in ARMS:
            choices = [row for row in summaries if row["case"] == case["name"] and row["arm"] == arm
                       and row["success_probability_lower"] >= config["target_success_probability"]
                       and row["resource_failures"] == row["worker_failures"] == row["incomplete_costs"] == 0]
            eligible[arm] = min(choices, key=lambda row: row["median_end_to_end_seconds"]) if choices else None
        ratio, bank = eligible["ratio"], eligible["bank"]
        saving = 1-ratio["median_end_to_end_seconds"]/bank["median_end_to_end_seconds"] if ratio and bank else None
        comparisons.append({"case": case["name"], "unit": case["unit"],
                            "target_estimation_abs_error": case["target_estimation_abs_error"],
                            "ratio_budget": ratio["budget"] if ratio else None,
                            "bank_budget": bank["budget"] if bank else None,
                            "cost_reduction": saving, "passes": saving is not None and saving >= .2})
    return {"status": "promote_on_declared_scope" if all(row["passes"] for row in comparisons)
            else "closed_on_declared_scope", "comparisons": comparisons,
            "quantum_advantage_claim": False, "hardware_claim": False}


def compare_payload(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label+" fields")
        for key, value in expected.items():
            compare_payload(actual[key], value, label+"."+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label+" length")
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare_payload(left, right, f"{label}[{i}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), label)
    else:
        require(actual == expected, label)


def validate_declaration(config):
    require(config["schema"] == "clifford_qc.cross_domain_ratio_declaration.v1", "declaration schema")
    require(config.get("result_format") == "json.gz;gzip_mtime=0"
            and config.get("case_timeout_seconds") == 1800, "record format/worker timeout")
    require(config["total_shot_budgets"] == [16384, 65536, 262144] and config["replicas"] == 128,
            "frozen budget/replica grid")
    require(config["master_seed"] == 20261028 and config["pilot_fraction"] == .25, "frozen streams/pilot")
    require(config["tau_s"] == config["ratio_delta"] == config["denominator_floor"] == .05
            and config["max_condition"] == 1e4 and config["target_success_probability"] == .9
            and config["success_probability_alpha"] == .05
            and config["owned_array_cap_bytes"] == 512*2**20
            and config["process_rss_cap_bytes"] == 1024*2**20, "frozen inference/resource rule")
    require(config["compiler"] == "graph_state" and config["storage"] == "packed"
            and config["pooling"] == "assigned" and config["block_size"] == "full_register", "protocol")
    require([case["name"] for case in config["cases"]]
            == ["tfim4_development", "h4_heldout_molecule", "lih_heldout_molecule"], "declared case grid")
    require([case["role"] for case in config["cases"]]
            == ["development_primary", "held_out_molecule", "held_out_molecule"], "held-out case roles")
    require(config["generators"] == {"tfim": ["I", "H/n", "Z0"],
            "molecule": ["I", "(I+G12)/sqrt(2)", "(I+G13)/sqrt(2)"]}, "fixed generator families")
    require(len(config["source_paths"]) == len(set(config["source_paths"]))
            and all(isinstance(path, str) and not Path(path).is_absolute()
                    and ".." not in Path(path).parts for path in config["source_paths"]), "source path contract")
    for case in config["cases"]:
        if case["kind"] == "tfim":
            require(case["n"] == 4 and case["unit"] == "t" and case["target_estimation_abs_error"] == .05,
                    "spin units/target")
            require((case["J"], case["h"], case["theta"], case["phi"])
                    == (1., 1.2, 1.1, .2), "development control")
        else:
            require(case["n"] == 8 and case["unit"] == "Ha" and case["target_estimation_abs_error"] == .0016
                    and case["n_electrons"] == 4 and case["sz"] == case["target_spin"] == 0
                    and case["pair_source"] == 1 and case["pair_targets"] == [2, 3]
                    and case["reference_bits"] == ["11110000", "11001100"]
                    and case["theta"] == .3 and case["phi"] == .2, "molecular scope/preparation")
            require(digest_bytes((ROOT/case["fcidump"]).read_bytes()) == case["fcidump_sha256"],
                    "FCIDUMP input bytes")
    for path, expected in config["historical_bindings"].items():
        require(digest_bytes((ROOT/path).read_bytes()) == expected, "historical evidence changed: "+path)


def check_history(record, config):
    require(git("rev-parse", "--is-shallow-repository").strip() == b"false", "full history required")
    declared, executed = record["declaration_commit"], record["execution_commit"]
    for value in (declared, executed):
        require(isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value),
                "full commit identity required")
    declaration_bytes = (ROOT/DECLARATION).read_bytes()
    require(record["declaration_sha256"] == digest_bytes(declaration_bytes), "declaration digest")
    require(git("show", f"{declared}:{DECLARATION}") == declaration_bytes
            and git("show", f"{executed}:{DECLARATION}") == declaration_bytes, "committed declaration binding")
    subprocess.run(["git", "merge-base", "--is-ancestor", declared, executed], cwd=ROOT, check=True)
    source_paths = {path for path in git("ls-tree", "-r", "--name-only", executed).decode().splitlines()
                    if path.startswith("clifford_qc/") and path.endswith(".py")}
    declared_sources = set(config["source_paths"])
    require(source_paths.issubset(declared_sources) and set(BENCHMARK_SOURCES).issubset(declared_sources),
            "declaration scientific source coverage")
    require(set(record["source_sha256"]) == declared_sources, "complete execution source coverage")
    for path, expected in record["source_sha256"].items():
        require(digest_bytes(git("show", f"{executed}:{path}")) == expected,
                "historical execution source digest: "+path)
    inputs = {case["fcidump"]: case["fcidump_sha256"] for case in config["cases"] if case["kind"] == "molecule"}
    require(record["input_sha256"] == inputs, "molecular input coverage")
    for path, expected in inputs.items():
        require(digest_bytes(git("show", f"{executed}:{path}")) == expected, "execution input binding")
    first = git("log", "--diff-filter=A", "--format=%H", "--", RESULT).decode().splitlines()
    if first:
        observation = first[-1]
        require(observation != executed and observation != declared, "outcomes require a later commit")
        subprocess.run(["git", "merge-base", "--is-ancestor", executed, observation], cwd=ROOT, check=True)
        original = json.loads(gzip.decompress(git("show", f"{observation}:{RESULT}")))
        require(record == original, "published observation payload changed")
    for path, expected in config["historical_bindings"].items():
        require(digest_bytes(git("show", f"{executed}:{path}")) == expected, "execution historical binding")


def verify_assessment(row, case, data):
    coefficients = complex_array(row["coefficients"], "coefficient encoding")
    frame = complex_array(row["retained_frame"], "retained-frame encoding")
    require(coefficients.shape == (3,) and frame.ndim == 2 and frame.shape[0] == 3
            and 1 <= frame.shape[1] <= 3, "retained coefficient/frame dimensions")
    close(frame@np.linalg.lstsq(frame, coefficients, rcond=None)[0], coefficients,
          "pilot direction is outside retained frame")
    pilot_s = complex_array(row["pilot_overlap"], "pilot overlap encoding")
    pilot_h = complex_array(row["pilot_hamiltonian"], "pilot Hamiltonian encoding")
    require(pilot_s.shape == pilot_h.shape == (3, 3), "pilot pencil dimensions")
    close(pilot_s, pilot_s.conj().T, "pilot overlap Hermiticity")
    close(pilot_h, pilot_h.conj().T, "pilot Hamiltonian Hermiticity")
    close(frame.conj().T@pilot_s@frame, np.eye(frame.shape[1]), "pilot overlap normalization")
    reduced_h = frame.conj().T@pilot_h@frame
    close(reduced_h, np.diag(np.diag(reduced_h)), "pilot Ritz frame diagonalization")
    require(np.all(np.diff(np.diag(reduced_h).real) >= -1e-9), "pilot roots must be energy ordered")
    close(coefficients, frame[:, 0], "pilot coefficients must be first retained Ritz root")
    close(row["pilot_energy"], reduced_h[0, 0].real, "pilot energy")
    overlap, projected = data["overlap"], data["projected"]
    denominator = float(np.vdot(coefficients, overlap@coefficients).real)
    require(denominator > 1e-12, "pilot direction has zero physical norm")
    truth = float(np.vdot(coefficients, projected@coefficients).real/denominator)
    s, h = frame.conj().T@overlap@frame, frame.conj().T@projected@frame
    values, vectors = np.linalg.eigh((s+s.conj().T)/2)
    keep = values > 1e-10*max(1., float(values.max()))
    require(keep.any(), "retained physical space is empty")
    orthogonal = vectors[:, keep]/np.sqrt(values[keep])
    minimum = float(np.linalg.eigvalsh(orthogonal.conj().T@h@orthogonal)[0])
    for field, expected in (("true_direction_energy", truth), ("retained_space_energy", minimum),
                            ("ground_energy", data["ground"])):
        close(row[field], expected, "independent "+field)
    if row["status"] == "resolved":
        estimate = row["estimate"]
        require(isinstance(estimate, (float, int)) and math.isfinite(estimate), "finite resolved estimate")
        for field, expected in (("estimation_error", abs(estimate-truth)),
                                ("retained_space_error", abs(estimate-minimum)),
                                ("ground_error", abs(estimate-data["ground"]))):
            close(row[field], expected, "independent "+field)
        interval = row["interval"]
        require(len(interval) == 2 and interval[0] <= estimate <= interval[1], "ratio interval")
        require(row["interval_contains_truth"] == (interval[0] <= truth <= interval[1]), "interval coverage flag")
    return truth


@lru_cache(maxsize=12)
def _coefficient_rows(case_json):
    data = independent_case(case_json)
    generators, h = data["generator_terms"], data["h_terms"]
    n = json.loads(case_json)["n"]
    rows = {}
    for i, left in enumerate(generators):
        adjoint = {code: value.conjugate() if isinstance(value, complex) else value
                   for code, value in left.items()}
        for j in range(i, len(generators)):
            right = generators[j]
            rows["overlap", i, j] = multiply(adjoint, right, n)
            rows["element", i, j] = multiply(adjoint, multiply(h, right, n), n)
    return rows


def _functional_terms(case, coefficients, kind):
    rows, total = _coefficient_rows(json.dumps(case, sort_keys=True)), {}
    for i in range(len(coefficients)):
        for j in range(i, len(coefficients)):
            weight = np.conjugate(coefficients[i])*coefficients[j]
            for code, value in rows[kind, i, j].items():
                total[code] = total.get(code, 0.) + float((weight*value).real)*(1 if i == j else 2)
    return total


def verify_ratio_interval(row, case, config, settings):
    coefficients = complex_array(row["coefficients"], "ratio coefficients")
    marginal_intervals = []
    functionals = []
    for kind, field in (("element", "numerator"), ("overlap", "denominator")):
        terms = _functional_terms(case, coefficients, kind)
        functionals.append(terms)
        covered = {0}
        variance_bound = 0.
        for setting, count in zip(settings, row["evaluation_counts"]):
            covered.update(setting.assigned_word_codes)
            width = 2*math.fsum(abs(terms.get(code, 0.)) for code in setting.assigned_word_codes)
            variance_bound += width*width/count
        require(all(abs(value) < 1e-9 or code in covered for code, value in terms.items()),
                "evaluation does not cover frozen functional")
        radius = math.sqrt(.5*math.log(4/config["ratio_delta"])*variance_bound)
        center = row[field]
        require(math.isfinite(center), "finite ratio moment")
        interval = [center-radius, center+radius]
        close(row[field+"_interval"], interval, "independent "+field+" Hoeffding interval")
        marginal_intervals.append(interval)
    h, s = row["numerator"], row["denominator"]
    raw = h/s if s != 0 else None
    if raw is None:
        require(row["raw_ratio"] is None, "zero denominator raw ratio")
    else:
        close(row["raw_ratio"], raw, "raw correlated ratio")
    valid = marginal_intervals[1][0] > config["denominator_floor"]
    if row["status"] == "resolved":
        require(valid, "independent denominator gate")
        close(row["estimate"], raw, "resolved ratio estimate")
        corners = [u/v for u in marginal_intervals[0] for v in marginal_intervals[1]]
        close(row["interval"], [min(corners), max(corners)], "independent four-corner ratio interval")
    elif row["status"] == "denominator_unresolved":
        require(not valid and row["interval"] is None, "unresolved denominator decision")
    moments = row["evaluation_group_moments"]
    require(len(moments) == len(settings), "evaluation group moments coverage")
    means = [functionals[0].get(0, 0.), functionals[1].get(0, 0.)]
    variances, covariance = [0., 0.], 0.
    for moment, setting, shots in zip(moments, settings, row["evaluation_counts"]):
        require(moment["shots"] == shots, "evaluation group moment count")
        require(all(math.isfinite(moment[key]) for key in ("sum_h", "sum_s", "sum_h2", "sum_s2", "sum_hs")),
                "finite joint group moments")
        group_means = [moment["sum_h"]/shots, moment["sum_s"]/shots]
        group_variances = []
        for i, square in enumerate(("sum_h2", "sum_s2")):
            bound = math.fsum(abs(functionals[i].get(code, 0.)) for code in setting.assigned_word_codes)
            require(abs(group_means[i]) <= bound+1e-8, "group functional range")
            variance = moment[square]/shots-group_means[i]**2
            require(variance >= -1e-8, "joint group variance")
            group_variances.append(max(0., variance))
            means[i] += group_means[i]
            variances[i] += max(0., variance)/shots
        group_covariance = moment["sum_hs"]/shots-group_means[0]*group_means[1]
        require(abs(group_covariance) <= math.sqrt(group_variances[0]*group_variances[1])+1e-8,
                "joint group covariance bound")
        covariance += group_covariance/shots
    close(row["numerator"], means[0], "independent numerator mean")
    close(row["denominator"], means[1], "independent denominator mean")
    close(row["numerator_denominator_covariance"], covariance, "independent numerator/denominator covariance")
    variance = None if raw is None else max(0., (variances[0]+raw**2*variances[1]-2*raw*covariance)/s**2)
    if variance is None:
        require(row["delta_variance"] is None, "undefined delta variance")
    else:
        close(row["delta_variance"], variance, "independent ratio delta variance")


def _worker_receipts(record, config):
    receipts = {receipt["case"]: receipt for receipt in record["worker_receipts"]}
    require(len(receipts) == len(record["worker_receipts"]) == len(config["cases"]), "case worker receipt coverage")
    pids = []
    for case in config["cases"]:
        receipt = receipts[case["name"]]
        require(receipt["execution_commit"] == record["execution_commit"]
                and receipt["declaration_commit"] == record["declaration_commit"]
                and receipt["declaration_sha256"] == record["declaration_sha256"], "worker execution binding")
        scheduled = len(config["total_shot_budgets"])*config["replicas"]*2
        observed = receipt["observed_outcomes"]
        require(type(observed) is int and 0 <= observed <= scheduled
                and receipt["scheduled_outcomes"] == scheduled
                and receipt["retained_observations"] == observed
                and receipt["imputed_failures"] == scheduled-observed, "worker observation denominator")
        case_rows = [row for row in record["rows"] if row["case"] == case["name"]]
        require(sum(row["status"] == "worker_failed" for row in case_rows) == scheduled-observed,
                "worker synthetic failure count")
        require(receipt["case_timeout_seconds"] == config["case_timeout_seconds"]
                and type(receipt["timed_out"]) is bool, "worker timeout policy")
        require(receipt["source_manifest_sha256"] == digest_bytes(json.dumps(
                    record["source_sha256"], sort_keys=True).encode()), "worker scientific source manifest")
        pid = receipt["worker_pid"]
        require(type(pid) is int and pid > 0, "worker process identity")
        pids.append(pid)
        require(type(receipt["completed"]) is bool, "worker completion flag")
        if receipt["completed"]:
            require(receipt["returncode"] == 0 and isinstance(receipt["checkpoint_sha256"], str)
                    and len(receipt["checkpoint_sha256"]) == 64 and observed == scheduled
                    and not receipt["timed_out"], "completed worker receipt")
        require(math.isfinite(receipt["worker_seconds"]) and receipt["worker_seconds"] >= 0,
                "worker elapsed time")
    require(len(set(pids)) == len(pids), "case workers must be distinct fresh processes")
    return receipts


def _check(record):
    config = json.loads((ROOT/DECLARATION).read_text())
    validate_declaration(config)
    require(record["schema"] == "clifford_qc.cross_domain_ratio_results.v1", "result schema")
    check_history(record, config)
    receipts = _worker_receipts(record, config)
    expected = {(case["name"], budget, replica, arm) for case in config["cases"]
                for budget in config["total_shot_budgets"] for replica in range(config["replicas"])
                for arm in ARMS}
    rows = record["rows"]
    keys = [(row["case"], row["budget"], row["replica"], row["arm"]) for row in rows]
    require(len(keys) == len(expected) and set(keys) == expected, "all scheduled attempts, including failures, required")
    entries = {entry["case"]: entry for entry in record["inputs"]}
    require(len(entries) == len(record["inputs"]) == len(config["cases"]), "unique input coverage")
    for ci, case in enumerate(config["cases"]):
        entry = entries[case["name"]]
        case_rows = [row for row in rows if row["case"] == case["name"]]
        unavailable = entry.get("status") in ("resource_failed", "worker_failed")
        provider, data = (None, None) if unavailable else (structural_provider(case), case_data(case))
        if unavailable:
            require(all(row["status"] in ("resource_failed", "worker_failed") and row["total_shots"] == 0
                        for row in case_rows), "unavailable input must retain all failed attempts")
        else:
            require(entry["kind"] == case["kind"] and entry["unit"] == case["unit"] and entry["n"] == case["n"]
                    and entry["target_estimation_abs_error"] == case["target_estimation_abs_error"], "input units/target")
            require(entry["sector_words"] == data["words"], "independent occupation sector")
            require(entry["reference_identity"] == provider.reference_identity
                    and entry["hamiltonian_identity"] == provider.hamiltonian_identity, "input physical identities")
            compare_payload(entry["settings"], settings_payload(provider.plan.settings), "compiled settings")
            compare_payload(entry["preparation"], preparation_stats(case), "preparation resources")
            require(entry["admitted_array_bytes"] == array_bound(case["n"])
                    <= config["owned_array_cap_bytes"], "input array admission")
        for row in case_rows:
            if row["status"] != "worker_failed":
                require(row["worker_pid"] == receipts[case["name"]]["worker_pid"], "row worker identity")
            require(row["unit"] == case["unit"]
                    and row["target_estimation_abs_error"] == case["target_estimation_abs_error"], "row units/target")
            bi = config["total_shot_budgets"].index(row["budget"])
            seed = config["master_seed"] + 100000*ci + 1000*bi + 3*row["replica"]
            require(row["pilot_seed"] == seed and row["evaluation_seed"] == seed+(1 if row["arm"] == "ratio" else 2),
                    "independent declared seed streams")
            require(row["status"] in ("resolved", "denominator_unresolved", "pilot_failed", "resource_failed",
                                      "evaluation_failed", "assessment_failed", "worker_failed"), "attempt status")
            require(row["total_shots"] == row["pilot_shots"]+row["evaluation_shots"]
                    and 0 <= row["total_shots"] <= row["budget"], "physical-shot accounting")
            require(type(row["costs_complete"]) is bool and type(row["physical_shots_unknown"]) is bool,
                    "physical accounting completeness flags")
            attempted = row["attempted_physical_shots"]
            if row["status"] == "worker_failed":
                require(attempted is None and row["physical_shots_unknown"]
                        and not row["costs_complete"] and row["total_shots"] == 0,
                        "unobserved worker spending must remain unknown")
            else:
                require(type(attempted) is int and row["total_shots"] <= attempted <= row["budget"],
                        "attempted physical shots")
            if row["physical_shots_unknown"]:
                require(not row["costs_complete"] and row["status"] != "resolved"
                        and (attempted is None or row["total_shots"] < attempted), "unknown shots cannot resolve")
            require(row["admitted_array_bytes"] == array_bound(case["n"]), "row array model")
            for field in ("classical_seconds", "simulator_seconds", "assessment_seconds"):
                require(math.isfinite(row[field]) and row[field] >= 0, "nonnegative timing: "+field)
            if row["costs_complete"]:
                require(math.isfinite(row["device_seconds"]) and row["device_seconds"] >= 0, "complete device cost")
                close(row["end_to_end_seconds"], row["device_seconds"]+row["classical_seconds"], "end-to-end costs")
            else:
                require(row["device_seconds"] is None and row["end_to_end_seconds"] is None
                        and row["status"] != "resolved", "incomplete cost must remain unavailable")
            pilot_counts = row.get("pilot_counts", [])
            expected_pilot = int(row["budget"]*config["pilot_fraction"])
            if pilot_counts:
                require(row["pilot_shots"] == expected_pilot and len(pilot_counts) == len(provider.plan.settings)
                        and sum(pilot_counts) == expected_pilot, "pilot counts")
                quotient, remainder = divmod(expected_pilot, len(provider.plan.settings))
                require(pilot_counts == [quotient+(i < remainder) for i in range(len(provider.plan.settings))], "uniform pilot schedule")
            counts, indices = row.get("evaluation_counts", []), row.get("evaluation_settings", [])
            require(len(counts) == len(indices), "evaluation schedule dimensions")
            require(row["evaluation_shots"] == 0 or sum(counts) == row["evaluation_shots"], "evaluation schedule accounting")
            require(all(type(i) is int and 0 <= i < len(provider.plan.settings) for i in indices)
                    and len(set(indices)) == len(indices), "evaluation setting indices")
            require(all(type(count) is int and count >= 2 for count in counts), "minimum evaluation coverage")
            if row["status"] in ("resolved", "denominator_unresolved"):
                require(row["total_shots"] == row["budget"], "full attempted nondeterministic budget")
                require(row["costs_complete"] and not row["physical_shots_unknown"] and pilot_counts,
                        "evaluated result needs complete cost and pilot schedule")
                verify_assessment(row, case, data)
                verify_ratio_interval(row, case, config, [provider.plan.settings[i] for i in indices])
            if row["status"] == "resolved":
                require(row["denominator_interval"][0] > config["denominator_floor"], "resolved denominator gate")
                require(row["process_peak_rss_bytes"] <= config["process_rss_cap_bytes"], "recorded RSS breach")
            else:
                require(row["estimate"] is None, "failed attempt has an estimate")
            device = device_seconds(provider.plan.settings, pilot_counts, case, config) if pilot_counts else 0.
            if row["evaluation_shots"]:
                device += device_seconds([provider.plan.settings[i] for i in indices], counts, case, config)
            if row["costs_complete"] and (row["status"] in ("resolved", "denominator_unresolved") or not row["total_shots"]):
                close(row["device_seconds"], device, "preparation/readout/device price", atol=1e-12)
    summaries = independent_summaries(rows, config)
    compare_payload(record["summaries"], summaries, "unconditional summaries")
    compare_payload(record["verdict"], independent_verdict(summaries, config), "frozen criterion/verdict")
    return len(rows)


def check(record):
    try:
        return _check(record)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError, subprocess.CalledProcessError) as error:
        raise AssertionError("malformed or inconsistent cross-domain record: "+str(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT/RESULT)
    args = parser.parse_args(argv)
    count = check(load_record(args.record))
    print(f"cross-domain independent science/chronology/accounting passed: {count} unconditional outcomes")


if __name__ == "__main__":
    main()

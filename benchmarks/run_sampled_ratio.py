"""Execute/replay the preregistered restart Step 5 IID Pauli-shot comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clifford_qc.backends.protocol import GroupSample, MeasurementBatch
from clifford_qc.measurement.ratio import evaluate_ratio, uniform_setting_plan
from clifford_qc.models import tfim
from clifford_qc.multivector import MV
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.subspace.linalg import solve_projected
from clifford_qc.subspace.sampled import SampledProjectedProvider

DECLARATION = "docs/restart/sampled_ratio_declaration.json"
RESULT = "docs/restart/sampled_ratio_results.json"
SOURCES = ("clifford_qc/sampled_reference.py", "clifford_qc/measurement/ratio.py",
           "clifford_qc/measurement/functionals.py", "clifford_qc/subspace/sampled.py",
           "clifford_qc/subspace/projection.py", "clifford_qc/measurement/cache.py",
           "clifford_qc/measurement/session.py", "clifford_qc/measurement/graph_state.py",
           "clifford_qc/measurement/planning.py", "clifford_qc/subspace/linalg.py",
           "benchmarks/run_sampled_ratio.py", "benchmarks/check_sampled_ratio.py")


def digest(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_declaration():
    config = json.loads((ROOT/DECLARATION).read_text())
    if (config["schema"] != "clifford_qc.sampled_ratio_declaration.v1"
            or config["generators"] != ["I", "H/n", "Z0"]
            or config["compiler"] != "graph_state" or config["pooling"] != "assigned"
            or config["block_size"] != "full_register" or config["storage"] != "packed"):
        raise ValueError("unsupported declared experiment")
    if len({c["name"] for c in config["cases"]}) != len(config["cases"]):
        raise ValueError("case names must be unique")
    for case in config["cases"]:
        if not 2 <= case["n"] <= 6 or case["role"] not in ("development_primary", "held_out"):
            raise ValueError("experiment is admitted only for declared small spin inputs")
    if config["replicas"] != 128 or config["total_shot_budgets"] != [4096, 16384, 65536]:
        raise ValueError("frozen replica/budget grid changed")
    return config


def declaration_commit():
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", DECLARATION, *SOURCES],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    # Untracked declarations have no HEAD blob and must not start a campaign.
    git("cat-file", "-e", f"HEAD:{DECLARATION}")
    return git("log", "-1", "--format=%H", "--", DECLARATION)


def dense_word(n, code):
    """Independent tensor-Pauli assessment; no native products or pairing."""
    local = (np.eye(2), np.array([[0, 1], [1, 0]]),
             np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]))
    out = np.ones((1, 1), dtype=complex)
    for q in range(n):
        out = np.kron(out, local[(code >> (2*q)) & 3])
    return out


def dense_operator(operator):
    return sum((value*dense_word(operator.n, code) for code, value in operator.terms.items()),
               np.zeros((2**operator.n, 2**operator.n), dtype=complex))


def build_case(case):
    n = case["n"]
    h = tfim(n, J=case["J"], h=case["h"]).hamiltonian.to_mv()
    state = np.ones(1, dtype=complex)
    for q in range(n):
        theta, phi = case["theta"]+0.1*(q % 2), (-1)**q*case["phi"]
        state = np.kron(state, [math.cos(theta/2), np.exp(1j*phi)*math.sin(theta/2)])
    generators = [("I", MV.scalar(n, 1)), ("H/n", h*(1/n)), ("Z0", MV(n, {3: 1}))]
    return h, generators, state


class JointShotSimulator:
    """Small statevector measurement device; never passed to inference as oracle.

    Only per-setting probabilities are cached. Each call draws fresh joint
    multinomial histograms, never independent word or Gaussian surrogates.
    Native gate matrices are applied independently of Pauli conjugation.
    """
    def __init__(self, state, n):
        self.state = state.copy()
        self.n = n
        self.tables = {}
        self.draw_seconds = 0.0

    def probabilities(self, setting):
        key = setting.clifford.identity
        if key not in self.tables:
            state = self.state.copy()
            index = np.arange(len(state))
            for name, qubits in setting.clifford.gates:
                mask = 1 << (self.n-1-qubits[0])
                if name == "CZ":
                    second = 1 << (self.n-1-qubits[1])
                    state[(index & mask != 0) & (index & second != 0)] *= -1
                elif name == "H":
                    low = index[index & mask == 0]
                    high = low | mask
                    a, b = state[low].copy(), state[high].copy()
                    state[low], state[high] = (a+b)/math.sqrt(2), (a-b)/math.sqrt(2)
                elif name in ("S", "SDG"):
                    state[index & mask != 0] *= 1j if name == "S" else -1j
                else:
                    raise ValueError("unsupported simulator gate")
            probabilities = np.abs(state)**2
            if not np.isclose(probabilities.sum(), 1, atol=1e-12):
                raise ValueError("measurement state lost normalization")
            self.tables[key] = probabilities/probabilities.sum()
        return self.tables[key]

    def draw(self, settings, counts, seed, identity):
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        groups, shots, plus = [], {}, {}
        for setting, count in zip(settings, counts):
            if not count:
                continue
            drawn = rng.multinomial(count, self.probabilities(setting))
            hist = {format(i, f"0{self.n}b"): int(value)
                    for i, value in enumerate(drawn) if value}
            groups.append(GroupSample(tuple(range(self.n)), (), hist, count,
                                      setting.assigned_word_codes, setting.key, setting.readouts))
            for code in setting.assigned_word_codes:
                sign, positions = setting.readouts[code]
                shots[code] = count
                plus[code] = sum(value for bits, value in hist.items()
                                 if sign*(-1)**sum(bits[p] == "1" for p in positions) == 1)
        result = MeasurementBatch(self.n, shots, plus, len(groups), tuple(groups), identity)
        self.draw_seconds += time.perf_counter()-started
        return result


def device_seconds(settings, counts, config, n):
    from clifford_qc.measurement.graph_state import native_stats
    card = config["device_scenario_us"]
    total = 0.0
    for setting, count in zip(settings, counts):
        _, _, d1, d2 = native_stats(setting.clifford)
        total += count*(n*card["preparation_per_qubit"] + card["readout_reset"]
                        + d1*card["one_qubit_layer"] + d2*card["two_qubit_layer"])
    return total*1e-6


def exact_binomial_lower(successes, attempts, alpha):
    """One-sided Clopper-Pearson inversion, using only the standard library."""
    if successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(64):
        p = (low+high)/2
        tail = sum(math.comb(attempts, k)*p**k*(1-p)**(attempts-k)
                   for k in range(successes, attempts+1))
        if tail < alpha:
            low = p
        else:
            high = p
    return (low+high)/2


def summarize(rows, config):
    summaries = []
    family = len(config["cases"])*len(config["total_shot_budgets"])*2
    for case in config["cases"]:
        for budget in config["total_shot_budgets"]:
            for arm in ("ratio", "bank"):
                group = [r for r in rows if (r["case"], r["budget"], r["arm"])
                         == (case["name"], budget, arm)]
                successes = sum(r["status"] == "resolved" and
                                r["estimation_error"] <= config["target_estimation_abs_error"]
                                for r in group)
                summaries.append({
                    "case": case["name"], "role": case["role"], "budget": budget, "arm": arm,
                    "attempts": len(group), "successes": successes,
                    "success_probability": successes/len(group),
                    "success_probability_lower": exact_binomial_lower(
                        successes, len(group), config["success_probability_alpha"]/family),
                    "pilot_failures": sum(r["status"] == "pilot_failed" for r in group),
                    "denominator_failures": sum(r["status"] == "denominator_unresolved" for r in group),
                    "resource_failures": sum(r["status"] == "resource_failed" for r in group),
                    "resolved": sum(r["status"] == "resolved" for r in group),
                    "resolved_interval_misses": sum(r.get("interval_contains_truth") is False for r in group),
                    "median_end_to_end_seconds": statistics.median(r["end_to_end_seconds"] for r in group),
                    "median_device_seconds": statistics.median(r["device_seconds"] for r in group),
                    "median_classical_seconds": statistics.median(r["classical_seconds"] for r in group),
                    "median_total_shots": statistics.median(r["total_shots"] for r in group),
                    "median_resolved_estimation_error": _median_present(group, "estimation_error"),
                    "median_resolved_retained_space_error": _median_present(group, "retained_space_error"),
                    "median_resolved_ground_error": _median_present(group, "ground_error"),
                })
    return summaries


def _median_present(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None and row["status"] == "resolved"]
    return statistics.median(values) if values else None


def verdict(summaries, config):
    comparisons = []
    for case in config["cases"]:
        eligible = {}
        for arm in ("ratio", "bank"):
            points = [r for r in summaries if r["case"] == case["name"] and r["arm"] == arm
                      and r["success_probability_lower"] >= config["target_success_probability"]
                      and r["resource_failures"] == 0]
            eligible[arm] = min(points, key=lambda r: r["median_end_to_end_seconds"]) if points else None
        a, b = eligible["ratio"], eligible["bank"]
        saving = 1-a["median_end_to_end_seconds"]/b["median_end_to_end_seconds"] if a and b else None
        comparisons.append({"case": case["name"],
                            "ratio_budget": a["budget"] if a else None,
                            "bank_budget": b["budget"] if b else None,
                            "cost_reduction": saving, "passes": saving is not None and saving >= 0.2})
    return {"status": "promote_on_declared_scope" if all(c["passes"] for c in comparisons)
            else "closed_on_declared_scope", "comparisons": comparisons,
            "quantum_advantage_claim": False, "hardware_claim": False}


def array_admission_bound(n):
    """Conservative pre-allocation bound for this <=6-qubit, three-row family.

    Twelve S/H rows, up to 4**n words each, doubled packed capacities and
    coefficient-map payload allowance; worst-case setting probability tables;
    eight dense workspaces plus a one-MiB small-array/sorting allowance.
    Python object headers/allocator overhead remain under the separate RSS cap.
    """
    dimension = 2**n
    return 128*12*4**n + 16*4**n*dimension + 8*16*dimension**2 + 1048576


def setup(case, config):
    admitted = array_admission_bound(case["n"])
    if admitted > config["owned_array_cap_bytes"]:
        raise MemoryError("owned-array admission failed before input allocation")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > config["process_rss_cap_bytes"]:
        raise MemoryError("process RSS ceiling already exceeded")
    h, generators, state = build_case(case)
    simulator = JointShotSimulator(state, case["n"])
    preparation = json.dumps({k: case[k] for k in ("n", "theta", "phi")}, sort_keys=True)
    reference = PreparedSampledReference(case["n"], preparation, simulator.draw)
    start = time.perf_counter()
    provider = SampledProjectedProvider(reference, h, generators)
    compile_seconds = time.perf_counter()-start
    return h, generators, state, reference, provider, compile_seconds, admitted, simulator


def run(config, execution_commit, declared_commit):
    rows, inputs = [], []
    for ci, case in enumerate(config["cases"]):
        setup_start = time.perf_counter()
        try:
            h, gens, state, reference, provider, compile_s, admitted, simulator = setup(case, config)
        except MemoryError as error:
            elapsed = time.perf_counter()-setup_start
            inputs.append({"case": case["name"], "status": "resource_failed",
                           "failure_reason": str(error)})
            for bi, budget in enumerate(config["total_shot_budgets"]):
                for replica in range(config["replicas"]):
                    seed = config["master_seed"]+100000*ci+1000*bi+3*replica
                    for arm in ("ratio", "bank"):
                        rows.append({"case": case["name"], "budget": budget, "replica": replica,
                                     "arm": arm, "status": "resource_failed", "failure_reason": str(error),
                                     "pilot_seed": seed, "evaluation_seed": seed+(1 if arm == "ratio" else 2),
                                     "pilot_shots": 0, "evaluation_shots": 0, "total_shots": 0,
                                     "estimate": None, "estimation_error": None, "retained_space_error": None,
                                     "ground_error": None, "device_seconds": 0.0, "classical_seconds": elapsed,
                                     "end_to_end_seconds": elapsed, "simulator_seconds": 0.0,
                                     "assessment_seconds": 0.0,
                                     "admitted_array_bytes": array_admission_bound(case["n"])})
            continue
        inputs.append({"case": case["name"], "reference_identity": provider.reference_identity,
                       "hamiltonian_identity": provider.hamiltonian_identity,
                       "provider_cache_identity": provider.cache_identity,
                       "settings": len(provider.plan.settings), "words": len(provider.plan.codes),
                       "coefficient_occurrences": provider.bank.resources()["coefficient_occurrences"],
                       "admitted_array_bytes": admitted, "compile_seconds": compile_s})
        for bi, budget in enumerate(config["total_shot_budgets"]):
            pilot_budget = int(budget*config["pilot_fraction"])
            for replica in range(config["replicas"]):
                base = config["master_seed"] + 100000*ci + 1000*bi + 3*replica
                start = time.perf_counter()
                before_draw = simulator.draw_seconds
                pilot_counts = uniform_setting_plan(provider.plan.settings, pilot_budget)
                pilot = reference.sample(provider.plan.settings, pilot_counts, seed=base)
                pilot_elapsed_s = time.perf_counter()-start
                pilot_draw_s = simulator.draw_seconds-before_draw
                pilot_validation_s = max(0.0, pilot_elapsed_s-pilot_draw_s)
                pilot_device_s = device_seconds(provider.plan.settings, pilot_counts, config, case["n"])
                outcomes = {}
                arms = ("ratio", "bank") if replica % 2 == 0 else ("bank", "ratio")
                for arm in arms:
                    row = {"case": case["name"], "budget": budget, "replica": replica, "arm": arm,
                           "pilot_seed": base, "evaluation_seed": base+(1 if arm == "ratio" else 2),
                           "pilot_shots": pilot_budget, "evaluation_shots": 0, "total_shots": pilot_budget,
                           "pilot_identity": pilot.identity, "status": "pilot_failed",
                           "estimation_error": None, "retained_space_error": None, "ground_error": None,
                           "estimate": None, "device_seconds": pilot_device_s,
                           "classical_seconds": compile_s+pilot_validation_s, "simulator_seconds": pilot_draw_s,
                           "assessment_seconds": 0.0, "admitted_array_bytes": admitted}
                    start = time.perf_counter()
                    try:
                        frozen = provider.freeze_ratio(
                            pilot, budget-pilot_budget, tau_s=config["tau_s"],
                            max_condition=config["max_condition"], delta=config["ratio_delta"],
                            denominator_floor=config["denominator_floor"], allocation=arm)
                    except (ValueError, np.linalg.LinAlgError, MemoryError) as error:
                        if isinstance(error, MemoryError):
                            row["status"] = "resource_failed"
                        row["failure_reason"] = str(error)
                        row["classical_seconds"] += time.perf_counter()-start
                        row["end_to_end_seconds"] = row["device_seconds"]+row["classical_seconds"]
                        outcomes[arm] = row
                        continue
                    row["classical_seconds"] += time.perf_counter()-start
                    start = time.perf_counter()
                    before_draw = simulator.draw_seconds
                    observed = reference.sample(frozen.settings, frozen.counts, seed=row["evaluation_seed"])
                    evaluation_elapsed_s = time.perf_counter()-start
                    draw_s = simulator.draw_seconds-before_draw
                    row["simulator_seconds"] += draw_s
                    row["classical_seconds"] += max(0.0, evaluation_elapsed_s-draw_s)
                    row["device_seconds"] += device_seconds(frozen.settings, frozen.counts, config, case["n"])
                    start = time.perf_counter()
                    reconstruction = provider.reconstruct(observed) if arm == "bank" else None
                    result = evaluate_ratio(frozen, observed)
                    if reconstruction is not None and result.raw_ratio is not None:
                        c = frozen.coefficients
                        s, hh = reconstruction.problem.overlap, reconstruction.problem.hamiltonian
                        rebuilt = float((c.conj()@hh@c).real/(c.conj()@s@c).real)
                        if not math.isclose(rebuilt, result.raw_ratio, rel_tol=1e-10, abs_tol=1e-10):
                            raise AssertionError("complete-bank contraction differs from joint functional ratio")
                    row["classical_seconds"] += time.perf_counter()-start
                    row.update(asdict(result))
                    row["evaluation_counts"] = list(frozen.counts)
                    row["evaluation_settings"] = [i for i, setting in enumerate(provider.plan.settings)
                                                  if setting.key in {s.key for s in frozen.settings}]
                    row["pilot_energy"] = frozen.pilot_energy
                    row["retained_rank"] = frozen.retained_frame.shape[1]
                    row["final_functional_words"] = len(set(frozen.numerator.coefficients) | set(frozen.denominator.coefficients))
                    # Retained coefficient-map footprint model (code/value + table).
                    row["final_functional_bytes"] = sum(
                        sys.getsizeof(dict(f.coefficients)) + sum(sys.getsizeof(k)+sys.getsizeof(v)
                        for k, v in f.coefficients.items()) for f in (frozen.numerator, frozen.denominator))
                    # Only assessment below sees amplitudes or exact matrices.
                    start = time.perf_counter()
                    dense_h = -case["J"]*sum((dense_word(case["n"], (3 << (2*q)) | (3 << (2*(q+1))))
                                for q in range(case["n"]-1)), np.zeros((len(state), len(state)), complex))
                    dense_h -= case["h"]*sum((dense_word(case["n"], 1 << (2*q))
                                             for q in range(case["n"])), np.zeros_like(dense_h))
                    basis = np.column_stack([dense_operator(g)@state for _, g in gens])
                    exact_s, exact_h = basis.conj().T@basis, basis.conj().T@dense_h@basis
                    c, frame = frozen.coefficients, frozen.retained_frame
                    truth = float((c.conj()@exact_h@c).real/(c.conj()@exact_s@c).real)
                    retained = solve_projected(frame.conj().T@exact_s@frame,
                                               frame.conj().T@exact_h@frame).ground_energy
                    ground = float(np.linalg.eigvalsh(dense_h)[0])
                    row.update({"true_direction_energy": truth, "retained_space_energy": retained,
                                "ground_energy": ground,
                                "estimation_error": abs(result.estimate-truth) if result.estimate is not None else None,
                                "retained_space_error": abs(result.estimate-retained) if result.estimate is not None else None,
                                "ground_error": abs(result.estimate-ground) if result.estimate is not None else None,
                                "interval_contains_truth": result.interval[0] <= truth <= result.interval[1]
                                if result.interval is not None else None})
                    row["assessment_seconds"] = time.perf_counter()-start
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                    row["process_peak_rss_bytes"] = rss
                    if rss > config["process_rss_cap_bytes"]:
                        row["status"] = "resource_failed"
                    row["end_to_end_seconds"] = row["device_seconds"]+row["classical_seconds"]
                    outcomes[arm] = row
                rows.extend(outcomes[arm] for arm in ("ratio", "bank"))
            print(f'{case["name"]}: {budget} shots, {config["replicas"]} replicas per arm', file=sys.stderr, flush=True)
    summaries = summarize(rows, config)
    return {"schema": "clifford_qc.sampled_ratio_results.v1", "declaration_sha256": digest(DECLARATION),
            "declaration_commit": declared_commit, "execution_commit": execution_commit,
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "source_sha256": {p: digest(p) for p in SOURCES}, "inputs": inputs,
            "rows": rows, "summaries": summaries, "verdict": verdict(summaries, config)}


def compare_records(expected, actual):
    """Replay values tightly; timing/RSS/provenance identify an execution only."""
    advisory = {"source_sha256", "environment", "execution_commit", "process_peak_rss_bytes"}

    def compare(a, b, path="record"):
        if isinstance(a, dict):
            ka = {k for k in a if k not in advisory and not k.endswith("seconds")}
            kb = {k for k in b if k not in advisory and not k.endswith("seconds")}
            if path == "record":
                ka -= {"verdict"}; kb -= {"verdict"}  # Actual timing can change the cost verdict.
            if ka != kb:
                raise AssertionError(f"{path}: record fields differ")
            for key in ka:
                compare(a[key], b[key], path+"."+key)
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise AssertionError(f"{path}: lengths differ")
            for i, (left, right) in enumerate(zip(a, b)):
                compare(left, right, f"{path}[{i}]")
        elif isinstance(a, float):
            if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
                raise AssertionError(f"{path}: numeric replay differs")
        elif a != b:
            raise AssertionError(f"{path}: replay differs")
    compare(expected, actual)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true", help="check declaration/builds without drawing outcomes")
    mode.add_argument("--execute", action="store_true", help="execute once after committing the declaration")
    mode.add_argument("--check", type=Path, help="fully replay an existing record")
    parser.add_argument("--output", type=Path, default=ROOT/RESULT)
    args = parser.parse_args(argv)
    config = load_declaration()
    if args.validate:
        for case in config["cases"]:
            _, _, _, _, provider, _, admitted, _ = setup(case, config)
            print(f'{case["name"]}: {len(provider.plan.settings)} settings; {admitted} admitted bytes; no shots drawn')
        return
    declared_commit = declaration_commit()
    if args.execute and args.output.exists():
        raise FileExistsError("observation record exists; use --check to replay it")
    result = run(config, git("rev-parse", "HEAD"), declared_commit)
    if args.check:
        compare_records(json.loads(args.check.read_text()), result)
        print("sampled ratio full replay passed")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result["verdict"], indent=2))


if __name__ == "__main__":
    main()

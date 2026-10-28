"""One declared cross-domain IID-shot validation with isolated case workers.

The historical spin-only experiment is never rewritten. No spectrum or exact
variance is supplied to sampled projection, pilot selection, or allocation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_sampled_ratio import (
    JointShotSimulator, dense_operator, dense_word, exact_binomial_lower,
)
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.measurement.graph_state import native_stats
from clifford_qc.measurement.ratio import evaluate_ratio, uniform_setting_plan
from clifford_qc.models import tfim
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.multivector import MV
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.subspace.sampled import SampledProjectedProvider

DECLARATION = "docs/restart/cross_domain_ratio_declaration.json"
RESULT = "docs/restart/cross_domain_ratio_results.json.gz"
BENCHMARK_SOURCES = (
    "benchmarks/run_cross_domain_ratio.py", "benchmarks/check_cross_domain_ratio.py",
    "benchmarks/run_sampled_ratio.py", "benchmarks/phase17_ceo_oracle.py",
    "benchmarks/check_orthogonal_active_space.py", "benchmarks/archive_provenance.py",
    "benchmarks/metadata_io.py",
)
FAILURES = frozenset({"pilot_failed", "denominator_unresolved", "resource_failed",
                      "evaluation_failed", "assessment_failed", "worker_failed"})


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def digest(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def scientific_sources(config=None):
    config = load_declaration() if config is None else config
    sources = tuple(config["source_paths"])
    package = {p for p in git("ls-files", "clifford_qc").splitlines() if p.endswith(".py")}
    if len(set(sources)) != len(sources) or not (package | set(BENCHMARK_SOURCES)).issubset(sources):
        raise ValueError("scientific source closure is incomplete")
    return sources


def load_declaration(path=None):
    """Check the fixed protocol and identities without drawing any outcomes."""
    c = json.loads((ROOT/DECLARATION if path is None else Path(path)).read_text())
    if c["schema"] != "clifford_qc.cross_domain_ratio_declaration.v1":
        raise ValueError("unknown cross-domain declaration")
    if (c["total_shot_budgets"] != [16384, 65536, 262144]
            or c["replicas"] != 128 or c["master_seed"] != 20261028
            or c["pilot_fraction"] != .25 or c["compiler"] != "graph_state"
            or c["storage"] != "packed" or c["pooling"] != "assigned"
            or c["tau_s"] != .05 or c["max_condition"] != 10000.
            or c["ratio_delta"] != .05 or c["denominator_floor"] != .05
            or c["target_success_probability"] != .9
            or c["owned_array_cap_bytes"] != 512*2**20
            or c["process_rss_cap_bytes"] != 1024*2**20
            or c.get("case_timeout_seconds") != 1800
            or c.get("result_format") != "json.gz;gzip_mtime=0"
            or c["block_size"] != "full_register" or c["success_probability_alpha"] != .05
            or c["generators"] != {"tfim": ["I", "H/n", "Z0"],
                "molecule": ["I", "(I+G12)/sqrt(2)", "(I+G13)/sqrt(2)"]}):
        raise ValueError("fixed cross-domain protocol changed")
    expected = [("tfim4_development", "tfim", 4, "t", .05),
                ("h4_heldout_molecule", "molecule", 8, "Ha", .0016),
                ("lih_heldout_molecule", "molecule", 8, "Ha", .0016)]
    if [(x["name"], x["kind"], x["n"], x["unit"], x["target_estimation_abs_error"])
            for x in c["cases"]] != expected:
        raise ValueError("declared cross-domain case grid changed")
    paths = ["benchmarks/data/h4_sto3g_r0.9.FCIDUMP", "benchmarks/data/lih_sto3g_r1.5949_cas4e4o.FCIDUMP"]
    if [x["role"] for x in c["cases"]] != ["development_primary", "held_out_molecule", "held_out_molecule"]:
        raise ValueError("declared case roles changed")
    if [x["fcidump"] for x in c["cases"] if x["kind"] == "molecule"] != paths:
        raise ValueError("declared molecular inputs changed")
    historical_paths = {"docs/restart/blueprint_2026-10-24.md", "docs/restart/sampled_ratio_declaration.json",
                        "docs/restart/sampled_ratio_results.json", "docs/restart/v1_evidence_snapshot.json"}
    if set(c["historical_bindings"]) != historical_paths:
        raise ValueError("historical evidence binding set changed")
    for case in c["cases"]:
        if case["kind"] == "molecule":
            if (case["n_electrons"] != 4 or case["sz"] != 0
                    or case["target_spin"] != 0 or case["pair_source"] != 1
                    or case["pair_targets"] != [2, 3]
                    or case["reference_bits"] != ["11110000", "11001100"]
                    or case["theta"] != .3 or case["phi"] != .2):
                raise ValueError("fixed molecular reference or correction changed")
            if digest(case["fcidump"]) != case["fcidump_sha256"]:
                raise ValueError("molecular FCIDUMP identity changed")
        elif (case["J"], case["h"], case["theta"], case["phi"]) != (1., 1.2, 1.1, .2):
            raise ValueError("development control changed")
    for path, expected_hash in c["historical_bindings"].items():
        if digest(path) != expected_hash:
            raise ValueError(f"historical evidence changed: {path}")
    return c


def array_admission_bound(n):
    return 128*12*4**n + 16*4**n*2**n + 8*16*4**n + 2**20


def encode_complex(value):
    a = np.asarray(value, complex)
    return np.stack((a.real, a.imag), axis=-1).tolist()


def decode_complex(value):
    a = np.asarray(value, float)
    return a[..., 0]+1j*a[..., 1]


def preparation_identity(case):
    keys = ("kind", "n", "theta", "phi") + (("reference_bits",) if case["kind"] == "molecule" else ())
    return json.dumps({k: case[k] for k in keys}, sort_keys=True)


def preparation_resources(case):
    if case["kind"] == "molecule":
        return {"one_qubit_gates": 6, "two_qubit_gates": 3,
                "one_qubit_depth": 3, "two_qubit_depth": 3}
    return {"one_qubit_gates": 2*case["n"], "two_qubit_gates": 0,
            "one_qubit_depth": 2, "two_qubit_depth": 0}


def preparation_circuit(case):
    if case["kind"] == "molecule":
        return [("X", (0,), 0.), ("X", (1,), 0.), ("RY", (2,), 2*case["theta"]),
                ("RZ", (2,), case["phi"]), ("CX", (2, 3), 0.), ("CX", (2, 4), 0.),
                ("CX", (2, 5), 0.), ("X", (2,), 0.), ("X", (3,), 0.)]
    return [(name, (q,), angle) for name in ("RY", "RZ") for q in range(case["n"])
            for angle in ((case["theta"]+.1*(q % 2)) if name == "RY" else (-1)**q*case["phi"],)]


def preparation_state_from_circuit(case):
    n = case["n"]
    state = np.zeros(2**n, complex); state[0] = 1
    indexes = np.arange(2**n)
    for name, qubits, angle in preparation_circuit(case):
        mask = 1 << (n-1-qubits[0])
        if name == "CX":
            target = 1 << (n-1-qubits[1])
            low = indexes[(indexes & mask != 0) & (indexes & target == 0)]
            high = low | target
            state[low], state[high] = state[high].copy(), state[low].copy()
        elif name == "RZ":
            state[indexes & mask == 0] *= np.exp(-.5j*angle)
            state[indexes & mask != 0] *= np.exp(.5j*angle)
        else:
            low = indexes[indexes & mask == 0]; high = low | mask
            a, b = state[low].copy(), state[high].copy()
            if name == "X":
                state[low], state[high] = b, a
            elif name == "RY":
                c, t = math.cos(angle/2), math.sin(angle/2)
                state[low], state[high] = c*a-t*b, t*a+c*b
            else:
                raise ValueError("unsupported declared preparation gate")
    return state


def build_case(case):
    n = case["n"]
    if case["kind"] == "tfim":
        h = tfim(n, J=case["J"], h=case["h"]).hamiltonian.to_mv()
        state = np.ones(1, complex)
        for q in range(n):
            theta, phi = case["theta"]+.1*(q % 2), (-1)**q*case["phi"]
            state = np.kron(state, [math.cos(theta/2), np.exp(1j*phi)*math.sin(theta/2)])
        generators = [("I", MV.scalar(n, 1)), ("H/n", h*(1/n)), ("Z0", MV(n, {3: 1}))]
        return h, generators, state
    if case["kind"] != "molecule":
        raise ValueError("unsupported case kind")
    if digest(case["fcidump"]) != case["fcidump_sha256"]:
        raise ValueError("molecular input bytes differ")
    model = fcidump_model(ROOT/case["fcidump"])
    if model.n != n or model.metadata["n_electrons"] != case["n_electrons"]:
        raise ValueError("FCIDUMP sector differs")
    h = model.hamiltonian.to_mv()
    state = np.zeros(2**n, complex)
    a, b = (int(bits, 2) for bits in case["reference_bits"])
    state[a], state[b] = math.cos(case["theta"]), np.exp(1j*case["phi"])*math.sin(case["theta"])
    generators = [("I", MV.scalar(n, 1))]
    source = case["pair_source"]
    for target in case["pair_targets"]:
        t = (cdag_op(n, 2*target)*cdag_op(n, 2*target+1)
             * c_op(n, 2*source+1)*c_op(n, 2*source))
        g = t-t.dagger()
        generators.append((f"(I+G{source}{target})/sqrt2", (MV.scalar(n, 1)+g)*(1/math.sqrt(2))))
    return h, generators, state


def sector_words(case):
    if case["kind"] == "tfim":
        return list(range(2**case["n"]))
    n, electrons = case["n"], case["n_electrons"]
    alpha = (electrons+int(round(2*case["sz"])) )//2
    beta = electrons-alpha
    return sorted(sum(1 << (n-1-j) for j in [*(2*p for p in up), *(2*p+1 for p in down)])
                  for up in itertools.combinations(range(n//2), alpha)
                  for down in itertools.combinations(range(n//2), beta))


def assessment_data(case, h, generators, state, *, include_spectrum=True):
    """Independent small-space oracle, invoked only after inference in a run."""
    words = sector_words(case)
    if case["kind"] == "tfim":
        dense_h = -case["J"]*sum((dense_word(case["n"], (3 << (2*q)) | (3 << (2*(q+1))))
                                     for q in range(case["n"]-1)), np.zeros((len(state), len(state)), complex))
        dense_h -= case["h"]*sum((dense_word(case["n"], 1 << (2*q)) for q in range(case["n"])),
                                 np.zeros_like(dense_h))
        operators = [dense_operator(g) for _, g in generators]
        vector = state
        ground = float(np.linalg.eigvalsh(dense_h)[0]) if include_spectrum else None
        spin_defect = None
    else:
        from benchmarks.phase17_ceo_oracle import hamiltonian, _fermionic_action
        from benchmarks.check_orthogonal_active_space import independent_spin_squared
        dense_h, _ = hamiltonian(ROOT/case["fcidump"], words)
        spin = independent_spin_squared(words, case["n"])
        spin_values, spin_vectors = np.linalg.eigh(spin)
        wanted = case["target_spin"]*(case["target_spin"]+1)
        frame = spin_vectors[:, abs(spin_values-wanted) <= 1e-10]
        if not frame.shape[1]:
            raise AssertionError("declared spin sector is empty")
        ground = float(np.linalg.eigvalsh(frame.conj().T@dense_h@frame)[0]) if include_spectrum else None
        vector = state[words]
        index = {word: i for i, word in enumerate(words)}
        operators = [np.eye(len(words), dtype=complex)]
        source, n = case["pair_source"], case["n"]
        for target in case["pair_targets"]:
            transfer = np.zeros_like(dense_h)
            sequence = [(2*target, True), (2*target+1, True), (2*source+1, False), (2*source, False)]
            for column, word in enumerate(words):
                acted = _fermionic_action(word, sequence, n)
                if acted is not None:
                    dest, sign = acted
                    transfer[index[dest], column] = sign
            operators.append((np.eye(len(words))+transfer-transfer.conj().T)/math.sqrt(2))
        spin_defect = float(np.linalg.norm((spin-wanted*np.eye(len(words)))@vector))
        if spin_defect > 1e-10:
            raise AssertionError("declared molecular preparation is spin contaminated")
    basis = np.column_stack([op@vector for op in operators])
    if case["kind"] == "molecule" and np.linalg.norm((spin-wanted*np.eye(len(words)))@basis) > 1e-10:
        raise AssertionError("molecular correction span escaped the target spin")
    s, hh = basis.conj().T@basis, basis.conj().T@dense_h@basis
    return {"overlap": s, "hamiltonian": hh, "ground_energy": ground,
            "sector_words": words, "reference_spin_defect": spin_defect}


def device_seconds(settings, counts, config, case):
    card = config["device_scenario_us"]
    prep = preparation_resources(case)
    preparation = (case["n"]*card["preparation_per_qubit"]
                   + prep["one_qubit_depth"]*card["one_qubit_layer"]
                   + prep["two_qubit_depth"]*card["two_qubit_layer"])
    total = 0.
    for setting, count in zip(settings, counts):
        _, _, d1, d2 = native_stats(setting.clifford)
        total += count*(preparation+card["readout_reset"]
                        + d1*card["one_qubit_layer"]+d2*card["two_qubit_layer"])
    return total*1e-6


def setting_metadata(settings):
    result = []
    for s in settings:
        _, _, d1, d2 = native_stats(s.clifford)
        result.append({"assigned_word_codes": list(s.assigned_word_codes),
                       "readouts": {str(code): [int(sign), list(positions)]
                                    for code, (sign, positions) in s.readouts.items()},
                       "gates": [[name, list(qubits)] for name, qubits in s.clifford.gates],
                       "d_1q": d1, "d_2q": d2})
    return result


def setup(case, config):
    admitted = array_admission_bound(case["n"])
    if admitted > config["owned_array_cap_bytes"]:
        raise MemoryError("owned-array admission failed before input allocation")
    if peak_rss_bytes() > config["process_rss_cap_bytes"]:
        raise MemoryError("fresh case worker already exceeds RSS ceiling")
    started = time.perf_counter()
    h, generators, state = build_case(case)
    simulator = JointShotSimulator(state, case["n"])
    simulator.shot_ledger = []
    simulator.raw_draw_seconds = 0.
    def draw(settings, counts, seed, identity):
        entry = {"seed": seed, "attempted_shots": sum(counts), "spent_shots": None,
                 "device_seconds": None, "simulator_seconds": 0.}
        simulator.shot_ledger.append(entry)
        start = time.perf_counter()
        try:
            batch = simulator.draw(settings, counts, seed, identity)
            entry["spent_shots"] = batch.hardware_shots
            actual_counts = tuple(sum(g.shots for g in batch.groups if g.setting_key == setting.key)
                                  for setting in settings)
            if sum(actual_counts) == batch.hardware_shots:
                entry["device_seconds"] = device_seconds(settings, actual_counts, config, case)
            return batch
        finally:
            entry["simulator_seconds"] = time.perf_counter()-start
            simulator.raw_draw_seconds += entry["simulator_seconds"]
    reference = PreparedSampledReference(case["n"], preparation_identity(case), draw)
    provider = SampledProjectedProvider(reference, h, generators)
    elapsed = time.perf_counter()-started
    if peak_rss_bytes() > config["process_rss_cap_bytes"]:
        raise MemoryError("case setup exceeds RSS ceiling")
    return h, generators, state, reference, provider, elapsed, admitted, simulator


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw*1024)


def _median(rows, key, resolved=False):
    values = [r[key] for r in rows if r.get(key) is not None and (not resolved or r["status"] == "resolved")]
    return statistics.median(values) if values else None


def summarize(rows, config):
    out = []
    family = len(config["cases"])*len(config["total_shot_budgets"])*2
    for case in config["cases"]:
        for budget in config["total_shot_budgets"]:
            for arm in ("ratio", "bank"):
                group = [r for r in rows if (r["case"], r["budget"], r["arm"]) == (case["name"], budget, arm)]
                if len(group) != config["replicas"]:
                    raise AssertionError("missing attempts in summary denominator")
                successes = sum(r["status"] == "resolved" and r.get("estimation_error") is not None
                                and r["estimation_error"] <= case["target_estimation_abs_error"] for r in group)
                out.append({"case": case["name"], "role": case["role"], "kind": case["kind"],
                            "unit": case["unit"], "target_estimation_abs_error": case["target_estimation_abs_error"],
                            "budget": budget, "arm": arm, "attempts": len(group), "successes": successes,
                            "success_probability": successes/len(group),
                            "success_probability_lower": exact_binomial_lower(successes, len(group),
                                                       config["success_probability_alpha"]/family),
                            "pilot_failures": sum(r["status"] == "pilot_failed" for r in group),
                            "denominator_failures": sum(r["status"] == "denominator_unresolved" for r in group),
                            "resource_failures": sum(r["status"] == "resource_failed" for r in group),
                            "worker_failures": sum(r["status"] == "worker_failed" for r in group),
                            "incomplete_costs": sum(not r.get("costs_complete", True) for r in group),
                            "other_failures": sum(r["status"] in {"evaluation_failed", "assessment_failed"} for r in group),
                            "resolved": sum(r["status"] == "resolved" for r in group),
                            "resolved_interval_misses": sum(r["status"] == "resolved" and r.get("interval_contains_truth") is False for r in group),
                            "median_end_to_end_seconds": _median(group, "end_to_end_seconds"),
                            "median_device_seconds": _median(group, "device_seconds"),
                            "median_classical_seconds": _median(group, "classical_seconds"),
                            "median_total_shots": _median(group, "total_shots"),
                            "median_resolved_estimation_error": _median(group, "estimation_error", True),
                            "median_resolved_retained_space_error": _median(group, "retained_space_error", True),
                            "median_resolved_ground_error": _median(group, "ground_error", True)})
    return out


def verdict(summaries, config):
    comparisons = []
    for case in config["cases"]:
        eligible = {}
        for arm in ("ratio", "bank"):
            candidates = [r for r in summaries if r["case"] == case["name"] and r["arm"] == arm
                          and r["success_probability_lower"] >= config["target_success_probability"]
                          and r["resource_failures"] == 0 and r["worker_failures"] == 0
                          and r["incomplete_costs"] == 0]
            eligible[arm] = min(candidates, key=lambda r: r["median_end_to_end_seconds"]) if candidates else None
        a, b = eligible["ratio"], eligible["bank"]
        reduction = 1-a["median_end_to_end_seconds"]/b["median_end_to_end_seconds"] if a and b else None
        comparisons.append({"case": case["name"], "unit": case["unit"],
                            "target_estimation_abs_error": case["target_estimation_abs_error"],
                            "ratio_budget": a["budget"] if a else None,
                            "bank_budget": b["budget"] if b else None,
                            "cost_reduction": reduction, "passes": reduction is not None and reduction >= .2})
    return {"status": "promote_on_declared_scope" if all(r["passes"] for r in comparisons)
            else "closed_on_declared_scope", "comparisons": comparisons,
            "quantum_advantage_claim": False, "hardware_claim": False}


def draw_accounting(entries):
    known = sum(e["spent_shots"] for e in entries if e["spent_shots"] is not None)
    unknown = any(e["spent_shots"] is None for e in entries)
    complete = not unknown and all(e["device_seconds"] is not None for e in entries)
    return {"known_shots": known, "attempted_shots": sum(e["attempted_shots"] for e in entries),
            "physical_shots_unknown": unknown, "costs_complete": complete,
            "device_seconds": sum(e["device_seconds"] for e in entries) if complete else None,
            "simulator_seconds": sum(e["simulator_seconds"] for e in entries)}


def evaluation_group_moments(plan, observations):
    out = []
    groups = {g.setting_key: g for g in observations.batch.groups}
    for setting, count in zip(plan.settings, plan.counts):
        group = groups[setting.key]
        totals = [0.]*5
        for bits, multiplicity in group.hist.items():
            values = []
            for functional in (plan.numerator, plan.denominator):
                value = 0.
                for code in setting.assigned_word_codes:
                    coefficient = functional.coefficients.get(code, 0.)
                    if coefficient:
                        sign, positions = setting.readouts[code]
                        value += coefficient*sign*(-1)**sum(bits[p] == "1" for p in positions)
                values.append(value)
            h, ss = values
            for index, value in enumerate((h, ss, h*h, ss*ss, h*ss)):
                totals[index] += multiplicity*value
        out.append(dict(zip(("sum_h", "sum_s", "sum_h2", "sum_s2", "sum_hs"), totals), shots=count))
    return out


def _base_row(case, budget, replica, arm, config, case_index):
    seed = config["master_seed"]+100000*case_index+1000*config["total_shot_budgets"].index(budget)+3*replica
    return {"case": case["name"], "kind": case["kind"], "unit": case["unit"],
            "target_estimation_abs_error": case["target_estimation_abs_error"],
            "budget": budget, "replica": replica, "arm": arm, "pilot_seed": seed,
            "evaluation_seed": seed+(1 if arm == "ratio" else 2), "status": "pilot_failed",
            "pilot_shots": 0, "evaluation_shots": 0, "total_shots": 0,
            "estimate": None, "estimation_error": None, "retained_space_error": None,
            "ground_error": None, "true_direction_energy": None, "retained_space_energy": None,
            "ground_energy": None, "device_seconds": 0., "classical_seconds": 0.,
            "setup_seconds": 0., "simulator_seconds": 0., "assessment_seconds": 0.,
            "admitted_array_bytes": array_admission_bound(case["n"]), "worker_pid": os.getpid(),
            "end_to_end_seconds": 0., "physical_shots_unknown": False,
            "attempted_physical_shots": 0, "costs_complete": True}


def _fill_failures(case, config, case_index, rows, reason, status="worker_failed"):
    existing = {(r["budget"], r["replica"], r["arm"]) for r in rows}
    for budget in config["total_shot_budgets"]:
        for replica in range(config["replicas"]):
            for arm in ("ratio", "bank"):
                if (budget, replica, arm) not in existing:
                    row = _base_row(case, budget, replica, arm, config, case_index)
                    row.update(status=status, failure_reason=reason)
                    rows.append(row)
    return sorted(rows, key=lambda r: (config["total_shot_budgets"].index(r["budget"]), r["replica"], r["arm"] == "bank"))


def load_record(path):
    path = Path(path)
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    if path.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _input_metadata(case, provider, admitted, setup_seconds):
    return {"case": case["name"], "kind": case["kind"], "unit": case["unit"], "n": case["n"],
            "target_estimation_abs_error": case["target_estimation_abs_error"],
            "reference_identity": provider.reference_identity,
            "hamiltonian_identity": provider.hamiltonian_identity,
            "provider_cache_identity": provider.cache_identity,
            "settings": setting_metadata(provider.plan.settings), "words": len(provider.plan.codes),
            "coefficient_occurrences": provider.bank.resources()["coefficient_occurrences"],
            "admitted_array_bytes": admitted, "setup_seconds": setup_seconds,
            "preparation": preparation_resources(case), "sector_words": sector_words(case)}


def _assess(row, frozen, result, data):
    from scipy.linalg import eigvalsh
    c, frame = frozen.coefficients, frozen.retained_frame
    s, h = data["overlap"], data["hamiltonian"]
    denominator = float((c.conj()@s@c).real)
    if not denominator > 0:
        raise AssertionError("exact frozen direction has nonpositive norm")
    truth = float((c.conj()@h@c).real/denominator)
    reduced_s, reduced_h = frame.conj().T@s@frame, frame.conj().T@h@frame
    retained = float(eigvalsh((reduced_h+reduced_h.conj().T)/2,
                             (reduced_s+reduced_s.conj().T)/2)[0])
    ground = data["ground_energy"]
    row.update(true_direction_energy=truth, retained_space_energy=retained, ground_energy=ground,
               estimation_error=abs(result.estimate-truth) if result.estimate is not None else None,
               retained_space_error=abs(result.estimate-retained) if result.estimate is not None else None,
               ground_error=abs(result.estimate-ground) if result.estimate is not None else None,
               interval_contains_truth=(result.interval[0] <= truth <= result.interval[1])
               if result.interval is not None else None)


def run_case(config, case_index, *, checkpoint=None):
    """Execute one case; unit tests use only independently declared toy configs."""
    case = config["cases"][case_index]
    rows = []
    setup_start = time.perf_counter()
    try:
        h, gens, state, reference, provider, setup_s, admitted, simulator = setup(case, config)
    except MemoryError as error:
        elapsed = time.perf_counter()-setup_start
        rows = _fill_failures(case, config, case_index, [], str(error), "resource_failed")
        for row in rows:
            row.update(setup_seconds=elapsed, classical_seconds=elapsed, end_to_end_seconds=elapsed,
                       worker_pid=os.getpid())
        output = {"input": {"case": case["name"], "kind": case["kind"], "status": "resource_failed",
                            "failure_reason": str(error)}, "rows": rows,
                  "worker_pid": os.getpid(), "status": "complete"}
        if checkpoint:
            atomic_json(checkpoint, output)
        return output
    metadata = _input_metadata(case, provider, admitted, setup_s)
    assessed = None
    for budget in config["total_shot_budgets"]:
        pilot_budget = int(budget*config["pilot_fraction"])
        for replica in range(config["replicas"]):
            base = config["master_seed"]+100000*case_index+1000*config["total_shot_budgets"].index(budget)+3*replica
            ledger_start = len(simulator.shot_ledger)
            started, before_draw = time.perf_counter(), simulator.raw_draw_seconds
            try:
                pilot_counts = uniform_setting_plan(provider.plan.settings, pilot_budget)
                pilot = reference.sample(provider.plan.settings, pilot_counts, seed=base)
                pilot_elapsed = time.perf_counter()-started
                pilot_draw = simulator.raw_draw_seconds-before_draw
                pilot_validation = max(0., pilot_elapsed-pilot_draw)
                pilot_device = device_seconds(provider.plan.settings, pilot_counts, config, case)
            except (ValueError, np.linalg.LinAlgError, MemoryError) as error:
                elapsed = time.perf_counter()-started
                spent = draw_accounting(simulator.shot_ledger[ledger_start:])
                classical = setup_s+max(0., elapsed-spent["simulator_seconds"])
                for arm in ("ratio", "bank"):
                    row = _base_row(case, budget, replica, arm, config, case_index)
                    row.update(status="resource_failed" if isinstance(error, MemoryError) else "pilot_failed",
                               failure_reason=str(error), setup_seconds=setup_s, classical_seconds=classical,
                               pilot_shots=spent["known_shots"], total_shots=spent["known_shots"],
                               attempted_physical_shots=spent["attempted_shots"],
                               physical_shots_unknown=spent["physical_shots_unknown"],
                               costs_complete=spent["costs_complete"], device_seconds=spent["device_seconds"],
                               simulator_seconds=spent["simulator_seconds"],
                               end_to_end_seconds=(None if spent["device_seconds"] is None
                                                   else classical+spent["device_seconds"]))
                    rows.append(row)
                if checkpoint and (replica+1) % 8 == 0:
                    atomic_json(checkpoint, {"input": metadata, "rows": rows, "status": "running", "worker_pid": os.getpid()})
                continue
            outcomes = {}
            for arm in (("ratio", "bank") if replica % 2 == 0 else ("bank", "ratio")):
                row = _base_row(case, budget, replica, arm, config, case_index)
                row.update(pilot_shots=pilot_budget, total_shots=pilot_budget,
                           pilot_identity=pilot.identity, pilot_counts=list(pilot_counts),
                           setup_seconds=setup_s, device_seconds=pilot_device,
                           attempted_physical_shots=pilot_budget,
                           classical_seconds=setup_s+pilot_validation, simulator_seconds=pilot_draw)
                started = time.perf_counter()
                try:
                    frozen = provider.freeze_ratio(pilot, budget-pilot_budget,
                                                   tau_s=config["tau_s"], max_condition=config["max_condition"],
                                                   delta=config["ratio_delta"], denominator_floor=config["denominator_floor"],
                                                   allocation=arm)
                    reconstructed_pilot = provider.reconstruct(pilot)
                    row["classical_seconds"] += time.perf_counter()-started
                    row.update(coefficients=encode_complex(frozen.coefficients),
                               retained_frame=encode_complex(frozen.retained_frame),
                               pilot_overlap=encode_complex(reconstructed_pilot.problem.overlap),
                               pilot_hamiltonian=encode_complex(reconstructed_pilot.problem.hamiltonian),
                               pilot_energy=frozen.pilot_energy, retained_rank=frozen.retained_frame.shape[1],
                               evaluation_counts=list(frozen.counts),
                               evaluation_settings=[i for i, setting in enumerate(provider.plan.settings)
                                                    if setting.key in {s.key for s in frozen.settings}],
                               frozen_plan_identity=frozen.identity)
                except (ValueError, np.linalg.LinAlgError, MemoryError) as error:
                    row["classical_seconds"] += time.perf_counter()-started
                    row.update(status="resource_failed" if isinstance(error, MemoryError) else "pilot_failed",
                               failure_reason=str(error), end_to_end_seconds=row["device_seconds"]+row["classical_seconds"])
                    outcomes[arm] = row
                    continue
                evaluation_ledger_start = len(simulator.shot_ledger)
                evaluation_start = time.perf_counter()
                try:
                    observed = reference.sample(frozen.settings, frozen.counts, seed=row["evaluation_seed"])
                    reconstruction = provider.reconstruct(observed) if arm == "bank" else None
                    result = evaluate_ratio(frozen, observed)
                    if reconstruction is not None and result.raw_ratio is not None:
                        c = frozen.coefficients
                        ss, hh = reconstruction.problem.overlap, reconstruction.problem.hamiltonian
                        rebuilt = float((c.conj()@hh@c).real/(c.conj()@ss@c).real)
                        if not math.isclose(rebuilt, result.raw_ratio, rel_tol=1e-10, abs_tol=1e-10):
                            raise AssertionError("bank contraction differs from frozen functional ratio")
                    if not all(math.isfinite(v) for v in (result.numerator, result.denominator)):
                        raise ValueError("nonfinite ratio inference")
                    row.update(asdict(result))
                    row["evaluation_group_moments"] = evaluation_group_moments(frozen, observed)
                    row["final_functional_words"] = len(set(frozen.numerator.coefficients) | set(frozen.denominator.coefficients))
                    row["final_functional_bytes"] = sum(sys.getsizeof(dict(f.coefficients))+
                        sum(sys.getsizeof(k)+sys.getsizeof(v) for k, v in f.coefficients.items())
                        for f in (frozen.numerator, frozen.denominator))
                except (ValueError, np.linalg.LinAlgError, MemoryError, AssertionError) as error:
                    row.update(status="resource_failed" if isinstance(error, MemoryError) else "evaluation_failed",
                               failure_reason=str(error), estimate=None)
                    result = None
                elapsed = time.perf_counter()-evaluation_start
                spent = draw_accounting(simulator.shot_ledger[evaluation_ledger_start:])
                row["evaluation_shots"] = spent["known_shots"]
                row["total_shots"] = pilot_budget+spent["known_shots"]
                row["attempted_physical_shots"] = pilot_budget+spent["attempted_shots"]
                row["physical_shots_unknown"] = spent["physical_shots_unknown"]
                row["costs_complete"] = spent["costs_complete"]
                row["simulator_seconds"] += spent["simulator_seconds"]
                row["classical_seconds"] += max(0., elapsed-spent["simulator_seconds"])
                row["device_seconds"] = (None if spent["device_seconds"] is None
                                          else pilot_device+spent["device_seconds"])
                row["end_to_end_seconds"] = (None if row["device_seconds"] is None
                                               else row["device_seconds"]+row["classical_seconds"])
                if result is None:
                    outcomes[arm] = row
                    continue
                started = time.perf_counter()
                try:
                    if assessed is None:
                        assessed = assessment_data(case, h, gens, state)
                    _assess(row, frozen, result, assessed)
                except (ValueError, np.linalg.LinAlgError, AssertionError) as error:
                    row.update(status="assessment_failed", failure_reason=str(error), estimate=None)
                row["assessment_seconds"] = time.perf_counter()-started
                row["process_peak_rss_bytes"] = peak_rss_bytes()
                if row["process_peak_rss_bytes"] > config["process_rss_cap_bytes"]:
                    row.update(status="resource_failed", estimate=None, failure_reason="process RSS ceiling breached")
                row["end_to_end_seconds"] = row["device_seconds"]+row["classical_seconds"]
                outcomes[arm] = row
            rows.extend(outcomes[arm] for arm in ("ratio", "bank"))
            if checkpoint and (replica+1) % 8 == 0:
                atomic_json(checkpoint, {"input": metadata, "rows": rows, "status": "running", "worker_pid": os.getpid()})
        print(f'{case["name"]}: {budget} shots, {config["replicas"]} replicas per arm', file=sys.stderr, flush=True)
    output = {"input": metadata, "rows": rows, "status": "complete", "worker_pid": os.getpid()}
    if checkpoint:
        atomic_json(checkpoint, output)
    return output


def _committed_preflight(config):
    sources = scientific_sources(config)
    inputs = [c["fcidump"] for c in config["cases"] if c["kind"] == "molecule"]
    paths = (DECLARATION, *sources, *inputs)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    for path in paths:
        git("cat-file", "-e", f"HEAD:{path}")
    return git("log", "-1", "--format=%H", "--", DECLARATION), sources


def _common_git_directory():
    path = Path(git("rev-parse", "--git-common-dir"))
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def claim_path():
    return _common_git_directory()/f'cross_domain_ratio_{digest(DECLARATION)}.claim.json'


def _run_workers(config, execution_commit, declared_commit, output, *, replay=False, authorization=None, replay_receipt=None):
    receipt_dir = _common_git_directory()/f'cross_domain_ratio_{digest(DECLARATION)[:16]}_execution'
    if replay:
        receipt_dir = Path(tempfile.mkdtemp(prefix="cross-domain-replay-", dir=_common_git_directory()))
    receipt_dir.mkdir(parents=True, exist_ok=True)
    inputs, rows, receipts = [], [], []
    for index, case in enumerate(config["cases"]):
        checkpoint = receipt_dir/f'{index}_{case["name"]}.json'
        command = [sys.executable, str(ROOT/"benchmarks/run_cross_domain_ratio.py"), "--worker", str(index),
                   "--worker-output", str(checkpoint), "--execution-commit", execution_commit,
                   "--declaration-commit", declared_commit, "--authorization", authorization]
        if replay:
            command.extend(["--worker-replay", "--replay-receipt", str(replay_receipt)])
        started = time.perf_counter()
        proc = subprocess.Popen(command, cwd=ROOT)
        timed_out = False
        try:
            proc.wait(timeout=config["case_timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
        if checkpoint.exists():
            worker = json.loads(checkpoint.read_text())
        else:
            worker = {"input": {"case": case["name"], "kind": case["kind"], "status": "worker_failed"}, "rows": []}
        completed = worker.get("status") == "complete" and proc.returncode == 0
        observed_outcomes = len(worker["rows"])
        worker["rows"] = _fill_failures(case, config, index, worker["rows"],
                                      f'case worker unavailable/incomplete, exit={proc.returncode}')
        for row in worker["rows"]:
            if row["status"] == "worker_failed":
                row.update(worker_pid=proc.pid, physical_shots_unknown=True, costs_complete=False,
                           attempted_physical_shots=None, device_seconds=None, end_to_end_seconds=None)
        inputs.append(worker["input"])
        rows.extend(worker["rows"])
        receipts.append({"case": case["name"], "returncode": proc.returncode, "completed": completed,
                         "retained_observations": observed_outcomes, "observed_outcomes": observed_outcomes,
                         "scheduled_outcomes": len(worker["rows"]),
                         "imputed_failures": len(worker["rows"])-observed_outcomes,
                         "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.exists() else None,
                         "worker_seconds": time.perf_counter()-started, "worker_pid": proc.pid,
                         "execution_commit": execution_commit, "declaration_commit": declared_commit,
                         "declaration_sha256": digest(DECLARATION), "timed_out": timed_out,
                         "case_timeout_seconds": config["case_timeout_seconds"],
                         "source_manifest_sha256": hashlib.sha256(json.dumps(
                             {p: digest(p) for p in scientific_sources(config)}, sort_keys=True).encode()).hexdigest()})
        if not replay:
            atomic_json(claim_path(), {"status": "running", "declaration_sha256": digest(DECLARATION),
                                      "execution_commit": execution_commit, "declaration_commit": declared_commit,
                                      "completed_cases": index+1, "worker_receipts": receipts,
                                      "claim_nonce": authorization})
    return {"schema": "clifford_qc.cross_domain_ratio_results.v1",
            "declaration_sha256": digest(DECLARATION), "declaration_commit": declared_commit,
            "execution_commit": execution_commit, "source_sha256": {p: digest(p) for p in scientific_sources(config)},
            "input_sha256": {c["fcidump"]: digest(c["fcidump"]) for c in config["cases"] if c["kind"] == "molecule"},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": __import__("scipy").__version__,
                            "threads": {key: os.environ.get(key) for key in
                                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
            "inputs": inputs, "rows": rows, "summaries": summarize(rows, config),
            "verdict": verdict(summarize(rows, config), config), "worker_receipts": receipts}


def compare_records(expected, actual):
    advisory = {"environment", "source_sha256", "execution_commit", "process_peak_rss_bytes",
                "checkpoint_sha256", "verdict", "worker_pid"}
    def compare(a, b, path="record"):
        if isinstance(a, dict):
            ka = {k for k in a if k not in advisory and not k.endswith("seconds")}
            kb = {k for k in b if k not in advisory and not k.endswith("seconds")}
            if ka != kb:
                raise AssertionError(f"{path}: record keys differ")
            for key in ka:
                compare(a[key], b[key], path+"."+key)
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise AssertionError(f"{path}: length differs")
            for i, (left, right) in enumerate(zip(a, b)):
                compare(left, right, f"{path}[{i}]")
        elif isinstance(a, float):
            if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
                raise AssertionError(f"{path}: numeric replay differs")
        elif a != b:
            raise AssertionError(f"{path}: scientific replay differs")
    compare(expected, actual)


def validate_execution_environment(config):
    import scipy
    expected = config["execution_environment"]
    observed = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    for key in ("python", "numpy", "scipy"):
        if observed[key] != expected[key]:
            raise ValueError(f"frozen execution environment mismatch for {key}: {observed[key]}")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError(f"frozen execution requires {key}=1")


def worker_authorized(args, execution_commit, declared_commit):
    if not args.authorization:
        raise ValueError("worker requires an authorized parent nonce")
    if args.worker_replay:
        if args.replay_receipt is None:
            raise ValueError("replay worker requires an existing-record receipt")
        receipt_path = args.replay_receipt.resolve()
        if receipt_path.parent != _common_git_directory():
            raise ValueError("replay authorization is outside common Git directory")
        receipt = json.loads(receipt_path.read_text())
        record = Path(receipt["record_path"])
        if (receipt.get("mode") != "replay" or not record.is_file()
                or hashlib.sha256(record.read_bytes()).hexdigest() != receipt.get("record_sha256")):
            raise ValueError("replay requires the unchanged existing result")
        observed = load_record(record)
        if (observed.get("schema") != "clifford_qc.cross_domain_ratio_results.v1"
                or observed.get("declaration_sha256") != digest(DECLARATION)):
            raise ValueError("replay result does not belong to this declaration")
    else:
        receipt = json.loads(claim_path().read_text())
        if receipt.get("status") != "running":
            raise ValueError("worker has no authorized running claim")
    if (receipt.get("claim_nonce") != args.authorization
            or receipt.get("execution_commit") != execution_commit
            or receipt.get("declaration_commit") != declared_commit
            or receipt.get("declaration_sha256") != digest(DECLARATION)):
        raise ValueError("worker parent identities or nonce differ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check", type=Path)
    mode.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-replay", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--authorization", help=argparse.SUPPRESS)
    parser.add_argument("--replay-receipt", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--execution-commit", help=argparse.SUPPRESS)
    parser.add_argument("--declaration-commit", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=ROOT/RESULT)
    args = parser.parse_args(argv)
    config = load_declaration()
    if args.validate:
        for case in config["cases"]:
            h, gens, state, _, provider, _, admitted, _ = setup(case, config)
            prepared = preparation_state_from_circuit(case)
            overlap = np.vdot(state, prepared)
            if abs(overlap) < 1e-12 or not np.allclose(prepared, state*overlap/abs(overlap), atol=1e-12):
                raise AssertionError("preparation circuit differs from declared state")
            checked = assessment_data(case, h, gens, state, include_spectrum=False)
            native = np.column_stack([dense_operator(g)@state for _, g in gens])
            s = native.conj().T@native
            hh = native.conj().T@dense_operator(h)@native
            if not (np.allclose(s, checked["overlap"], atol=1e-10, rtol=0)
                    and np.allclose(hh, checked["hamiltonian"], atol=1e-10, rtol=0)):
                raise AssertionError("independent structural projection disagrees")
            print(f'{case["name"]}: {len(provider.plan.settings)} settings; {admitted} admitted bytes; no shots drawn')
        return
    if args.execute or args.check or args.worker is not None:
        validate_execution_environment(config)
    declared_commit, sources = _committed_preflight(config)
    execution_commit = git("rev-parse", "HEAD")
    if args.worker is not None:
        if args.worker_output is None or not 0 <= args.worker < len(config["cases"]):
            raise ValueError("invalid private worker arguments")
        if args.execution_commit != execution_commit or args.declaration_commit != declared_commit:
            raise ValueError("worker execution identities differ")
        worker_authorized(args, execution_commit, declared_commit)
        if args.worker_output.exists():
            raise FileExistsError("worker checkpoint already exists")
        run_case(config, args.worker, checkpoint=args.worker_output)
        return
    if args.check:
        expected = load_record(args.check)
        if (expected.get("schema") != "clifford_qc.cross_domain_ratio_results.v1"
                or expected.get("declaration_sha256") != digest(DECLARATION)):
            raise ValueError("existing record does not match the frozen declaration")
        nonce = uuid.uuid4().hex
        receipt_path = _common_git_directory()/f"cross_domain_replay_{nonce}.json"
        atomic_json(receipt_path, {"mode": "replay", "claim_nonce": nonce,
                                  "execution_commit": execution_commit, "declaration_commit": declared_commit,
                                  "declaration_sha256": digest(DECLARATION),
                                  "record_path": str(args.check.resolve()),
                                  "record_sha256": hashlib.sha256(args.check.read_bytes()).hexdigest()})
        try:
            actual = _run_workers(config, execution_commit, declared_commit, args.check, replay=True,
                                  authorization=nonce, replay_receipt=receipt_path)
            compare_records(expected, actual)
        finally:
            receipt_path.unlink()
        print("cross-domain full seed replay passed")
        return
    if args.output.exists():
        raise FileExistsError("observation record exists; use --check")
    claim = {"status": "running", "declaration_sha256": digest(DECLARATION),
             "execution_commit": execution_commit, "declaration_commit": declared_commit,
             "claim_nonce": uuid.uuid4().hex, "completed_cases": 0}
    with claim_path().open("x") as stream:
        json.dump(claim, stream, sort_keys=True)
    try:
        result = _run_workers(config, execution_commit, declared_commit, args.output, authorization=claim["claim_nonce"])
        if args.output.exists():
            raise FileExistsError("outcome path appeared during execution")
        atomic_json(args.output, result, exclusive=True)
        atomic_json(claim_path(), {**claim, "status": "completed", "result_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})
        print(json.dumps(result["verdict"], indent=2))
    except BaseException as error:
        atomic_json(claim_path(), {**claim, "status": "failed", "reason": str(error)})
        raise


if __name__ == "__main__":
    main()

"""Check Phase 2M observations independently; optional isolated pencil rebuild."""
from __future__ import annotations

import argparse
import base64
from dataclasses import fields
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks import bank_policy_contract as contract

SCIENCE_KEYS = {"manifest", "config", "labels", "indices", "energy", "energy_history",
                "growth", "scores", "solves", "stopped_reason", "logical", "evidence_tier", "resource_scope"}
LOGICAL_KEYS = {"pair_ledger", "pairs_built", "coefficient_occurrences", "word_universe",
                "selected_word_universe", "selected_word_sha256", "pair_ownership",
                "retained_block_pairs", "selection_pairs"}
COST_KEYS = {"preparation_seconds", "solve_seconds", "first_pair_build_seconds", "recompute_seconds",
             "postprocess_seconds", "baseline_rss_bytes", "solve_end_rss_bytes", "worker_high_water_rss_bytes",
             "worker_sampled_solve_peak_rss_bytes", "worker_rss_observations",
             "solve_end_delta_bytes_per_resident_coefficient", "physical_pair_builds", "recomputed_rows",
             "evicted_rows", "resident_rows", "resident_coefficients", "resident_word_universe",
             "packed_live_bytes", "packed_reserved_bytes", "coefficient_payload_estimate_bytes",
             "spill_bytes", "row_observations", "product_temporary_row_limit", "instrumentation_in_process_costs"}
PARENT_KEYS = {"pid", "launched_pid", "returncode", "wall_seconds", "sampled_peak_rss_bytes",
               "sampled_solve_peak_rss_bytes", "sampled_solve_delta_rss_bytes", "rss_ceiling_bytes",
               "timeout_seconds", "sample_seconds", "launch_method"}
WORKER_KEYS = {"pid", "parent_pid", "start_ticks", "python", "numpy", "scipy", "threads"}


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def decode_matrix(item, size):
    if set(item) != {"shape", "dtype", "data"} or item["dtype"] != "<c16":
        raise ValueError("matrix encoding differs")
    if item["shape"] != [size, size] or any(type(v) is not int for v in item["shape"]):
        raise ValueError("matrix dimensions differ")
    raw = base64.b64decode(item["data"], validate=True)
    if len(raw) != size * size * 16:
        raise ValueError("matrix payload length differs")
    value = np.frombuffer(raw, dtype="<c16").reshape(size, size)
    if not np.isfinite(value).all() or not np.array_equal(value, value.conj().T):
        raise ValueError("matrix is non-finite or non-Hermitian")
    return value


def hex_number(value):
    if type(value) is not str:
        raise ValueError("scientific float must use hexadecimal encoding")
    number = float.fromhex(value)
    if not math.isfinite(number) or number.hex() != value:
        raise ValueError("scientific float is non-finite or non-canonical")
    return number


def cell_problems(cell, system, arm, config):
    problems = []
    if cell["system"] != system["id"] or cell["arm"] != list(arm):
        return ["cell identity/order differs"]
    if cell["status"] != "completed":
        if cell["status"] not in {"memory_limit", "timeout", "worker_failed"}:
            problems.append("unknown failure status")
        return problems
    if set(cell) != {"system", "arm", "status", "scientific", "costs", "worker", "parent"}:
        problems.append("completed cell key set differs")
    science, costs, worker, parent = [cell[key] for key in ("scientific", "costs", "worker", "parent")]
    for name, value, keys in (("scientific", science, SCIENCE_KEYS), ("costs", costs, COST_KEYS),
                               ("worker", worker, WORKER_KEYS), ("parent", parent, PARENT_KEYS)):
        if set(value) != keys:
            problems.append(f"{name} key set differs")
    if science["evidence_tier"] != "exact" or science["resource_scope"] != "S/H_bank_only":
        problems.append("evidence/scope differs")
    if system.get("kind") != "toy" and science["manifest"] != system["manifest"]:
        problems.append("candidate manifest differs")
    expected_options = dict(config["options"], exact_ground_energy=system.get("exact_ground_energy"))
    if contract.canonical(science["config"]) != contract.canonical(expected_options):
        problems.append("solver options differ")
    labels, indices, solves = science["labels"], science["indices"], science["solves"]
    maximum = config["options"]["max_size"] + 1
    if not 1 <= len(indices) <= maximum or len(labels) != len(indices) or len(set(indices)) != len(indices):
        problems.append("basis size/indices differ")
    if indices[0] != 0 or labels[0] != "I":
        problems.append("initial identity differs")
    count = science["manifest"]["candidate_count"] + 1
    if any(not integer(i) or i >= count for i in indices):
        problems.append("basis index outside full pool")
    if len(solves) != len(indices):
        problems.append("missing projected solve")
    for size, pencil in enumerate(solves, 1):
        if set(pencil) != {"indices", "S", "H"} or pencil["indices"] != indices[:size]:
            problems.append("projected solve order differs")
        decode_matrix(pencil["S"], size)
        decode_matrix(pencil["H"], size)
    if len(science["energy_history"]) != len(solves) or len(science["growth"]) != len(indices) - 1:
        problems.append("trajectory length differs")
    energies = [hex_number(value) for value in science["energy_history"]]
    if science["energy"] != science["energy_history"][-1]:
        problems.append("final energy differs")
    if any(b > a + 1e-10 for a, b in zip(energies, energies[1:])):
        problems.append("non-monotone Ritz trajectory")
    for size, step in enumerate(science["growth"], 2):
        from clifford_qc.subspace.adaptive import GrowthRecord
        if set(step) != {f.name for f in fields(GrowthRecord)} - {"step_seconds"}:
            problems.append("growth diagnostic key set differs")
        if step["basis_size"] != size or step["selected_label"] != labels[size - 1]:
            problems.append("growth label/size differs")
        if step["energy"] != science["energy_history"][size - 1]:
            problems.append("growth energy differs")
    for event in science["scores"]:
        from clifford_qc.subspace.adaptive import CandidateScore
        if set(event) != {"basis", "candidate"} or set(event["candidate"]) != {f.name for f in fields(CandidateScore)}:
            problems.append("candidate diagnostic key set differs")
        candidate = event["candidate"]
        if not integer(candidate["index"], 1) or candidate["index"] >= count:
            problems.append("scored candidate outside full pool")
        for key in ("residual_coupling", "predicted_lowering", "orthogonal_fraction", "score"):
            hex_number(candidate[key])
        if event["basis"] != indices[:len(event["basis"])]:
            problems.append("scoring basis differs")
    logical = science["logical"]
    if set(logical) != LOGICAL_KEYS:
        problems.append("logical resource key set differs")
    for key in LOGICAL_KEYS - {"pair_ledger", "selected_word_sha256", "pair_ownership"}:
        if not integer(logical[key]):
            problems.append(f"logical count type differs: {key}")
    pairs = logical["pair_ledger"]
    seen = set()
    coefficient_count = 0
    for row in pairs:
        if len(row) != 4 or any(not integer(v) for v in row):
            raise ValueError("invalid pair ledger")
        i, j, overlap, element = row
        if i > j or j >= count or (i, j) in seen:
            problems.append("duplicate or invalid logical pair")
        seen.add((i, j))
        coefficient_count += overlap + element
    selected = set(indices)
    retained = sum(i in selected and j in selected for i, j in seen)
    if logical["pairs_built"] != len(pairs) or logical["coefficient_occurrences"] != coefficient_count:
        problems.append("cumulative logical count differs")
    if logical["retained_block_pairs"] != retained or retained != len(indices) * (len(indices) + 1) // 2:
        problems.append("retained logical pair ownership differs")
    if logical["selection_pairs"] != len(pairs) - retained:
        problems.append("selection logical pair ownership differs")
    ownership = logical["pair_ownership"]
    if len(ownership) != count or any(len(row) != 3 or not integer(row[1]) or not integer(row[2]) for row in ownership):
        problems.append("word ownership ledger differs")
    elif sum(row[1] + row[2] for row in ownership) != coefficient_count or sum(row[1] for row in ownership) != logical["word_universe"]:
        problems.append("word ownership accounting differs")
    if not integer(logical["selected_word_universe"]) or logical["selected_word_universe"] > logical["word_universe"]:
        problems.append("selected word count exceeds cumulative count")
    observations = costs["row_observations"]
    if not observations or any(len(row) != 2 or not integer(row[0]) for row in observations):
        raise ValueError("missing/invalid lifetime observations")
    if costs["resident_rows"] != observations[-1][0]:
        problems.append("final residency differs")
    if arm[2] == "stream_recompute":
        limits = {2 * (m * (m + 1) // 2 + config["execution"]["frontier_pairs"])
                  for m in range(1, maximum + 1)}
        if any(not integer(limit) or limit not in limits or rows > limit for rows, limit in observations):
            problems.append("streaming row bound failed")
    elif any(limit is not None for _, limit in observations) or costs["evicted_rows"] != 0 or costs["recomputed_rows"] != 0:
        problems.append("retain_all lifetime counters differ")
    if costs["recomputed_rows"] != 2 * (costs["physical_pair_builds"] - len(pairs)):
        problems.append("physical versus logical build count differs")
    if costs["coefficient_payload_estimate_bytes"] != 24 * costs["resident_coefficients"] or costs["spill_bytes"] != 0:
        problems.append("payload/spill accounting differs")
    if arm[1] == "object" and (costs["packed_live_bytes"] != 0 or costs["packed_reserved_bytes"] != 0):
        problems.append("object arm claims packed arrays")
    if costs["packed_reserved_bytes"] < costs["packed_live_bytes"]:
        problems.append("packed reservation below live arrays")
    for key, value in costs.items():
        if key.endswith("_seconds") and (not contract.finite_number(value) or value < 0):
            problems.append(f"invalid timing: {key}")
        if key.endswith("_bytes") and (not integer(value)):
            problems.append(f"invalid byte count: {key}")
    for key in ("physical_pair_builds", "recomputed_rows", "evicted_rows", "resident_rows",
                "resident_coefficients", "resident_word_universe", "product_temporary_row_limit"):
        if not integer(costs[key]):
            problems.append(f"invalid cost count: {key}")
    if not integer(costs["worker_rss_observations"], 1):
        problems.append("worker RSS sampler did not observe the solve")
    if costs["instrumentation_in_process_costs"] is not True or costs["product_temporary_row_limit"] != 2:
        problems.append("cost scope differs")
    numerator = max(0, costs["solve_end_rss_bytes"] - costs["baseline_rss_bytes"])
    expected_density = numerator / costs["resident_coefficients"] if costs["resident_coefficients"] else None
    if costs["solve_end_delta_bytes_per_resident_coefficient"] != expected_density:
        problems.append("process byte density differs")
    if worker["pid"] != parent["launched_pid"] or worker["parent_pid"] != parent["pid"] or worker["pid"] == parent["pid"]:
        problems.append("worker process identity differs")
    if not integer(worker["start_ticks"], 1) or worker["threads"] != {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}:
        problems.append("worker start/thread identity differs")
    for key in ("rss_ceiling_bytes", "timeout_seconds", "sample_seconds"):
        if parent[key] != config["execution"][key]:
            problems.append(f"supervision setting differs: {key}")
    if parent["returncode"] != 0 or parent["launch_method"] != "exec_fresh_python":
        problems.append("worker launch/completion differs")
    if type(parent["returncode"]) is not int or any(not integer(block[key], 1) for block, key in
        ((parent, "pid"), (parent, "launched_pid"), (worker, "pid"), (worker, "parent_pid"))):
        problems.append("process identity types differ")
    if max(parent["sampled_peak_rss_bytes"], costs["worker_high_water_rss_bytes"]) > parent["rss_ceiling_bytes"]:
        problems.append("completed worker exceeded RSS ceiling")
    if parent["wall_seconds"] > parent["timeout_seconds"] + 5:
        problems.append("completed worker exceeded timeout")
    return problems


def comparison_problems(record, config):
    problems = []
    cells = record["cells"]
    if len(cells) != 4 * len(config["systems"]):
        return ["matrix cell count differs"]
    identities = set()
    for offset, system in enumerate(config["systems"]):
        block = cells[offset * 4:offset * 4 + 4]
        for cell, arm in zip(block, contract.ARMS):
            try:
                problems += [f"{system['id']}/{arm[0]}: {message}" for message in
                             cell_problems(cell, system, arm, config)]
                if cell["status"] == "completed":
                    identity = (cell["worker"]["pid"], cell["worker"]["start_ticks"])
                    if identity in identities:
                        problems.append("worker process was reused")
                    identities.add(identity)
                    if cell["parent"]["pid"] != record["execution"]["parent_pid"]:
                        problems.append("parent execution identity differs")
            except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
                problems.append(f"{system['id']}/{arm[0]}: malformed cell: {exc}")
        if all(cell["status"] == "completed" for cell in block):
            baseline = contract.canonical(block[0]["scientific"])
            for cell in block[1:]:
                if contract.canonical(cell["scientific"]) != baseline:
                    problems.append(f"{system['id']}/{cell['arm'][0]}: scientific/logical equivalence failed")
    return problems


def environment_problems(record, reference):
    """Check the recorded interpreter and libraries, not the reader's environment."""
    problems = []
    stamp = record["provenance"]
    python = stamp["python"]
    # The record contract pins Python's minor release; patch releases may differ.
    if type(python) is not str or python.split(".")[:2] != reference["python"].split(".")[:2]:
        problems.append("execution environment differs: python")
    for key in ("numpy", "scipy"):
        if stamp["dependencies"][key] != reference["dependencies"][key]:
            problems.append(f"execution environment differs: {key}")
    for cell in record["cells"]:
        if cell["status"] == "completed":
            # Every fresh worker is launched through the parent's sys.executable.
            if cell["worker"]["python"] != python:
                problems.append("worker environment differs: python")
            for key in ("numpy", "scipy"):
                if cell["worker"][key] != stamp["dependencies"][key]:
                    problems.append(f"worker environment differs: {key}")
    return problems


def check_record(record, config, *, provenance=True):
    problems = []
    if set(record) != {"schema", "phase", "configuration_sha256", "claim_boundary", "cells", "verdict", "execution", "provenance"}:
        problems.append("record key set differs")
    if record["schema"] != contract.SCHEMA or record["phase"] != "2M-C/D" or record["claim_boundary"] != contract.CLAIM_BOUNDARY:
        problems.append("record identity/scope differs")
    if record["configuration_sha256"] != contract.EXPECTED_CONFIG_SHA256:
        problems.append("declaration binding differs")
    observations = comparison_problems(record, config)
    completed = all(cell["status"] == "completed" for cell in record["cells"])
    expected = "INCOMPLETE" if not completed else "INVALID" if observations else "VALID"
    if record["verdict"] != expected:
        problems.append("verdict differs")
    # INVALID/INCOMPLETE remain inspectable; they never pass a completion gate.
    problems += observations
    if expected != "VALID":
        problems.append(f"campaign is {expected}; Phase 2M completion is not established")
    if record["execution"]["serial_workers"] is not True:
        problems.append("serial worker declaration differs")
    if provenance:
        stamp = record["provenance"]
        execution = stamp["git_sha"]
        if stamp["git_dirty"] is not False or set(stamp["source_blobs"]) != set(contract.SOURCES):
            problems.append("source provenance differs")
        for path in contract.SOURCES:
            if stamp["source_blobs"][path] != contract.git("rev-parse", f"{execution}:{path}"):
                problems.append(f"execution source binding differs: {path}")
        config_path = str(contract.CONFIG.relative_to(ROOT))
        declaration = contract.git("log", "--diff-filter=A", "--format=%H", "--", config_path).splitlines()[-1]
        if stamp["declaration_commit"] != declaration:
            problems.append("declaration commit identity differs")
        contract.git("merge-base", "--is-ancestor", declaration, execution)
        path = str(contract.RECORD.relative_to(ROOT))
        commits = contract.git("log", "--diff-filter=A", "--format=%H", "--", path).splitlines()
        if not commits:
            problems.append("record is not committed")
        else:
            first = commits[-1]
            if first == execution or contract.git("diff-tree", "--no-commit-id", "--name-only", "-r", first) != path:
                problems.append("first record must be committed alone after execution")
            contract.git("merge-base", "--is-ancestor", execution, first)
        reference = contract.read_json(ROOT / config["environment_reference"])["provenance"]
        problems += environment_problems(record, reference)
    return problems


def reference_check(request):
    """Build final pencils with direct products, without any row-store backend."""
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.models.fcidump import fcidump_model
    from clifford_qc.subspace import identity_generator
    from clifford_qc.subspace.fermionic_generators import determinant_excitations, occupied_spin_orbitals
    from clifford_qc.subspace.linalg import solve_projected
    spec, science = request["system"], request["scientific"]
    if spec.get("kind") == "toy":
        from clifford_qc.models.spin import tfim
        from clifford_qc.algorithms.pools import local_pool, odd_y_filter
        from clifford_qc.subspace import pauli_orbit
        model = tfim(spec["n_qubits"])
        pool = pauli_orbit([op.word for op in odd_y_filter(local_pool(model.n))])
    else:
        model = fcidump_model(ROOT / spec["source"]["path"])
        pool = determinant_excitations(model.n, occupied_spin_orbitals(model), max_rank=2)
    generators = [identity_generator(model.n), *pool]
    selected = [generators[index] for index in science["indices"]]
    rho = ExactMVBackend().state(model.reference, ())
    size = len(selected)
    s, h = np.zeros((size, size), complex), np.zeros((size, size), complex)
    acted = [model.hamiltonian.to_mv() * generator.mv for generator in selected]
    words = set()
    scale = float(2**model.n)
    for j in range(size):
        for i in range(j + 1):
            # Canonical arithmetic follows the frozen generator registration
            # order, not adaptive basis position. Preserve signed zeros when
            # conjugating an entry whose selected coordinates reverse it.
            left, right = (i, j) if science["indices"][i] <= science["indices"][j] else (j, i)
            adjoint = selected[left].mv.dagger()
            overlap, element = adjoint * selected[right].mv, adjoint * acted[right]
            words.update(overlap.terms)
            words.update(element.terms)
            sv, hv = scale * overlap.trace_pairing(rho), scale * element.trace_pairing(rho)
            s[i, j], h[i, j] = (sv, hv) if left == i else (sv.conjugate(), hv.conjugate())
    for j in range(size):
        s[j, j], h[j, j] = s[j, j].real, h[j, j].real
        for i in range(j):
            s[j, i], h[j, i] = s[i, j].conjugate(), h[i, j].conjugate()
    pencil = science["solves"][-1]
    valid = contract.matrix_record(s) == pencil["S"] and contract.matrix_record(h) == pencil["H"]
    cfg = science["config"]
    result = solve_projected(s, h, [g.label for g in selected], tau_s=cfg["tau_s"],
                             rel_tau=cfg["rel_tau"], max_condition=cfg["max_condition"])
    return {"matrices_bitwise_equal": bool(valid),
            "energy_bitwise_equal": float(result.ground_energy).hex() == science["energy"],
            "labels_equal": [g.label for g in selected] == science["labels"],
            "selected_word_count_equal": len(words) == science["logical"]["selected_word_universe"],
            "selected_word_digest_equal": contract.digest(sorted(words)) == science["logical"]["selected_word_sha256"]}


def rebuild_reference(spec, science):
    with tempfile.TemporaryDirectory(prefix="phase2m-reference-") as directory:
        request, output = Path(directory) / "request.json", Path(directory) / "output.json"
        contract.atomic_json(request, {"system": spec, "scientific": science})
        import os
        env = dict(os.environ)
        env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reference-request", str(request),
                        "--reference-output", str(output)], cwd=ROOT, env=env, check=True, timeout=21600)
        return contract.read_json(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="also rebuild final pencils in isolated processes")
    parser.add_argument("--reference-request", help=argparse.SUPPRESS)
    parser.add_argument("--reference-output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.reference_request:
            contract.atomic_json(args.reference_output, reference_check(contract.read_json(args.reference_request)))
            return 0
        config = contract.load_config()
        if not contract.RECORD.exists():
            print("OK: declaration intact; molecular policy record absent; no outcomes evaluated")
            return 0
        record = contract.read_json(contract.RECORD)
        problems = check_record(record, config)
        if args.rebuild and not problems:
            for offset, spec in enumerate(config["systems"]):
                checked = rebuild_reference(spec, record["cells"][offset * 4]["scientific"])
                if not all(value is True for value in checked.values()):
                    problems.append(f"{spec['id']}: direct pencil rebuild differs")
        if problems:
            raise ValueError("; ".join(problems[:30]))
        print("OK: isolated policy equivalence, row bounds, cost accounting and commit provenance; VALID")
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as exc:
        print(f"Phase 2M policy record failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

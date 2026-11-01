"""Preregistered equal-shot selector; not an end-to-end speedup experiment."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.models import tfim
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.multivector import MV
from clifford_qc.subspace.linalg import solve_projected
from benchmarks.check_cross_domain_ratio import case_data  # assessment only

DECLARATION = ROOT / "docs/restart/noise_aware_selection_declaration.json"
INPUTS = ROOT / "docs/restart/cross_domain_ratio_declaration.json"
OUTPUT = ROOT / "docs/restart/noise_aware_selection_results.json.gz"
SOURCE_PATHS = ["docs/restart/noise_aware_selection_declaration.json",
                "docs/restart/cross_domain_ratio_declaration.json",
                "benchmarks/run_noise_aware_selection.py", "benchmarks/check_cross_domain_ratio.py",
                "benchmarks/phase17_ceo_oracle.py", "benchmarks/check_orthogonal_active_space.py",
                "clifford_qc/multivector.py", "clifford_qc/subspace/linalg.py",
                "clifford_qc/models/fcidump.py", "clifford_qc/models/spin.py",
                "clifford_qc/fermion.py", "benchmarks/data/h4_sto3g_r0.9.FCIDUMP",
                "benchmarks/data/lih_sto3g_r1.5949_cas4e4o.FCIDUMP"]


def setup(case):
    n = case["n"]
    if case["kind"] == "tfim":
        h = tfim(n, J=case["J"], h=case["h"]).hamiltonian.to_mv()
        psi = np.ones(1, complex)
        for q in range(n):
            theta, phi = case["theta"] + .1*(q % 2), (-1)**q*case["phi"]
            psi = np.kron(psi, [np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
        gens = [MV.scalar(n, 1), h/n, MV.from_label("Z"+"I"*(n-1))]
    else:
        h = fcidump_model(ROOT/case["fcidump"]).hamiltonian.to_mv()
        psi = np.zeros(2**n, complex)
        psi[int(case["reference_bits"][0], 2)] = np.cos(case["theta"])
        psi[int(case["reference_bits"][1], 2)] = np.exp(1j*case["phi"])*np.sin(case["theta"])
        gens = [MV.scalar(n, 1)]
        source = case["pair_source"]
        for target in case["pair_targets"]:
            transfer = (cdag_op(n, 2*target)*cdag_op(n, 2*target+1)
                        *c_op(n, 2*source+1)*c_op(n, 2*source))
            gens.append((MV.scalar(n, 1)+transfer-transfer.dagger())/np.sqrt(2))
    pairs = []
    for g in gens[1:]:
        selected = [gens[0], g]
        pairs.append({kind: [[a.dagger()*(h if kind == "H" else MV.scalar(n, 1))*b
                               for b in selected] for a in selected] for kind in ("S", "H")})
    words = sorted({code for pair in pairs for matrix in pair.values() for row in matrix
                    for op in row for code in op.terms if code != 0})
    return h, psi, pairs, words


def expectation(psi, code, n):
    indices = np.arange(2**n, dtype=np.int64)
    target = indices.copy()
    factors = np.ones(2**n, complex)
    for q in range(n):
        letter, mask = (code >> (2*q)) & 3, 1 << (n-1-q)
        occupied = (indices & mask) != 0
        if letter in (1, 2):
            target ^= mask
        if letter == 2:
            factors *= np.where(occupied, -1j, 1j)
        elif letter == 3:
            factors *= np.where(occupied, -1., 1.)
    value = np.vdot(psi[target], factors*psi)
    if abs(value.imag) > 1e-10 or abs(value.real) > 1 + 1e-10:
        raise ValueError("invalid Pauli expectation")
    return float(np.clip(value.real, -1, 1))


def attenuation(code, n, p):
    return (1-2*p)**sum((code >> (2*q)) & 3 in (1, 2) for q in range(n))


def matrix_values(pair, values):
    out = []
    for key in ("S", "H"):
        m = np.array([[sum(coef*values.get(code, 0.) for code, coef in op.terms.items())
                       for op in row] for row in pair[key]], complex)
        out.append((m+m.conj().T)/2)
    return out


def energy(pair, values):
    s, h = matrix_values(pair, values)
    diag = np.real(np.diag(s))
    if np.any(diag <= 1e-8):
        return None
    normalized = s/np.sqrt(diag[:, None]*diag[None, :])
    if np.linalg.eigvalsh(normalized)[0] <= 1e-6:
        return None
    try:
        r = solve_projected(s, h, tau_s=1e-6, max_condition=1e8)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return float(r.ground_energy) if r.effective_rank == 2 else None


def bias_penalty(pair, h, n, p):
    maxima = {}
    for key in ("S", "H"):
        maxima[key] = max(sum(abs(c)*(1-attenuation(code, n, p))
                              for code, c in op.terms.items())
                          for row in pair[key] for op in row)
    return .1*(maxima["S"]*sum(abs(c) for c in h.terms.values())+maxima["H"])


def _ratio(a, b):
    return None if b == 0 else float(a/b)


def run():
    declaration_bytes = DECLARATION.read_bytes()
    declaration = json.loads(declaration_bytes)
    assert declaration["schema"] == "clifford_qc.noise_aware_selection.declaration.v1"
    source = json.loads(INPUTS.read_text())
    assert [case["name"] for case in source["cases"]] == declaration["case_names"]
    outcomes, summaries = [], []
    shots, replicas = 5000, 64
    for ci, case in enumerate(source["cases"]):
        started = time.perf_counter()
        h, psi, pairs, words = setup(case)
        n = case["n"]
        data = case_data(case)  # assessment, never used by either selector
        ground = data["ground"]
        ideal = {0: 1., **{code: expectation(psi, code, n) for code in words}}
        loss_failure = 10*max(1., sum(abs(c) for c in h.terms.values()))
        setup_seconds = time.perf_counter()-started
        for pi, p in enumerate((.02, .1)):
            atten = np.array([attenuation(code, n, p) for code in words])
            probabilities = np.clip((1+np.array([ideal[code] for code in words])*atten)/2, 0, 1)
            exact_noisy = {0: 1., **{code: float(2*prob-1) for code, prob in zip(words, probabilities)}}
            penalties = [bias_penalty(pair, h, n, p) for pair in pairs]
            rows = []
            for replicate in range(replicas):
                rng = np.random.default_rng(20261101+10000*ci+1000*pi+replicate)
                train_counts = rng.binomial(shots, probabilities)
                eval_counts = rng.binomial(shots, probabilities)
                train = {0: 1., **{code: float(2*v/shots-1) for code, v in zip(words, train_counts)}}
                evaluate = {0: 1., **{code: float(2*v/shots-1) for code, v in zip(words, eval_counts)}}
                training = [energy(pair, train) for pair in pairs]
                scores = {"baseline": training,
                          "noise_aware": [None if v is None else v+penalties[i]
                                          for i, v in enumerate(training)]}
                arms = {}
                for arm in ("baseline", "noise_aware"):
                    valid = [(score, i) for i, score in enumerate(scores[arm]) if score is not None]
                    selected = min(valid)[1] if valid else None
                    evaluated = None if selected is None else energy(pairs[selected], evaluate)
                    failed = selected is None or evaluated is None
                    arms[arm] = {"candidate": selected,
                        "status": "selection_failed" if selected is None else
                                  "evaluation_failed" if evaluated is None else "success",
                        "evaluation_energy": evaluated,
                        "ideal_selected_span_energy": None if selected is None else energy(pairs[selected], ideal),
                        "noisy_exact_selected_span_energy": None if selected is None else energy(pairs[selected], exact_noisy),
                        "loss": loss_failure if failed else abs(evaluated-ground)}
                row = {"replicate": replicate, "seed": 20261101+10000*ci+1000*pi+replicate,
                       "training_counts": train_counts.tolist(), "evaluation_counts": eval_counts.tolist(),
                       "training_energies": training, "arms": arms}
                rows.append(row)
            base = np.array([row["arms"]["baseline"]["loss"] for row in rows])
            aware = np.array([row["arms"]["noise_aware"]["loss"] for row in rows])
            rng = np.random.default_rng(20261201+100*ci+pi)
            draw = rng.integers(0, replicas, size=(2000, replicas))
            denom = np.median(base[draw], axis=1)
            numer = np.median(aware[draw], axis=1)
            ratios = None if np.any(denom == 0) else np.quantile(numer/denom, [.025, .975]).tolist()
            med_base, med_aware = float(np.median(base)), float(np.median(aware))
            failures = {arm: sum(row["arms"][arm]["status"] != "success" for row in rows)
                        for arm in ("baseline", "noise_aware")}
            accepted = (med_base > 0 and med_aware <= .8*med_base and
                        failures["noise_aware"] <= failures["baseline"] and ratios is not None and ratios[1] < 1)
            summary = {"case": case["name"], "p": p, "ground_energy": ground,
                       "words": words, "unique_words": len(words), "shots_per_word_per_batch": shots,
                       "shots_per_arm_per_replicate": 2*len(words)*shots,
                       "settings_per_batch": len(words), "candidate_penalties": penalties,
                       "failure_loss": loss_failure, "failures": failures,
                       "median_loss": {"baseline": med_base, "noise_aware": med_aware},
                       "median_ratio": _ratio(med_aware, med_base), "bootstrap_ratio_95": ratios,
                       "cell_accepted": bool(accepted), "setup_seconds_advisory": setup_seconds}
            summaries.append(summary)
            outcomes.append({"case": case["name"], "p": p, "rows": rows})
    heldout = [x for x in summaries if x["case"] != "tfim4_development"]
    return {"schema": "clifford_qc.noise_aware_selection.results.v1",
            "declaration_sha256": hashlib.sha256(declaration_bytes).hexdigest(),
            "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_PATHS},
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "evidence": "finite-shot simulation; supplied noise model; no hardware or end-to-end cost claim",
            "summaries": summaries, "outcomes": outcomes,
            "accepted": all(row["cell_accepted"] for row in heldout)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    result = run()
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as handle:
        handle.write(gzip.compress(payload, mtime=0))
    print("saved", OUTPUT, "accepted", result["accepted"])


if __name__ == "__main__":
    main()

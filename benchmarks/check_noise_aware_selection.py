"""Independent replay of preregistered noise-selection results.

Never imports the producer or its Pauli products, state preparation, projection
solver, sampling helpers or summaries. Molecular FCIDUMP parsing and the
occupation/Pauli-letter oracle are shared with the cross-domain independent
checker; this is not independent molecular integral generation.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.check_cross_domain_ratio import case_data
from benchmarks.phase17_ceo_oracle import multiply

DESIGN = ROOT / "docs/restart/noise_aware_selection_declaration.json"
SOURCE = ROOT / "docs/restart/cross_domain_ratio_declaration.json"
RESULT = ROOT / "docs/restart/noise_aware_selection_results.json.gz"


def require(value, message):
    if not value:
        raise AssertionError(message)


def near(a, b, label, atol=2e-9):
    if a is None or b is None:
        require(a is b, f"{label}: null mismatch")
    else:
        require(np.allclose(a, b, atol=atol, rtol=1e-9), f"{label}: {a!r} != {b!r}")


def make_operators(data, n):
    h, gs = data["h_terms"], data["generator_terms"]
    pairs = []
    for g in gs[1:]:
        selected = [gs[0], g]
        operators = {}
        for kind in ("S", "H"):
            mid = {0: 1} if kind == "S" else h
            operators[kind] = [[multiply(multiply({k: v.conjugate() for k, v in a.items()}, mid, n), b, n)
                                for b in selected] for a in selected]
        pairs.append(operators)
    words = sorted({k for pair in pairs for matrix in pair.values() for row in matrix
                    for operator in row for k in operator if k != 0})
    return pairs, words


def ideal_expectations(data, n, words):
    # Occupation lookup and letter action independent of the producer's vectorized
    # state action; individual Pauli words may leave the symmetry sector.
    state = {int(k): complex(v) for k, v in zip(data["words"], data["state"]) if abs(v) > 1e-16}
    result = {0: 1.}
    for code in words:
        total = 0j
        for source, amplitude in state.items():
            target, phase = source, 1+0j
            for q in range(n):
                letter = (code >> (2*q)) & 3
                bit = (source >> (n-1-q)) & 1
                if letter in (1, 2):
                    target ^= 1 << (n-1-q)
                if letter == 2:
                    phase *= 1j if bit == 0 else -1j
                elif letter == 3 and bit:
                    phase *= -1
            total += state.get(target, 0j).conjugate()*phase*amplitude
        require(abs(total.imag) < 1e-10 and abs(total.real) <= 1+1e-10, "Pauli expectation")
        result[code] = float(np.clip(total.real, -1, 1))
    return result


def scale(code, n, p):
    count = sum(1 for q in range(n) if ((code >> (2*q)) & 3) in (1, 2))
    return (1-2*p)**count


def evaluate(pair, values):
    matrices = []
    for key in ("S", "H"):
        a = np.asarray([[sum(c*values[k] for k, c in op.items()) for op in row]
                        for row in pair[key]], dtype=complex)
        matrices.append((a+a.conj().T)/2)
    s, h = matrices
    diag = np.real(np.diag(s))
    if np.min(diag) <= 1e-8:
        return None
    d = np.diag(1/np.sqrt(diag))
    metric = d@s@d
    eigen, vectors = np.linalg.eigh(metric)
    if eigen[0] <= 1e-6:
        return None
    frame = d@vectors@np.diag(eigen**-.5)
    return float(np.linalg.eigvalsh(frame.conj().T@h@frame)[0])


def penalty(pair, h, n, p):
    bound = {}
    for key in ("S", "H"):
        bound[key] = max(sum(abs(c)*(1-scale(k, n, p)) for k, c in operator.items())
                         for row in pair[key] for operator in row)
    return .1*(bound["S"]*sum(abs(c) for c in h.values())+bound["H"])


def check(record):
    design_bytes = DESIGN.read_bytes()
    design = json.loads(design_bytes)
    require(record["schema"] == "clifford_qc.noise_aware_selection.results.v1", "result schema")
    require(record["declaration_sha256"] == hashlib.sha256(design_bytes).hexdigest(), "declaration digest")
    for path, expected in record["source_sha256"].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, f"source digest: {path}")
    cases = json.loads(SOURCE.read_text())["cases"]
    require([c["name"] for c in cases] == design["case_names"], "case grid")
    require(len(record["summaries"]) == len(record["outcomes"]) == 6, "cell denominator")
    accepted_cells = []
    for ci, case in enumerate(cases):
        n = case["n"]
        data = case_data(case)
        pairs, words = make_operators(data, n)
        ideal = ideal_expectations(data, n, words)
        norm = sum(abs(v) for v in data["h_terms"].values())
        for pi, p in enumerate((.02, .1)):
            ix = 2*ci+pi
            summary, outcome = record["summaries"][ix], record["outcomes"][ix]
            require(summary["case"] == outcome["case"] == case["name"] and summary["p"] == outcome["p"] == p,
                    "cell ordering")
            require(summary["words"] == words and summary["unique_words"] == len(words), "word universe")
            require(summary["settings_per_batch"] == len(words) and summary["shots_per_word_per_batch"] == 5000
                    and summary["shots_per_arm_per_replicate"] == 10000*len(words), "shot boundary")
            near(summary["ground_energy"], data["ground"], "ground energy")
            failure_loss = 10*max(1., norm)
            near(summary["failure_loss"], failure_loss, "failure loss")
            penalties = [penalty(pair, data["h_terms"], n, p) for pair in pairs]
            near(summary["candidate_penalties"], penalties, "channel penalties")
            probabilities = np.array([np.clip((1+ideal[k]*scale(k, n, p))/2, 0, 1) for k in words])
            exact_noisy = {0: 1., **{k: float(2*prob-1) for k, prob in zip(words, probabilities)}}
            require(len(outcome["rows"]) == 64, "replicate denominator")
            all_losses = {"baseline": [], "noise_aware": []}
            failures = {"baseline": 0, "noise_aware": 0}
            for rep, row in enumerate(outcome["rows"]):
                seed = 20261101+10000*ci+1000*pi+rep
                require(row["replicate"] == rep and row["seed"] == seed, "seed grid")
                rng = np.random.default_rng(seed)
                train_counts = rng.binomial(5000, probabilities)
                eval_counts = rng.binomial(5000, probabilities)
                require(row["training_counts"] == train_counts.tolist() and row["evaluation_counts"] == eval_counts.tolist(),
                        "sampled counts or stream order")
                train = {0: 1., **{k: float(2*x/5000-1) for k, x in zip(words, train_counts)}}
                evaluation = {0: 1., **{k: float(2*x/5000-1) for k, x in zip(words, eval_counts)}}
                energies = [evaluate(pair, train) for pair in pairs]
                for got, expected in zip(row["training_energies"], energies):
                    near(got, expected, "training energy")
                for arm in ("baseline", "noise_aware"):
                    valid = [(value+(penalties[i] if arm == "noise_aware" else 0), i)
                             for i, value in enumerate(energies) if value is not None]
                    selected = min(valid)[1] if valid else None
                    value = None if selected is None else evaluate(pairs[selected], evaluation)
                    status = "selection_failed" if selected is None else "evaluation_failed" if value is None else "success"
                    actual = row["arms"][arm]
                    require(actual["candidate"] == selected and actual["status"] == status, "selection and status")
                    near(actual["evaluation_energy"], value, "evaluation energy")
                    near(actual["ideal_selected_span_energy"], None if selected is None else evaluate(pairs[selected], ideal),
                         "ideal selected span")
                    near(actual["noisy_exact_selected_span_energy"], None if selected is None else evaluate(pairs[selected], exact_noisy),
                         "noisy selected span")
                    loss = failure_loss if status != "success" else abs(value-data["ground"])
                    near(actual["loss"], loss, "individual loss")
                    all_losses[arm].append(loss)
                    failures[arm] += status != "success"
            require(summary["failures"] == failures, "failure counts")
            base = np.array(all_losses["baseline"])
            aware = np.array(all_losses["noise_aware"])
            mb, ma = float(np.median(base)), float(np.median(aware))
            near(summary["median_loss"]["baseline"], mb, "baseline median")
            near(summary["median_loss"]["noise_aware"], ma, "aware median")
            near(summary["median_ratio"], None if mb == 0 else ma/mb, "median ratio")
            rng = np.random.default_rng(20261201+100*ci+pi)
            sample = rng.integers(0, 64, size=(2000, 64))
            denominator, numerator = np.median(base[sample], axis=1), np.median(aware[sample], axis=1)
            interval = None if np.any(denominator == 0) else np.quantile(numerator/denominator, [.025, .975]).tolist()
            near(summary["bootstrap_ratio_95"], interval, "bootstrap interval")
            accepted = bool(mb > 0 and ma <= .8*mb and failures["noise_aware"] <= failures["baseline"]
                            and interval is not None and interval[1] < 1)
            require(summary["cell_accepted"] == accepted, "cell verdict")
            if ci > 0:
                accepted_cells.append(accepted)
    require(record["accepted"] == all(accepted_cells), "overall held-out verdict")
    return len(accepted_cells)


def main():
    record = json.loads(gzip.decompress(RESULT.read_bytes()))
    print("independent selector replay passed; held-out cells:", check(record))


if __name__ == "__main__":
    main()

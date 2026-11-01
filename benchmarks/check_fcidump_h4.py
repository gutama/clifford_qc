"""Independent bounded audit of the published H4 occupation-space claim.

Does not import the producer, native JW/Pauli products, A-CASE, or its solver.
FCIDUMP parsing is shared with the library; the occupation-sign Hamiltonian and
Pauli-letter algebra are from the separately implemented Phase 17 oracle.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.phase17_ceo_oracle import _fermionic_action, hamiltonian, ladder, multiply

RECORD = ROOT / "benchmarks/reference_results/fcidump_h4.json"
FCIDUMP = ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"
RECORD_SHA256 = "f397d181d5d70801f13ba0e03e2a8edafc9bcd82ccf1658f10f1f3370f796b11"
INPUT_SHA256 = "bf32568461477cf1ec8e7c4aaae737b0c6c80bcbd83d51f783ec3a3daed5a922"
N = 8
REFERENCE = int("11110000", 2)


def require(value, message):
    if not value:
        raise AssertionError(message)


def close(actual, expected, label, tolerance=1e-10):
    if not np.allclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def excitation(label):
    match = re.fullmatch(r"E\(([0-9,]+)<-([0-9,]+)\)", label)
    require(match is not None, f"invalid excitation label: {label}")
    creation = tuple(int(x) for x in match.group(1).split(","))
    annihilation = tuple(int(x) for x in match.group(2).split(","))
    require(len(creation) == len(annihilation) in (1, 2), "invalid excitation rank")
    require(tuple(sorted(set(creation))) == creation and tuple(sorted(set(annihilation))) == annihilation,
            "excitation indices must be ordered and distinct")
    require(set(creation) <= {4, 5, 6, 7} and set(annihilation) <= {0, 1, 2, 3},
            "excitation not relative to the declared reference")
    require(sum(x % 2 for x in creation) == sum(x % 2 for x in annihilation), "spin-z change")
    return [(q, True) for q in creation] + [(q, False) for q in annihilation]


def generator_terms(sequence):
    terms = {0: 1.0}
    for q, create in sequence:
        terms = multiply(terms, ladder(N, q, create, fermionic=True), N)
    return {code: coefficient - coefficient.conjugate() for code, coefficient in terms.items()
            if abs(coefficient - coefficient.conjugate()) > 1e-12}


def audit(record, input_path=FCIDUMP):
    require(record["schema"] == "clifford_qc.active_space_result.v1", "record schema")
    require(hashlib.sha256(Path(input_path).read_bytes()).hexdigest() == INPUT_SHA256, "FCIDUMP digest")
    require(record["input"]["sha256"] == INPUT_SHA256, "record input binding")
    basis = [word for word in range(2**N)
             if sum((word >> (N-1-q)) & 1 for q in range(0, N, 2)) == 2
             and sum((word >> (N-1-q)) & 1 for q in range(1, N, 2)) == 2]
    require(len(basis) == record["system"]["sector_dimension"] == 36, "sector dimension")
    h, h_words = hamiltonian(input_path, basis)
    close(h, h.conj().T, "Hamiltonian Hermiticity")
    require(len(h_words) == record["system"]["hamiltonian_terms"] == 185, "Hamiltonian word count")
    spectrum = np.linalg.eigvalsh(h)
    close(spectrum[:4], record["mapped_oracle"]["lowest_four_energies"], "sector spectrum")
    close(spectrum[0], record["independent_reference"]["fci_energy"], "external FCI")
    close(h[basis.index(REFERENCE), basis.index(REFERENCE)].real,
          record["independent_reference"]["rhf_energy"], "reference energy")
    labels = record["adaptive_acase"]["labels"]
    require(labels[0] == "I" and len(labels) == len(set(labels)) == 9, "selected basis")
    cols = [basis.index(REFERENCE)]
    terms = [{0: 1.0}]
    for label in labels[1:]:
        seq = excitation(label)
        acted = _fermionic_action(REFERENCE, seq, N)
        require(acted is not None and acted[0] in basis, "invalid selected direction")
        cols.append(basis.index(acted[0]))
        terms.append(generator_terms(seq))
    require(len(cols) == len(set(cols)), "selected directions must be independent")
    prefix = [float(np.linalg.eigvalsh(h[np.ix_(cols[:k], cols[:k])])[0])
              for k in range(1, len(cols)+1)]
    close(prefix, record["adaptive_acase"]["energy_history"], "prefix energies")
    close(prefix[-1], record["adaptive_acase"]["ground_energy"], "selected energy")
    close(prefix[-1]-spectrum[0], record["adaptive_acase"]["error_hartree"], "selected error")
    words = set()
    max_h = 0
    for a, b in itertools.product(terms, repeat=2):
        adjoint = {code: coeff.conjugate() for code, coeff in a.items()}
        overlap = multiply(adjoint, b, N)
        element = multiply(multiply(adjoint, h_words, N), b, N)
        words.update(overlap)
        words.update(element)
        max_h = max(max_h, len(element))
    require(len(words) == record["adaptive_acase"]["word_universe"] == 7371, "selected word universe")
    require(max_h == record["adaptive_acase"]["max_hamiltonian_element_support"] == 3840, "maximum H-row support")
    all_cols = [basis.index(REFERENCE)]
    for rank in (1, 2):
        for occupied in itertools.combinations(range(4), rank):
            for virtual in itertools.combinations(range(4, 8), rank):
                if sum(q % 2 for q in occupied) != sum(q % 2 for q in virtual):
                    continue
                seq = [(q, True) for q in virtual] + [(q, False) for q in occupied]
                acted = _fermionic_action(REFERENCE, seq, N)
                require(acted is not None and acted[0] in basis, "invalid full SD direction")
                all_cols.append(basis.index(acted[0]))
    require(len(all_cols) == len(set(all_cols)) == record["complete_singles_doubles"]["basis_size"] == 27,
            "complete SD basis")
    full_energy = float(np.linalg.eigvalsh(h[np.ix_(all_cols, all_cols)])[0])
    close(full_energy, record["complete_singles_doubles"]["ground_energy"], "complete SD energy")
    close(full_energy-spectrum[0], record["complete_singles_doubles"]["error_hartree"], "complete SD error")
    return {"sector_dimension": len(basis), "prefixes_checked": len(prefix),
            "selected_words": len(words), "max_h_support": max_h, "full_sd_size": len(all_cols)}


def main():
    raw = RECORD.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == RECORD_SHA256, "frozen record digest")
    result = audit(json.loads(raw))
    print("independent H4 occupation and Pauli-letter audit passed:", result)


if __name__ == "__main__":
    main()

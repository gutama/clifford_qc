"""Stim validation harness (Phase 3).

Clifford-only IR programs convert to ``stim.Circuit``/``stim.Tableau``.
``CliffordMap`` exposes the conjugation action ``P -> U P U†`` computed by
stim's stabilizer tableau, which the tests cross-check against exact MV
conjugation at small ``n`` and use alone at large ``n`` to validate the
Gottesman-Knill / blade-preservation claims.
"""

from __future__ import annotations

import stim

from ..ir import PauliWord, Program

# IR Clifford name -> stim gate name
STIM_NAMES = {"X": "X", "Y": "Y", "Z": "Z", "H": "H", "S": "S", "SDG": "S_DAG",
              "CX": "CX", "CZ": "CZ", "SWAP": "SWAP"}


def program_to_stim(program: Program) -> stim.Circuit:
    """Extract the Clifford-only circuit of a program as a stim.Circuit."""
    if not program.is_clifford_only():
        raise ValueError("program contains non-Clifford rotors; extract or lower them first")
    circuit = stim.Circuit()
    for op in program.ops:
        circuit.append(STIM_NAMES[op.name], list(op.qubits))
    return circuit


def clifford_tableau(program: Program) -> stim.Tableau:
    return program_to_stim(program).to_tableau()


def _pauli_word_to_stim(word: PauliWord) -> stim.PauliString:
    return stim.PauliString(word.label)


def _stim_to_pauli_word(pauli: stim.PauliString, n: int) -> tuple[complex, PauliWord]:
    label = "".join("IXYZ"[pauli[j]] for j in range(n))
    return complex(pauli.sign), PauliWord.from_label(label, n)


class CliffordMap:
    """The conjugation action of a Clifford program on Pauli words.

    ``conjugate(P)`` returns ``(phase, P')`` with ``U P U† = phase * P'``.
    A Clifford always maps one Pauli word to one Pauli word with phase in
    {1, -1, i, -i} — the algebraic form of the Gottesman-Knill statement
    (single blades stay single blades).
    """

    def __init__(self, n: int, tableau: stim.Tableau):
        if len(tableau) != n:
            raise ValueError(f"tableau acts on {len(tableau)} qubits, expected {n}")
        self.n = n
        self.tableau = tableau

    @staticmethod
    def from_program(program: Program) -> "CliffordMap":
        return CliffordMap(program.n, clifford_tableau(program))

    def conjugate(self, word: PauliWord) -> tuple[complex, PauliWord]:
        if word.n != self.n:
            raise ValueError("word and map act on different qubit counts")
        return _stim_to_pauli_word(self.tableau(_pauli_word_to_stim(word)), self.n)

    def generator_images(self) -> dict[str, tuple[complex, PauliWord]]:
        """Images of the X_j / Z_j generators, keyed by 'X3'-style names."""
        out = {}
        for j in range(self.n):
            for letter, output in (("X", self.tableau.x_output(j)),
                                   ("Z", self.tableau.z_output(j))):
                out[f"{letter}{j}"] = _stim_to_pauli_word(output, self.n)
        return out

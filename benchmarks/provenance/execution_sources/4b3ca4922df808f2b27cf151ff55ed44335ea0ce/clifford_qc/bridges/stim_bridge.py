"""Stim validation harness (Phase 3).

Clifford-only IR programs convert to ``stim.Circuit``/``stim.Tableau``.
``CliffordMap`` exposes the conjugation action ``P -> U P U†`` computed by
stim's stabilizer tableau, which the tests cross-check against exact MV
conjugation at small ``n`` and use alone at large ``n`` to validate the
Gottesman-Knill / blade-preservation claims.
"""

from __future__ import annotations

from ..capabilities import require

require('stabilizer_backend', feature='clifford_qc.bridges.stim_bridge')

import stim

from ..ir import PauliWord, Program, Rotor, clifford_angle_index

# IR Clifford name -> stim gate name
STIM_NAMES = {"X": "X", "Y": "Y", "Z": "Z", "H": "H", "S": "S", "SDG": "S_DAG",
              "CX": "CX", "CZ": "CZ", "SWAP": "SWAP"}

_TARGETS = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}


def _pauli_product_targets(word: PauliWord) -> list:
    targets = []
    for j in word.support():
        if targets:
            targets.append(stim.target_combiner())
        targets.append(_TARGETS[word.letter(j)](j))
    return targets


def _append_clifford_rotor(circuit: stim.Circuit, word: PauliWord, k: int) -> None:
    """Append exp(-i k pi P/4) for k in {0,1,2,3} (up to global phase).

    stim's SPP gate conjugates exactly like exp(-i pi P/4), so k=1 is SPP,
    k=3 its dagger, and k=2 is the Pauli word itself (exp(-i pi P/2) = -iP).
    """
    if k == 0 or not word.support():
        return
    if k == 2:
        for j in word.support():
            circuit.append(word.letter(j), [j])
    else:
        circuit.append("SPP" if k == 1 else "SPP_DAG", _pauli_product_targets(word))


def program_to_stim(program: Program, values=None) -> stim.Circuit:
    """Extract the Clifford circuit of a program as a stim.Circuit.

    NamedClifford gates map to their stim names; rotors are accepted when
    their (resolved) angle is a multiple of pi/2 and lower to SPP/SPP_DAG
    or Pauli gates. Any other rotor raises.
    """
    bindings = program.parameters.bind(values) if values is not None else None
    circuit = stim.Circuit()
    for op in program.ops:
        if isinstance(op, Rotor):
            if not op.is_clifford(bindings):
                raise ValueError(
                    f"rotor exp(-i theta {op.word.label}/2) with theta={op.angle!r} "
                    "is not a Clifford operation; extract or lower it first")
            k = clifford_angle_index(op.resolved_angle(bindings))
            _append_clifford_rotor(circuit, op.word, k)
        else:
            circuit.append(STIM_NAMES[op.name], list(op.qubits))
    return circuit


def clifford_tableau(program: Program, values=None) -> stim.Tableau:
    return program_to_stim(program, values).to_tableau()


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
    def from_program(program: Program, values=None) -> "CliffordMap":
        return CliffordMap(program.n, clifford_tableau(program, values))

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

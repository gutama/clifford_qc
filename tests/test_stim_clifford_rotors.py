"""Clifford-angle Pauli rotors lower to stim and agree with exact MV
conjugation (backlog item 14: 'stim and MV conjugation agree')."""

import math
import random

import pytest

stim = pytest.importorskip("stim")

from clifford_qc import Program, PauliWord, conjugate_pauli_word
from clifford_qc.bridges.stim_bridge import CliffordMap, program_to_stim

LETTERS = "IXYZ"
ANGLES = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, -math.pi / 2, 2 * math.pi]


def _random_clifford_rotor_program(n, depth, rng):
    prog = Program(n)
    for _ in range(depth):
        if rng.random() < 0.5:
            label = "".join(rng.choice(LETTERS) for _ in range(n))
            if set(label) == {"I"}:
                continue
            prog.rotor(label, rng.choice(ANGLES))
        else:
            name = rng.choice(["H", "S", "CX", "CZ"])
            if name in ("CX", "CZ"):
                a, b = rng.sample(range(n), 2)
                prog.clifford(name, a, b)
            else:
                prog.clifford(name, rng.randrange(n))
    return prog


@pytest.mark.parametrize("seed", range(8))
def test_clifford_rotor_conjugation_matches_mv(seed):
    rng = random.Random(seed)
    n = rng.choice([2, 3])
    prog = _random_clifford_rotor_program(n, depth=6, rng=rng)
    cmap = CliffordMap.from_program(prog)
    for code in range(1, 4 ** n):
        word = PauliWord(n, code)
        phase_stim, image_stim = cmap.conjugate(word)
        phase_mv, image_mv = conjugate_pauli_word(prog, word)
        assert image_stim.code == image_mv.code
        assert phase_stim == pytest.approx(phase_mv, abs=1e-9)


def test_non_clifford_rotor_rejected():
    prog = Program(2)
    prog.rotor("XY", 0.4)
    with pytest.raises(ValueError, match="not a Clifford"):
        program_to_stim(prog)


def test_parameterized_rotor_needs_clifford_binding():
    from clifford_qc import Parameter
    prog = Program(2)
    prog.rotor("XY", Parameter("a"))
    with pytest.raises(ValueError, match="not a Clifford"):
        program_to_stim(prog)
    circuit = program_to_stim(prog, values=[math.pi / 2])
    assert len(circuit) == 1


def test_identity_and_full_turn_rotors_drop_out():
    prog = Program(2)
    prog.rotor("XY", 0.0).rotor("ZZ", 2 * math.pi)
    assert len(program_to_stim(prog)) == 0

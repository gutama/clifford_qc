"""Stim validation harness (Phase 3): Clifford extraction, CliffordMap
cross-validation against exact MV conjugation, and large-n checks of the
Gottesman-Knill / blade-preservation claim."""

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from clifford_qc import Program, conjugate_pauli_word
from clifford_qc.ir import PauliWord
from clifford_qc.bridges.stim_bridge import (
    CliffordMap, clifford_tableau, program_to_stim,
)

ONE_QUBIT = ["X", "Y", "Z", "H", "S", "SDG"]
TWO_QUBIT = ["CX", "CZ", "SWAP"]


def random_clifford_program(n, depth, rng):
    prog = Program(n)
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.5:
            a, b = rng.choice(n, size=2, replace=False)
            prog.clifford(str(rng.choice(TWO_QUBIT)), int(a), int(b))
        else:
            prog.clifford(str(rng.choice(ONE_QUBIT)), int(rng.integers(n)))
    return prog


def random_word(n, rng):
    label = "".join(rng.choice(list("IXYZ")) for _ in range(n))
    if set(label) == {"I"}:
        label = "X" + label[1:]
    return PauliWord.from_label(label, n)


class TestExtraction:
    def test_rejects_non_clifford(self):
        with pytest.raises(ValueError):
            program_to_stim(Program(1).rotor("Z", 0.3))

    def test_every_gate_converts(self):
        prog = Program(2)
        for name in ONE_QUBIT:
            prog.clifford(name, 0)
        for name in TWO_QUBIT:
            prog.clifford(name, 0, 1)
        circuit = program_to_stim(prog)
        assert len(circuit) == len(ONE_QUBIT) + len(TWO_QUBIT)
        assert clifford_tableau(prog) is not None


class TestCliffordMapAgainstMV:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_circuits_small_n(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 5))
        prog = random_clifford_program(n, depth=15, rng=rng)
        cmap = CliffordMap.from_program(prog)
        for _ in range(8):
            word = random_word(n, rng)
            stim_phase, stim_word = cmap.conjugate(word)
            mv_phase, mv_word = conjugate_pauli_word(prog, word)
            assert stim_word.label == mv_word.label
            assert abs(stim_phase - mv_phase) < 1e-9

    def test_known_map(self):
        prog = Program(2).clifford("H", 0).clifford("CX", 0, 1)
        cmap = CliffordMap.from_program(prog)
        phase, word = cmap.conjugate(PauliWord.from_label("XI"))
        assert (phase, word.label) == (1 + 0j, "ZI")
        phase, word = cmap.conjugate(PauliWord.from_label("ZI"))
        assert (phase, word.label) == (1 + 0j, "XX")

    def test_generator_images_match_mv(self):
        rng = np.random.default_rng(11)
        prog = random_clifford_program(3, depth=12, rng=rng)
        images = CliffordMap.from_program(prog).generator_images()
        for j in range(3):
            for letter in "XZ":
                label = "".join(letter if k == j else "I" for k in range(3))
                mv_phase, mv_word = conjugate_pauli_word(prog, PauliWord.from_label(label))
                phase, word = images[f"{letter}{j}"]
                assert word.label == mv_word.label and abs(phase - mv_phase) < 1e-9


class TestLargeN:
    """Gottesman-Knill at scales the dense/MV backends cannot reach."""

    def test_blade_preservation_n100(self):
        rng = np.random.default_rng(42)
        n = 100
        prog = random_clifford_program(n, depth=400, rng=rng)
        cmap = CliffordMap.from_program(prog)
        words = [random_word(n, rng) for _ in range(25)]
        for word in words:
            phase, image = cmap.conjugate(word)
            # a Clifford maps one Pauli word to exactly one Pauli word
            assert isinstance(image, PauliWord) and image.n == n
            assert phase in (1, -1, 1j, -1j)
        assert cmap.preserves_support_size(words)

    def test_group_property_composition(self):
        rng = np.random.default_rng(7)
        n = 60
        p1 = random_clifford_program(n, depth=100, rng=rng)
        p2 = random_clifford_program(n, depth=100, rng=rng)
        combined = Program(n, [*p1.ops, *p2.ops])
        m1, m2 = CliffordMap.from_program(p1), CliffordMap.from_program(p2)
        mc = CliffordMap.from_program(combined)
        for _ in range(10):
            word = random_word(n, rng)
            ph1, w1 = m1.conjugate(word)
            ph2, w2 = m2.conjugate(w1)
            phc, wc = mc.conjugate(word)
            assert wc.label == w2.label and abs(phc - ph1 * ph2) < 1e-9

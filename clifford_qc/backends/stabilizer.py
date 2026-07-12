"""Stabilizer-state backend (requires the ``stim`` extra).

Evaluates Pauli expectations of Clifford programs — named Clifford gates
plus Clifford-angle Pauli rotors — through stim's tableau simulator, so
discrete Clifford-point searches run at qubit counts far beyond the dense
or multivector backends. Every expectation on a stabilizer state is
exactly +1, -1, or 0.

Import this module directly (``from clifford_qc.backends.stabilizer
import StimBackend``); it is deliberately not re-exported by
``clifford_qc.backends`` so the core package keeps importing with numpy
alone.
"""

from __future__ import annotations

import stim

from ..ir import PauliSum, PauliWord, Program
from ..bridges.stim_bridge import program_to_stim


class StimBackend:
    """Exact expectations on stabilizer states prepared by Clifford programs."""

    def simulator(self, program: Program, values=None) -> stim.TableauSimulator:
        sim = stim.TableauSimulator()
        sim.do_circuit(program_to_stim(program, values))
        return sim

    def expectation(self, program: Program, observable: PauliSum, values=None,
                    initial_state=None) -> float:
        if initial_state is not None:
            raise ValueError("StimBackend prepares from |0...0>; bake the "
                             "initial state into the program")
        sim = self.simulator(program, values)
        total = 0.0
        for word, coeff in observable.items():
            ev = sim.peek_observable_expectation(stim.PauliString(word.label))
            total += coeff.real * ev
        return total

    def word_expectation(self, program: Program, word: PauliWord, values=None) -> int:
        """<w> on the stabilizer state: exactly -1, 0, or +1."""
        return self.simulator(program, values).peek_observable_expectation(
            stim.PauliString(word.label))

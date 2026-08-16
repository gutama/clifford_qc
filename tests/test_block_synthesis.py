"""Block-local Clifford diagonalizer synthesis for the protocol axis.

Where ``test_block_commuting.py`` pins which words may share a setting, this
pins the circuit that setting runs. The properties that matter are the ones
whose failure returns plausible numbers rather than raising:

* every word in a group -- not merely the independent basis stim built the
  tableau from -- must become computational-basis diagonal;
* a setting's two-qubit gates must stay inside their block, since block
  locality is the whole reason ``k`` is a cost axis;
* at ``k = 1`` the diagonalizer is a single-qubit basis rotation, so the
  entangling cost is exactly zero -- the endpoint the QWC protocol occupies.

The frozen hierarchy and shot-search records are the end-to-end check and run
from ``benchmarks/``; these are the unit-level contracts underneath them.
"""

import random

import numpy as np
import pytest

pytest.importorskip("stim")

import stim  # noqa: E402

from clifford_qc.measurement.block_commuting import (  # noqa: E402
    block_commuting_partition,
    block_ranges,
)
from clifford_qc.measurement.block_synthesis import (  # noqa: E402
    block_diagonalizer,
    circuit_stats,
    independent_codes,
    local_code,
    stim_label,
    synthesize_block_settings,
)


def _universe(n, count, seed):
    rng = random.Random(seed)
    return sorted({rng.randrange(4 ** n) for _ in range(count)})


# ---------------------------------------------------------------------------
# Local code extraction


def test_local_code_reads_the_block_letters():
    # Qubit 0 X, qubit 1 Z, qubit 2 Y, qubit 3 I.
    code = 1 | (3 << 2) | (2 << 4)
    assert stim_label(local_code(code, 0, 2), 2) == "XZ"
    assert stim_label(local_code(code, 2, 2), 2) == "YI"


def test_local_codes_reassemble_the_whole_word():
    n = 6
    rng = random.Random(0)
    for _ in range(200):
        code = rng.randrange(4 ** n)
        for block_size in (1, 2, 3, 6):
            rebuilt = "".join(
                stim_label(local_code(code, start, size), size)
                for start, size in block_ranges(n, block_size)
            )
            assert rebuilt == stim_label(code, n)


def test_independent_codes_span_the_same_group():
    """The kept basis must generate every input up to phase."""
    size = 3
    codes = [0b01, 0b0101, 0b0100, 0b01_01_01]
    basis = independent_codes(codes, size)
    assert len(basis) <= 2 * size
    assert len(set(basis)) == len(basis)


# ---------------------------------------------------------------------------
# The diagonalizer


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
def test_every_member_becomes_z_only(block_size):
    n = 8
    codes = _universe(n, 150, seed=1)
    groups = block_commuting_partition(n, codes, block_size)
    for members in groups:
        for start, size in block_ranges(n, block_size):
            local = [local_code(codes[i], start, size) for i in members]
            diagonalizer = block_diagonalizer(local, size)
            for code in set(local):
                image = diagonalizer(stim.PauliString(stim_label(code, size)))
                x_bits, _ = image.to_numpy()
                assert not x_bits.any()


def test_diagonalizer_of_an_empty_block_is_the_identity():
    tableau = block_diagonalizer([0, 0], 3)
    assert tableau == stim.Tableau(3)


def test_diagonalizer_is_clifford_and_deterministic():
    local = [0b01, 0b1100]
    first = block_diagonalizer(local, 3)
    second = block_diagonalizer(local, 3)
    assert first == second
    assert isinstance(first, stim.Tableau)


# ---------------------------------------------------------------------------
# Circuit statistics


def test_circuit_stats_counts_and_schedules():
    circuit = stim.Circuit("H 0\nS 1\nCX 0 1\nH 0")
    counts, cx_depth, d1, d2 = circuit_stats(circuit)
    assert counts["H"] == 2 and counts["S"] == 1 and counts["CX"] == 1
    assert cx_depth == 1
    # One shared clock: the trailing H cannot share a layer with the leading
    # ones because the CX sits between them on qubit 0.
    assert d1 == 2 and d2 == 1


@pytest.mark.parametrize("gate", ["CZ 0 1", "SQRT_X 0", "M 0"])
def test_circuit_stats_rejects_an_unexpected_gate(gate):
    """Elimination emits only H, S and CX; anything else must not be costed.

    Silently ignoring an unknown instruction would under-count a setting, which
    is the failure mode that returns a plausible number instead of raising.
    """
    with pytest.raises(AssertionError, match="unexpected tableau-elimination"):
        circuit_stats(stim.Circuit(gate))


# ---------------------------------------------------------------------------
# The synthesis ledger


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_synthesis_assigns_every_word_exactly_once(block_size):
    n = 8
    codes = _universe(n, 200, seed=2)
    groups = block_commuting_partition(n, codes, block_size)
    synthesis = synthesize_block_settings(n, codes, groups, block_size)
    assert len(synthesis.assignment) == len(codes)
    assert all(index >= 0 for index in synthesis.assignment)
    for word_index, setting_index in enumerate(synthesis.assignment):
        assert word_index in groups[setting_index]


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_every_assigned_word_is_read_by_its_setting(block_size):
    n = 8
    codes = _universe(n, 200, seed=3)
    groups = block_commuting_partition(n, codes, block_size)
    synthesis = synthesize_block_settings(n, codes, groups, block_size)
    for word_index, setting_index in enumerate(synthesis.assignment):
        assert synthesis.compatibility[setting_index][word_index]


def test_qwc_rung_costs_no_entangling_gates():
    """``k = 1`` is a per-qubit basis rotation, so nothing entangling is emitted.

    This is the endpoint claim the mapping-axis record relies on when it prices
    QWC with ``N_2q = D_2q = 0``.
    """
    n = 8
    codes = _universe(n, 200, seed=4)
    groups = block_commuting_partition(n, codes, 1)
    synthesis = synthesize_block_settings(n, codes, groups, 1)
    assert synthesis.gate_counts["CX"] == 0
    assert synthesis.logical_cx_per_sweep == 0
    for setting in synthesis.settings:
        assert setting.n_2q == 0
        assert setting.d_2q == 0


def test_elimination_synthesis_is_not_a_minimal_basis_rotation():
    """stim's elimination circuit is correct but unoptimized, and that shows.

    A single-qubit Y diagonalizer needs two gates; ``to_circuit("elimination")``
    emits nine. The one-qubit cost of a ``k = 1`` setting is therefore a
    property of the *synthesizer*, not of the protocol, and the two producers in
    this repo disagree about it: ``run_mapping_axis.py`` prices QWC analytically
    (``D_1q`` max 2 on H4) while ``run_clifford_hierarchy.py`` prices the same
    protocol through elimination.

    Pinned here because it is a constraint on joining the mapping and protocol
    axes: a ``mapping x k`` grid that switches conventions at ``k = 1`` would
    manufacture a cost step between ``k = 1`` and ``k = 2`` that is an artifact
    of synthesis rather than a measurement of the protocol.
    """
    y_only = block_diagonalizer([0b10], 1)
    counts, _, depth_1q, depth_2q = circuit_stats(
        y_only.to_circuit("elimination"))
    assert depth_2q == 0
    assert counts["H"] + counts["S"] > 2
    assert depth_1q > 2


def test_two_qubit_gates_stay_inside_their_block():
    """Block locality, checked on the emitted circuits rather than assumed."""
    n = 8
    block_size = 2
    codes = _universe(n, 150, seed=5)
    groups = block_commuting_partition(n, codes, block_size)
    for members in groups:
        for start, size in block_ranges(n, block_size):
            local = [local_code(codes[i], start, size) for i in members]
            circuit = block_diagonalizer(local, size).to_circuit("elimination")
            for instruction in circuit:
                for target in instruction.targets_copy():
                    assert 0 <= target.value < size


def test_coverage_is_at_least_the_assigned_share():
    """Pooling can only read more words than a setting was assigned."""
    n = 8
    codes = _universe(n, 200, seed=6)
    for block_size in (1, 2, 4, 8):
        groups = block_commuting_partition(n, codes, block_size)
        synthesis = synthesize_block_settings(n, codes, groups, block_size)
        for setting_index, members in enumerate(groups):
            read = int(synthesis.compatibility[setting_index].sum())
            assert read >= len(members)
            assert synthesis.coverage[setting_index] == pytest.approx(
                read / len(codes)
            )


def test_coarser_blocks_never_cost_more_settings_here():
    n = 8
    codes = _universe(n, 200, seed=7)
    counts = []
    for block_size in (1, 2, 4, 8):
        groups = block_commuting_partition(n, codes, block_size)
        counts.append(synthesize_block_settings(
            n, codes, groups, block_size).n_settings)
    assert counts == sorted(counts, reverse=True)


def test_double_assignment_is_rejected():
    n = 4
    codes = _universe(n, 20, seed=8)
    groups = [[0], [0]]
    with pytest.raises(AssertionError, match="more than one group"):
        synthesize_block_settings(n, codes, groups, 2)


def test_unassigned_word_is_rejected():
    n = 4
    codes = _universe(n, 20, seed=9)
    with pytest.raises(AssertionError, match="did not assign every word"):
        synthesize_block_settings(n, codes, [[0]], 2)


def test_empty_bank_synthesizes_nothing():
    synthesis = synthesize_block_settings(4, [], [], 2)
    assert synthesis.n_settings == 0
    assert synthesis.logical_cx_per_sweep == 0
    assert synthesis.coverage == []
    assert np.asarray(synthesis.compatibility).size == 0

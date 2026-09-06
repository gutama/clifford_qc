"""Public Phase 14a compiled measurement-plan boundary."""

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("stim")

from clifford_qc import X, Y
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import (
    CompiledMeasurementSampler,
    GroupedWordCache,
    compile_block_measurement_plan,
)
from clifford_qc.states import bell_density, plus_density


def test_measurement_import_does_not_eagerly_load_stim():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import clifford_qc.measurement; "
                "assert 'stim' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_public_plan_covers_qwc_and_fully_commuting_endpoints():
    codes = tuple(sorted(PauliWord.from_label(label).code
                         for label in ("XX", "YY", "ZZ")))
    qwc = compile_block_measurement_plan(2, codes, 1)
    fully_commuting = compile_block_measurement_plan(2, codes, 2)

    assert len(qwc.groups) == 3
    assert len(fully_commuting.groups) == 1
    assert len(qwc.settings) == len(qwc.resources) == 3
    assert len(fully_commuting.settings) == len(fully_commuting.resources) == 1
    assert all(resource.n_2q == resource.d_2q == 0 for resource in qwc.resources)
    assert fully_commuting.resources[0].n_2q > 0
    assert qwc.assignment == (0, 1, 2)
    assert fully_commuting.assignment == (0, 0, 0)
    assert qwc.compatibility.shape == (3, 3)
    assert fully_commuting.compatibility.shape == (1, 3)
    assert bool(fully_commuting.compatibility.all())
    with pytest.raises(ValueError):
        fully_commuting.compatibility[0, 0] = False


def test_public_plan_settings_preserve_signed_joint_readouts():
    words = [PauliWord.from_label(label) for label in ("XX", "YY", "ZZ")]
    codes = tuple(sorted(word.code for word in words))
    plan = compile_block_measurement_plan(2, codes, 2)

    batch = CompiledMeasurementSampler(17).sample_from_state(
        bell_density(), plan.settings, 200
    )
    cache = GroupedWordCache(2)
    cache.add_batch(batch)
    expected = {"XX": 1.0, "YY": -1.0, "ZZ": 1.0}
    for word in words:
        assert cache.candidate_estimate({word.code: 1.0}) == pytest.approx(
            expected[word.label]
        )


def test_declared_spin_conserving_jw_source_invokes_phase13_gate():
    code = PauliWord.from_label("ZIII").code
    valid = X(4, 0) * X(4, 1) + Y(4, 0) * Y(4, 1)
    plan = compile_block_measurement_plan(
        4, [code], 1, spin_conserving_jw_hamiltonian=valid
    )
    assert plan.spin_conserving_x_rank == 1

    violating = (
        X(4, 0) * X(4, 1)
        + X(4, 0) * X(4, 2)
        + X(4, 0) * X(4, 3)
    )
    with pytest.raises(ValueError, match=r"r_X=3 exceeds 2"):
        compile_block_measurement_plan(
            4, [code], 1, spin_conserving_jw_hamiltonian=violating
        )

    # Generic grouping makes no spin-conservation claim and therefore remains
    # usable for arbitrary Pauli problems.
    assert compile_block_measurement_plan(4, [code], 1).spin_conserving_x_rank is None


def test_supplied_groups_are_checked_before_synthesis():
    codes = [PauliWord.from_label(label).code for label in ("XX", "XZ")]
    with pytest.raises(ValueError, match="not block-wise commuting"):
        compile_block_measurement_plan(2, codes, 1, groups=[[0, 1]])
    with pytest.raises(ValueError, match="assign every word exactly once"):
        compile_block_measurement_plan(2, codes, 1, groups=[[0], [0]])


def test_compiled_multi_reader_variance_matches_monte_carlo():
    """Pooling splits, rather than duplicates, a multiply-readable word."""
    xx = PauliWord.from_label("XX")
    zz = PauliWord.from_label("ZZ")
    codes = tuple(sorted((xx.code, zz.code)))
    plan = compile_block_measurement_plan(
        2, codes, 2, groups=((0,), (1,))
    )
    shots = 300
    estimates = []
    predicted = []
    for seed in range(300):
        batch = CompiledMeasurementSampler(seed).sample_from_state(
            plus_density(2), plan.settings, shots
        )
        cache = GroupedWordCache(2, pooling="shots")
        cache.add_batch(batch)
        weights = cache.group_weights(zz.code)
        assert len(weights) == 2
        assert sum(weights.values()) == pytest.approx(1.0)
        estimates.append(cache.candidate_estimate({zz.code: 1.0}))
        predicted.append(sum(
            variance / count
            for count, variance, _ in cache.candidate_group_terms({zz.code: 1.0})
        ))

    empirical = np.var(estimates, ddof=1)
    assert np.mean(predicted) == pytest.approx(empirical, rel=0.25)
    assert empirical == pytest.approx(1.0 / (2 * shots), rel=0.25)

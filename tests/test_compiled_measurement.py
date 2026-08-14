"""Exact joint sampling checks for Clifford-diagonalized settings."""

import subprocess
import sys

import pytest

pytest.importorskip("stim")

from benchmarks.run_clifford_hierarchy import _compiled_settings
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.cache import GroupedWordCache
from clifford_qc.measurement.compiled import CompiledMeasurementSampler
from clifford_qc.states import bell_density


def test_importing_measurement_does_not_eagerly_load_subspace():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import clifford_qc.measurement; "
                "assert 'clifford_qc.subspace' not in sys.modules; "
                "assert 'clifford_qc.subspace.adaptive' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_compiled_bell_setting_preserves_joint_pauli_outcomes():
    words = [PauliWord.from_label(label) for label in ("XX", "YY", "ZZ")]
    codes = sorted(word.code for word in words)
    settings = _compiled_settings(2, codes, [list(range(len(codes)))], 2)
    assert len(settings) == 1
    setting = settings[0]
    assert set(setting.readouts) == set(codes)

    batch = CompiledMeasurementSampler(17).sample_from_state(
        bell_density(), settings, 200
    )
    cache = GroupedWordCache(2)
    cache.add_batch(batch)
    expected = {"XX": 1.0, "YY": -1.0, "ZZ": 1.0}
    for word in words:
        assert cache.candidate_estimate({word.code: 1.0}) == pytest.approx(
            expected[word.label]
        )


def test_compiled_pooling_reuses_compatible_setting_histograms():
    xx = PauliWord.from_label("XX")
    zz = PauliWord.from_label("ZZ")
    # Both single-word settings happen to diagonalize ZZ, but only the second
    # owns it.  Pooled reconstruction must use both joint histograms.
    settings = _compiled_settings(2, sorted([xx.code, zz.code]), [[0], [1]], 2)
    cache = GroupedWordCache(2, pooling="shots")
    cache.add_batch(CompiledMeasurementSampler(4).sample_from_state(
        bell_density(), settings, 25
    ))
    assert cache.reader_count(zz.code) == 2
    assert cache.shots(zz.code) == 50
    assert cache.candidate_estimate({zz.code: 1.0}) == pytest.approx(1.0)

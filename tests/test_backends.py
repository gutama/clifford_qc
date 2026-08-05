"""Backend protocol conformance: exact MV vs dense reference, finite-shot
sampling statistics and determinism."""

import pytest

from clifford_qc import Program
from clifford_qc.backends import (
    ExactMVBackend, DenseStatevectorBackend, FiniteShotBackend, MeasurementBatch,
    Backend, SamplingBackend,
)
from clifford_qc.ir import PauliSum, PauliWord


def _program():
    prog = Program(3)
    prog.clifford("H", 0).clifford("CX", 0, 1)
    prog.rotor("XYI", 0.7).rotor("IZZ", -0.4)
    return prog


OBS = PauliSum.from_labels({"ZZI": 0.5, "XII": -1.0, "IYZ": 0.25})


def test_exact_mv_and_dense_agree():
    exact = ExactMVBackend().expectation(_program(), OBS)
    dense = DenseStatevectorBackend().expectation(_program(), OBS)
    assert exact == pytest.approx(dense, abs=1e-10)


def test_protocol_conformance():
    assert isinstance(ExactMVBackend(), Backend)
    assert isinstance(DenseStatevectorBackend(), Backend)
    assert isinstance(FiniteShotBackend(seed=0), SamplingBackend)
    assert not isinstance(FiniteShotBackend(seed=0), Backend)


def test_dense_state_method_returns_the_same_density_as_exact():
    assert DenseStatevectorBackend().state(_program()).is_close(
        ExactMVBackend().state(_program()), 1e-10)


def test_support_history_tracked():
    backend = ExactMVBackend()
    backend.state(_program(), ())
    assert len(backend.last_support_history) == 5  # initial + 4 ops
    assert backend.support_peak == max(backend.last_support_history)


def test_finite_shot_deterministic_by_seed():
    words = [PauliWord.from_label("ZZI"), PauliWord.from_label("XII")]
    a = FiniteShotBackend(seed=42).sample_paulis(_program(), words, 500)
    b = FiniteShotBackend(seed=42).sample_paulis(_program(), words, 500)
    c = FiniteShotBackend(seed=43).sample_paulis(_program(), words, 500)
    assert a.plus_counts == b.plus_counts
    assert a.plus_counts != c.plus_counts or a.shots != c.shots
    assert a.circuits == 2
    assert all(v == 500 for v in a.shots.values())


def test_finite_shot_means_converge_to_exact():
    word = PauliWord.from_label("ZZI")
    exact = ExactMVBackend().expectation(_program(), PauliSum.from_labels({"ZZI": 1.0}))
    batch = FiniteShotBackend(seed=1).sample_paulis(_program(), [word], 200_000)
    assert batch.mean(word.code) == pytest.approx(exact, abs=0.01)


def test_per_word_shot_map():
    words = [PauliWord.from_label("ZZI"), PauliWord.from_label("XII")]
    shots = {words[0].code: 100, words[1].code: 50}
    batch = FiniteShotBackend(seed=5).sample_paulis(_program(), words, shots)
    assert batch.shots[words[0].code] == 100
    assert batch.shots[words[1].code] == 50


def test_measurement_batches_are_immutable_and_report_hardware_shots():
    word = PauliWord.from_label("ZZI")
    batch = FiniteShotBackend(seed=5).sample_paulis(_program(), [word], 20)
    assert batch.hardware_shots == 20
    with pytest.raises(TypeError):
        batch.shots[word.code] = 1

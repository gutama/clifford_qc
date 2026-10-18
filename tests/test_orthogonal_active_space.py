"""Orthogonal-space invariants and adversarial independent replay checks."""
from copy import deepcopy
from dataclasses import replace
import json
import shutil
import subprocess

import numpy as np
import pytest

from benchmarks import check_orthogonal_active_space as checker
from benchmarks import run_orthogonal_active_space as producer
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.dense_reference import to_matrix
from clifford_qc.ir import Program
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace.orthogonal import occupation_packets, solve_orthogonal_corrections


def test_complete_complex_correction_space_recovers_spectrum():
    rng = np.random.default_rng(71)
    matrix = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    matrix += matrix.conj().T
    frame, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    result = solve_orthogonal_corrections(lambda x: matrix @ x, frame[:, :2], frame[:, 2:])
    np.testing.assert_allclose(result.energies, np.linalg.eigvalsh(matrix), atol=1e-12)
    assert result.residual_norm < 1e-12
    assert result.orthogonality_defect < 1e-12


def test_residual_is_in_full_space_and_actions_are_counted():
    h = np.array([[0., 1., 0.], [1., 2., .7], [0., .7, 4.]])
    calls = []

    def apply(column):
        calls.append(column.copy())
        return h @ column

    result = solve_orthogonal_corrections(apply, np.eye(3)[:, :1], np.eye(3)[:, 1:2])
    assert len(calls) == 2
    assert result.residual_norm > 0.1
    np.testing.assert_allclose(result.projected_hamiltonian @ result.coefficients[:, 0],
                               result.energy * result.coefficients[:, 0], atol=1e-12)
    assert result.residual_norm == pytest.approx(np.linalg.norm(h @ result.state - result.energy * result.state))


def test_orthogonality_is_required_to_whole_classical_span():
    classical = np.eye(4)[:, :2]
    correction = np.array([[0.], [1.], [1.], [0.]]) / np.sqrt(2)
    with pytest.raises(ValueError, match="mutually orthonormal"):
        solve_orthogonal_corrections(lambda x: x, classical, correction)


@pytest.mark.parametrize("classical,corrections", [
    (np.zeros((3, 0)), None), (np.ones(3), None),
    (np.eye(3)[:, :1], np.ones((2, 1))),
    (np.eye(3)[:, :1], np.full((3, 1), np.nan)),
    (np.ones((3, 2)), None),
])
def test_invalid_spaces_are_rejected(classical, corrections):
    with pytest.raises(ValueError):
        solve_orthogonal_corrections(lambda x: x, classical, corrections)


@pytest.mark.parametrize("apply", [lambda x: x[:-1], lambda x: x * np.nan,
                                     lambda x: np.array([1., 0.])])
def test_bad_hamiltonian_actions_are_rejected(apply):
    with pytest.raises(ValueError):
        solve_orthogonal_corrections(apply, np.eye(2))


def test_disjoint_occupation_packets_have_known_identity_overlap():
    words = np.array([0, 1, 2, 3, 4, 5])
    amplitudes = np.array([0, 0, 1j, -2, 3j, 4])
    packets = occupation_packets(words, amplitudes, [0, 1], 6)
    assert [packet.pattern for packet in packets] == [2, 4]
    frame = np.column_stack([packet.vector for packet in packets])
    np.testing.assert_allclose(frame.conj().T @ frame, np.eye(2), atol=1e-14)
    assert np.max(abs(frame[:2])) == 0
    for column in frame.T:
        pivot = column[np.argmax(abs(column))]
        assert abs(pivot.imag) < 1e-14 and pivot.real > 0


@pytest.mark.parametrize("words,amplitudes,excluded,mask", [
    ([0, 1], [1., 1.], [0], 1),
    ([0, 0], [0., 1.], [], 1),
    ([0., 1.], [0., 1.], [], 1),
    ([0, 1], [0., 1.], [2], 1),
    ([0, 1], [0., np.nan], [], 1),
    ([0, 1], [0., 1.], [], -1),
])
def test_packet_inputs_do_not_silently_change_the_space(words, amplitudes, excluded, mask):
    with pytest.raises(ValueError):
        occupation_packets(words, amplitudes, excluded, mask)


@pytest.fixture(scope="module")
def synthetic_case():
    """An undeclared Hubbard fixture; no campaign molecular energy is read."""
    model = hubbard(4, t=1.0, U=3.0)
    reference = Program(8)
    for mode in range(4):
        reference.clifford("X", mode)
    model = replace(model, reference=reference)
    backend = SectorStatevectorBackend(8, 4, 0)
    dense = to_matrix(model.hamiltonian)
    matrix = dense[np.ix_(backend.basis, backend.basis)]
    config = json.loads(producer.CONFIG.read_text())
    case = {"id": "synthetic_hubbard", "path": "undeclared", "active_spatial_orbitals": [1, 2]}
    config["cases"] = [case]
    config["noise"]["replicas"] = 5
    patch = pytest.MonkeyPatch()
    patch.setattr(producer, "fcidump_model", lambda path: model)
    try:
        record = producer.run_case(case, config, 0)
    finally:
        patch.undo()
    return record, case, config, matrix, backend.basis.tolist()


def test_independent_replay_accepts_synthetic_pilot(synthetic_case):
    entry, case, config, matrix, words = synthetic_case
    assert checker.verify_case(entry, case, config, matrix, words) == []
    assert entry["trajectory"][-1]["arms"]["orthogonal"]["energy"] < entry["classical_energy"]


def test_packet_serialization_accepts_only_numerically_small_support_changes():
    row = {"pattern": 2, "indices": [1, 2],
           "coefficients": {"real": [.6, -.8], "imag": [0, 0]}}
    words = [2, 3, 6, 0]
    expected = np.array([2e-11, .6, -.8, 0], complex)
    assert checker.packet_agrees(row, expected, words, [3], 2, 1e-12)
    assert checker.packet_agrees(row, -expected, words, [3], 2, 1e-12)
    expected[0] = 1e-3
    assert not checker.packet_agrees(row, expected, words, [3], 2, 1e-12)
    expected[0] = 0
    assert not checker.packet_agrees(row, expected, words, [1, 3], 2, 1e-12)
    assert not checker.packet_agrees(row, expected, words, [3], 1, 1e-12)


@pytest.mark.parametrize("mutation", [
    "missing_prefix", "wrong_energy", "wrong_pencil", "wrong_packet", "wrong_order",
    "missing_arm", "wrong_work", "wrong_target", "missing_replica", "wrong_noise",
    "wrong_noise_summary", "wrong_classical_space", "optimistic_failure_count",
])
def test_independent_replay_rejects_changed_science(synthetic_case, mutation):
    original, case, config, matrix, words = synthetic_case
    entry = deepcopy(original)
    arm = entry["trajectory"][-1]["arms"]["orthogonal"]
    if mutation == "missing_prefix":
        entry["trajectory"].pop(1)
    elif mutation == "wrong_energy":
        arm["energy"] += 0.01
    elif mutation == "wrong_pencil":
        arm["projected_hamiltonian"]["real"][0][0] += 0.01
    elif mutation == "wrong_packet":
        entry["packet_ranking"][0]["coefficients"]["real"][0] += 0.1
    elif mutation == "wrong_order":
        entry["packet_ranking"].reverse()
    elif mutation == "missing_arm":
        entry["trajectory"][-1]["arms"].pop("same_span_mixed")
    elif mutation == "wrong_work":
        entry["resources"]["selection_h_action_calls"] -= 1
    elif mutation == "wrong_target":
        arm["target_reached"] = not arm["target_reached"]
    elif mutation == "missing_replica":
        entry["noise"]["scales"][1]["replicas"].pop()
    elif mutation == "wrong_noise":
        entry["noise"]["scales"][1]["replicas"][0]["orthogonal_known_overlap"]["error"] += .1
    elif mutation == "wrong_noise_summary":
        entry["noise"]["scales"][1]["summary"]["mixed_noisy_overlap"]["conditional_rmse"] += .1
    elif mutation == "wrong_classical_space":
        entry["classical_indices"].pop()
    elif mutation == "optimistic_failure_count":
        entry["noise"]["scales"][1]["summary"]["mixed_noisy_overlap"]["failures"] += 1
    try:
        problems = checker.verify_case(entry, case, config, matrix, words)
    except (KeyError, IndexError, ValueError):
        # The CLI converts malformed records into explicit verification failure.
        return
    assert problems


def test_noise_uses_centered_energy_and_counts_failures(synthetic_case):
    entry, case, config, matrix, words = synthetic_case
    last = entry["trajectory"][-1]["arms"]["orthogonal"]
    h = checker.decode(last["projected_hamiltonian"])
    c = checker.decode(entry["classical_coefficients"])
    altered = deepcopy(config)
    altered["noise"]["minimum_overlap_eigenvalue"] = 1e6
    probe = producer.noise_probe(h, entry["classical_energy"], c, 4,
                                 entry["exact_reference"]["energy"], altered, 0)
    assert checker.verify_noise(probe, h, entry["classical_energy"], c,
                                entry["exact_reference"]["energy"], altered, 0) == []
    for scale in probe["scales"]:
        summary = scale["summary"]["mixed_noisy_overlap"]
        assert summary["failures"] == altered["noise"]["replicas"]
        assert summary["conditional_rmse"] is None
        assert summary["unconditional_target_fraction"] == 0
    shifted = producer.noise_probe(h + 123 * np.eye(h.shape[0]), entry["classical_energy"] + 123,
                                   c, 4, entry["exact_reference"]["energy"] + 123, config, 0)
    original = entry["noise"]
    for left, right in zip(original["scales"], shifted["scales"]):
        for a, b in zip(left["replicas"], right["replicas"]):
            for name in producer.NOISE_ARMS:
                assert a[name]["error"] == pytest.approx(b[name]["error"], abs=1e-10)


def test_default_is_result_free_and_outputs_are_never_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(producer, "execute", lambda config: pytest.fail("execution reached"))
    assert producer.main([]) == 0
    output = tmp_path / "record.json"
    output.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        producer.main(["--execute", "--output", str(output)])
    assert output.read_text() == "prior evidence"


def test_committed_record_replays_independently():
    if not checker.RECORD.exists():
        pytest.skip("campaign is not executed until implementation is committed")
    record = json.loads(checker.RECORD.read_text())
    assert checker.verify(record) == []
    for mutation in ("case_scope", "evidence_tier", "source_identity"):
        changed = deepcopy(record)
        if mutation == "case_scope":
            changed["cases"].pop()
        elif mutation == "evidence_tier":
            changed["evidence"]["noise"]["tier"] = "finite_sample_empirical"
        else:
            changed["source_sha256"]["clifford_qc/subspace/orthogonal.py"] = "0" * 64
        assert checker.verify(changed)


def test_original_execution_history_restores_without_changing_branch_refs(tmp_path):
    if not checker.RECORD.exists():
        pytest.skip("observations are required to test the execution archive")
    record = json.loads(checker.RECORD.read_text())
    root = tmp_path / "baseline"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    baseline = "69048adaffcd0b9525aeda46f7cab32a9b794df3"
    subprocess.run(["git", "fetch", "--quiet", "--no-tags", str(checker.ROOT), baseline],
                   cwd=root, check=True)
    assert subprocess.run(["git", "cat-file", "-e", record["source_commit"]],
                          cwd=root, capture_output=True).returncode != 0
    target = root / checker.EXECUTION_BUNDLE
    target.parent.mkdir(parents=True)
    shutil.copyfile(checker.ROOT / checker.EXECUTION_BUNDLE, target)
    checker.ensure_execution_history(record, root)
    subprocess.run(["git", "merge-base", "--is-ancestor", record["declaration_commit"],
                    record["source_commit"]], cwd=root, check=True)
    assert subprocess.check_output(["git", "for-each-ref"], cwd=root) == b""
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="bundle digest"):
        checker.ensure_execution_history(record, root)

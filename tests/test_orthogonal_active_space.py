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
           "coefficients": {"real": [-.6, .8], "imag": [0, 0]}}
    words = [2, 3, 6, 0]
    expected = np.array([2e-11, -.6, .8, 0], complex)
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
    require_full_history()
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
    require_full_history()
    record = json.loads(checker.RECORD.read_text())
    root = tmp_path / "baseline"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    baseline = "69048adaffcd0b9525aeda46f7cab32a9b794df3"
    subprocess.run(["git", "fetch", "--quiet", "--no-tags", str(checker.ROOT), baseline],
                   cwd=root, check=True)
    pairs = [(checker.RECORD, checker.EXECUTION_BUNDLE),
             (checker.V2_RECORD, checker.V2_EXECUTION_BUNDLE)]
    for record_path, bundle_path in pairs:
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        assert subprocess.run(["git", "cat-file", "-e", record["source_commit"]],
                              cwd=root, capture_output=True).returncode != 0
        target = root / bundle_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checker.ROOT / bundle_path, target)
        checker.ensure_execution_history(record, root)
        subprocess.run(["git", "merge-base", "--is-ancestor", record["declaration_commit"],
                        record["source_commit"]], cwd=root, check=True)
        assert subprocess.check_output(["git", "for-each-ref"], cwd=root) == b""
        revised = record["schema"].endswith(".v2")
        config, declaration = (checker.V2_CONFIG, checker.V2_DECLARATION) if revised else (checker.CONFIG, checker.DECLARATION)
        for path in [record_path, config, declaration, *[checker.ROOT / p for p in record["source_sha256"]]]:
            packaged = root / path.relative_to(checker.ROOT)
            packaged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, packaged)
        assert checker.source_problems(record, config, declaration, record_path, root=root) == []
        target.write_bytes(target.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match="bundle digest"):
            checker.ensure_execution_history(record, root)


def require_full_history():
    from benchmarks.archive_provenance import history_kind
    try:
        history_kind(checker.ROOT)
    except ValueError as exc:
        pytest.skip(str(exc))


def test_revised_committed_record_replays():
    require_full_history()
    if not checker.V2_RECORD.exists():
        pytest.skip("revised observations follow the implementation commit")
    assert checker.verify(json.loads(checker.V2_RECORD.read_text())) == []


def test_packet_records_enforce_canonical_phase():
    row = {"pattern": 2, "indices": [1, 2],
           "coefficients": {"real": [-.6, .8], "imag": [0., 0.]}}
    expected = np.array([0., -.6, .8, 0.], complex)
    for factor in (-1, 1j):
        bad = deepcopy(row)
        bad["coefficients"] = producer.encode(factor * expected[1:3])
        assert not checker.packet_agrees(bad, expected, [2, 3, 6, 0], [3], 2, 1e-12)


def test_noise_only_packets_are_rejected_before_normalization():
    words = [0, 2, 4, 6]
    amplitudes = [0, 2e-12, 1, 0]
    assert [p.pattern for p in occupation_packets(words, amplitudes, [0], 6)] == [4]
    # The relative threshold also rejects a 1e-6 group beside a 1e4 residual.
    assert [p.pattern for p in occupation_packets(words, [0, 1e-6, 1e4, 0], [0], 6)] == [4]


def test_packet_and_result_arrays_are_owned_and_immutable():
    from clifford_qc.subspace.orthogonal import OccupationPacket
    source = np.array([0., 1.])
    packet = OccupationPacket(1, source)
    other = OccupationPacket(1, source)
    source[1] = 99
    assert packet.vector[1] == 1 and packet != other
    assert isinstance(hash(packet), int)
    result = solve_orthogonal_corrections(lambda x: x, np.eye(2))
    for array in (packet.vector, result.basis, result.actions, result.coefficients,
                  result.energies, result.projected_hamiltonian):
        with pytest.raises(ValueError):
            array.flat[0] = 99
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_measured_overlap_defect_is_not_replaced_by_identity():
    basis = np.eye(2, dtype=complex)
    basis[0, 0] += 2e-11
    action = producer.CountedAction(type("Identity", (), {"matvec": staticmethod(lambda x: x)})())
    observed, _, overlap = producer.readout(basis, action, 1e-10, orthogonal=True)
    assert observed["overlap_identity_defect"] > 3e-11
    np.testing.assert_array_equal(overlap, basis.conj().T @ basis)
    np.testing.assert_array_equal(checker.decode(observed["projected_overlap"]), overlap)


def test_spin_packets_merge_partners_and_remove_triplets():
    from clifford_qc.subspace.spin_packets import singlet_packets, spin_squared_operator
    backend = SectorStatevectorBackend(4, 2, 0)
    s2 = backend.operator(spin_squared_operator(4)).restrict(np.arange(backend.dimension))
    independent = checker.independent_spin_squared(backend.basis.tolist(), 4)
    np.testing.assert_allclose(s2, independent, atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(s2), [0, 0, 0, 2], atol=1e-14)
    # One open-shell determinant carries singlet and triplet components.
    amplitudes = np.zeros(backend.dimension)
    index = backend.basis.tolist().index(9)
    amplitudes[index] = 1
    packets = singlet_packets(backend.basis, amplitudes, [], 15, 4, s2)
    assert len(packets) == 1
    vector = packets[0].vector
    assert np.count_nonzero(abs(vector) > 1e-12) == 2
    assert vector[index] == pytest.approx(-vector[backend.basis.tolist().index(6)])
    assert np.linalg.norm(s2 @ vector) < 1e-14
    values, vectors = np.linalg.eigh(s2)
    triplet = vectors[:, np.argmax(values)]
    assert not singlet_packets(backend.basis, triplet, [], 15, 4, s2)
    assert not singlet_packets(backend.basis, amplitudes * 2e-12, [], 15, 4, s2)


def test_partition_check_rejects_missing_duplicate_and_overlapping_indices():
    assert producer.partition_is_complete([0, 2], [1, 3], 4)
    for classical, external in [([0, 0], [1, 2, 3]), ([0, 2], [1]), ([0, 2], [1, 2, 3])]:
        assert not producer.partition_is_complete(classical, external, 4)


def test_execution_preflight_rejects_untracked_and_wrong_environment_before_science(monkeypatch):
    monkeypatch.setattr(producer, "run_case", lambda *a: pytest.fail("science ran"))
    clean = {"provenance": {"git_sha": "a" * 40, "git_dirty": False}}
    monkeypatch.setattr(producer, "stamp_record", lambda record: clean)
    monkeypatch.setattr(producer, "git", lambda *a: "?? notes.txt" if a[0] == "status" else "a" * 40)
    with pytest.raises(ValueError, match="including untracked"):
        producer.execute(producer.load_design())
    def wrong_environment(record):
        raise ValueError("undeclared numerical environment")
    monkeypatch.setattr(producer, "stamp_record", wrong_environment)
    with pytest.raises(ValueError, match="undeclared numerical environment"):
        producer.execute(producer.load_design())


@pytest.mark.parametrize("path_kind", ["parent", "symlink"])
def test_input_paths_cannot_escape_repository(monkeypatch, tmp_path, path_kind):
    config = json.loads(producer.CONFIG.read_text())
    if path_kind == "parent":
        config["cases"][0]["path"] = "../outside.FCIDUMP"
    else:
        link = tmp_path / "escape.FCIDUMP"
        outside = tmp_path.parent / f"{tmp_path.name}-outside.FCIDUMP"
        outside.write_text("outside")
        link.symlink_to(outside)
        monkeypatch.setattr(producer, "ROOT", tmp_path)
        config["cases"][0]["path"] = "escape.FCIDUMP"
    config_path = tmp_path / "design.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="escape"):
        producer.load_design(config_path)


def test_preserved_observation_binding_rejects_metadata_changes():
    require_full_history()
    record = json.loads(checker.RECORD.read_text())
    for field in ("timing", "storage", "provenance"):
        changed = deepcopy(record)
        if field == "timing":
            changed["cases"][0]["resources"]["wall_seconds_including_reference_and_noise"] += 1
        elif field == "storage":
            changed["cases"][0]["resources"]["operator_storage"]["test_fake_payload"] = 10
        else:
            changed["provenance"]["utc"] = "changed after observation"
        assert "inspected record differs from preserved observation bytes" in checker.source_problems(
            changed, checker.CONFIG, checker.DECLARATION, checker.RECORD)


def test_declaration_absence_check_rejects_code_present_at_declaration(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.invalid")
    for path in (producer.CONFIG, producer.DECLARATION):
        target = tmp_path / path.relative_to(producer.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    git("add", ".")
    git("commit", "-qm", "declaration only")
    declaration = git("rev-parse", "HEAD")
    implementation = tmp_path / "clifford_qc/subspace/spin_packets.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# revised implementation")
    git("add", ".")
    git("commit", "-qm", "implementation")
    source = git("rev-parse", "HEAD")
    producer.validate_declaration(declaration, source, producer.load_design(), root=tmp_path)
    git("commit", "--allow-empty", "-qm", "later commit")
    with pytest.raises(ValueError, match="already exists"):
        producer.validate_declaration(source, git("rev-parse", "HEAD"), producer.load_design(), root=tmp_path)


def test_shallow_history_refuses_before_bundle_operations(tmp_path):
    from benchmarks.archive_provenance import ShallowHistoryError
    source = tmp_path / "source"
    source.mkdir()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.invalid")
    git("commit", "--allow-empty", "-qm", "first")
    git("commit", "--allow-empty", "-qm", "second")
    clone = tmp_path / "shallow"
    git("clone", "--quiet", "--depth", "1", source.as_uri(), str(clone))
    with pytest.raises(ShallowHistoryError, match="full git history"):
        checker.ensure_execution_history({}, clone)
    with pytest.raises(ValueError, match="full Git history"):
        checker.ensure_execution_history({}, tmp_path / "archive")


def test_report_conclusions_follow_records_and_include_every_violation():
    from benchmarks.summarize_orthogonal_active_space import render
    record = json.loads(checker.RECORD.read_text())
    text = render(record)
    assert "-1.862533" in text
    assert "34 / 64" in text and "30 / 64" in text and "25 / 64" in text
    changed = deepcopy(record)
    changed["cases"] = changed["cases"][:1]
    changed["cases"][0]["trajectory"][-1]["arms"]["orthogonal"]["error"] = 1
    assert "0/1 inputs" in render(changed)
    changed["cases"][0]["trajectory"][-1]["arms"]["orthogonal"]["energy"] += 1
    assert "monotonicity holds on 0/1" in render(changed)


def test_report_folds_machine_dependent_rounding():
    from benchmarks.summarize_orthogonal_active_space import diagnostic
    assert diagnostic(-5.15079e-19, 6) == diagnostic(2.22e-16, 3) == "< 1e-12"
    assert diagnostic(0.00732457, 6) == "0.00732457"

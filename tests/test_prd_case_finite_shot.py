import copy
from pathlib import Path

import pytest

from benchmarks import run_prd_case_finite_shot as finite
from benchmarks import run_prd_case_paper_suite as suite
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.pauli import X
from clifford_qc.states import ket_density
from clifford_qc.subspace import Generator, MatrixElementBank, identity_generator


CONFIG = Path("benchmarks/configs/prd_case_paper_suite_v3.json")


def test_v3_freezes_the_complete_180_cell_protocol():
    manifest = suite.load_manifest(CONFIG)
    frozen = finite.load_frozen_bases(CONFIG, manifest)
    tasks = finite.build_tasks(manifest)

    assert manifest["schema"] == suite.FINITE_SHOT_SCHEMA
    assert len(tasks) == 3 * 3 * 20 == 180
    assert manifest["finite_shot_tier"]["M"] == 7
    assert manifest["finite_shot_tier"]["packet_K"] == 16
    assert set(frozen["systems"]) == {
        "h4_stretched", "hubbard_2x2_u4", "h2o_qsci_stretched"}
    assert all(len(payload["packets"]) == 6
               for payload in frozen["systems"].values())


def test_frozen_basis_digest_rejects_post_preregistration_edits(tmp_path):
    manifest = suite.load_manifest(CONFIG)
    frozen = finite.load_frozen_bases(CONFIG, manifest)
    tampered = copy.deepcopy(frozen)
    tampered["systems"]["h4_stretched"]["packets"][0][
        "real_coefficients"][0] += 1e-6
    basis_path = tmp_path / "bases.json"
    basis_path.write_text(__import__("json").dumps(tampered), encoding="utf-8")
    overlay = __import__("json").loads(CONFIG.read_text(encoding="utf-8"))
    overlay["basis_file"] = basis_path.name

    # Point a minimal config directory at the tampered basis while preserving
    # the already-resolved manifest; the loader must fail on the payload hash.
    altered = copy.deepcopy(manifest)
    altered["finite_shot_tier"]["basis_file"] = basis_path.name
    with pytest.raises(ValueError, match="digest is invalid"):
        finite.load_frozen_bases(tmp_path / "manifest.json", altered)


def test_static_qwc_plan_spends_the_declared_total_and_is_outcome_independent():
    manifest = suite.load_manifest(CONFIG)
    n = 2
    rho = ket_density(n, "00")
    bank = MatrixElementBank(
        rho, -1.0 * X(n, 0),
        [identity_generator(n), Generator("x0", X(n, 0))])
    session = SharedMeasurement(bank)

    left, left_ledger = finite._static_plan(
        session, manifest["finite_shot_tier"], 100)
    right, right_ledger = finite._static_plan(
        session, manifest["finite_shot_tier"], 100)

    assert finite._plan_shots(session.groups, left) == 100
    assert left == right
    assert left_ledger["plan_sha256"] == right_ledger["plan_sha256"]
    assert left_ledger["overlap_budget"] == 25
    assert left_ledger["hamiltonian_budget"] == 75


def test_reseed_reuses_deterministic_group_plans_but_restarts_the_rng():
    rho = ket_density(1, "0")
    groups = [[PauliWord(1, 1)]]  # X on |0>: genuinely stochastic outcomes.
    backend = FiniteShotBackend(seed=17)

    assert backend.prepare_grouped_state(rho, groups) == 1
    first = backend.sample_grouped_from_state(rho, groups, 37)
    backend.reseed(17)
    second = backend.sample_grouped_from_state(rho, groups, 37)

    assert first.groups[0].hist == second.groups[0].hist
    assert backend.prepare_grouped_state(rho, groups) == 1

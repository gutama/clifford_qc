"""Q18-S2 correctness on the undeclared Hubbard dimer; no declared-prefix result here."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from benchmarks import check_phase15_certifying_screen as checker
from benchmarks import check_phase15_certifying_screen_declaration as declaration
from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import check_phase15_preregistration as preflight
from benchmarks import check_phase15_residual_successor_declaration as q18s1_declaration
from benchmarks import phase15_certifying_screen as screen
from benchmarks import run_phase15_certifying_screen as producer
from benchmarks import run_phase15_residual_successor as q18s1
from benchmarks.phase15_residual_successor import derivative_check as q18s1_derivative_check
from benchmarks.run_mapping_axis import _raw_pool
from benchmarks.run_phase15_h2_preflight import _digest
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import SecondMomentBank
from clifford_qc.subspace.generators import identity_generator

LINEAGE_SIZE = 2


def _q18s1_entry(model, selected):
    """Q18-S1's own producer path on a toy bank: the lineage target."""
    config = copy.deepcopy(q18s1_declaration.load_config())
    bank = preflight.first_moment_bank(model, selected)
    second = SecondMomentBank(bank)
    root = second.residual(bank.solve(), 0)
    sh, combined = set(bank.word_set()), set(bank.word_set()) | set(second.word_set())
    committed = {
        "validation_energy": root.energy, "variance": root.variance,
        "cancellation_scale": root.cancellation_scale, "variance_resolved": root.resolved,
        "preflight_sh_word_universe": len(sh), "ledger_sh_word_universe": len(sh),
        "raw_combined_word_universe": len(combined), "raw_combined_words_sha256": _digest(combined),
        "mapping_protocol": "qwc_groups",
        "mapping_settings": len(q18.group_words(sh, "qwc_groups", model.n)),
        "mapping_energy_variance": None,
    }
    frozen = q18.quantities_for(model, selected, committed, "qwc_groups", "qwc_basis_cover")
    return q18s1.evaluate_bank(config, "hubbard_2x2", model, selected, frozen,
                               root.as_dict())


@pytest.fixture(scope="module")
def toy():
    model = hubbard(2)
    generators = [identity_generator(model.n), *_raw_pool(model)]
    bank = preflight.first_moment_bank(model, generators)
    history = [float(bank.solve(list(range(m))).ground_energy)
               for m in range(1, len(generators) + 1)]
    config = copy.deepcopy(declaration.load_config())
    config["trajectories"]["rows"] = {"toy": {
        "level4": False, "candidate_family": "levels 0-3", "stopped_reason": "toy",
        "labels": [g.label for g in generators], "energy_history": history,
        "prefixes": list(range(LINEAGE_SIZE, len(generators) + 1))}}
    spectrum = screen.sector_spectrum(model)
    premises = screen.sector_premises(model, [generators])
    config["measured_before_freezing"] = {
        "spectrum": spectrum, "premises": premises,
        "trajectories": {"toy": {"pool_size": len(generators)}}}
    inputs = {"toy": (generators, len(generators))}
    predecessor = _q18s1_entry(model, generators[:LINEAGE_SIZE])
    raw = json.dumps(config).encode()
    return {"model": model, "generators": generators, "config": config, "raw": raw,
            "inputs": inputs, "predecessor": predecessor, "spectrum": spectrum}


@pytest.fixture()
def toy_record(toy, monkeypatch):
    monkeypatch.setattr(declaration, "FROZEN_BANK_SIZE", LINEAGE_SIZE)
    record = producer.run_screen(toy["config"], config_bytes=toy["raw"], model=toy["model"],
                                 inputs=toy["inputs"], predecessor=toy["predecessor"])
    record["provenance"] = {key: None for key in checker.PROVENANCE_KEYS}
    record["provenance"].update(git_sha="0" * 40, git_dirty=False)
    return record


def _rederive(toy, record):
    return checker.rederivation_problems(toy["config"], record, toy["raw"], toy["predecessor"])


# ------------------------------------------------------------------ the toy run

def test_toy_record_rederives_and_rebuilds(toy, toy_record):
    assert _rederive(toy, toy_record) == []
    assert checker.recompute_problems(toy["config"], toy_record, model=toy["model"],
                                      inputs=toy["inputs"],
                                      predecessor=toy["predecessor"]) == []
    assert toy_record["lineage"]["passes"]
    json.dumps(toy_record, allow_nan=False)


def test_toy_domain_status_and_verdict(toy_record):
    prefixes = toy_record["trajectories"]["toy"]["prefixes"]
    # M = 2 is affordable but its interval holds E1; M = 3 certifies and is not.
    assert not prefixes["2"]["certifying"] and prefixes["2"]["role"] == "diagnostic"
    assert prefixes["3"]["certifying"] and prefixes["3"]["role"] == "decision"
    assert prefixes["4"]["status"] == "UNRESOLVED" and not prefixes["4"]["in_domain"]
    assert toy_record["decision"]["domain"] == ["toy/3"]
    for entry in (prefixes["2"], prefixes["3"]):
        assert all(entry["deterministic_checks"].values())
    expected = screen.verdict_of(prefixes.values())
    assert toy_record["decision"]["verdict"] == expected


def test_lineage_compares_against_q18_s1s_own_path(toy, toy_record):
    entry = toy_record["trajectories"]["toy"]["prefixes"][str(LINEAGE_SIZE)]
    assert all(producer.lineage_entry(entry, toy["predecessor"]).values())
    forged = copy.deepcopy(toy["predecessor"])
    forged["protocols"]["qwc_groups"]["residual_settings"] += 1
    assert not all(producer.lineage_entry(entry, forged).values())
    forged = copy.deepcopy(toy["predecessor"])
    forged["finite_differences"]["functional"]["evaluations"][0]["plus"] += 1e-6
    assert not all(producer.lineage_entry(entry, forged).values())


# ------------------------------------------------------------------ the pencil

def _toy_rows(toy, size):
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.subspace.elements import MatrixElementBank

    model = toy["model"]
    rho = ExactMVBackend().state(model.reference, ())
    bank = MatrixElementBank(rho, model.hamiltonian, toy["generators"][:size])
    second = SecondMomentBank(bank)
    rows = q18.block_rows(bank, second, list(range(size)))
    return bank, second, rows, q18.row_words(rows)


def test_compiled_pencil_is_q18s_linear_map(toy):
    for size in (2, 3, 4):
        bank, _, rows, words = _toy_rows(toy, size)
        means = q18.reference_means(bank.reference, words)
        rng = np.random.default_rng(size)
        perturbed = {w: means[w] + 1e-2 * rng.standard_normal() for w in words}
        compiled = screen.CompiledRows(rows, size, words)
        for a, b in zip(q18.projected_matrices(rows, size, perturbed),
                        compiled.matrices(compiled.vector(perturbed))):
            assert np.max(np.abs(a - b)) <= 1e-14 * max(1.0, np.max(np.abs(a)))


def test_compiled_derivative_matches_q18_s1s_validator(toy):
    bank, second, rows, words = _toy_rows(toy, 3)
    means = q18.reference_means(bank.reference, words)
    weights, _ = q18.residual_functional(rows, 3, means)
    weights = {int(w): float(v) for w, v in weights.items()}
    scale = second.residual(bank.solve(), 0).cancellation_scale
    rule = toy["config"]["derivative_validation"]
    directions = q18.check_directions(weights, set(bank.word_set()), seed=rule["seed"])
    compiled = screen.CompiledRows(rows, 3, words)
    for key in rule["directions"]:
        mine = screen.derivative_check(compiled, means, weights, directions[key], rule=rule,
                                       cancellation_scale=scale)
        theirs = q18s1_derivative_check(rows, 3, means, weights, directions[key], rule=rule,
                                        cancellation_scale=scale)
        assert mine["passes"] and theirs["passes"]
        for a, b in zip(mine["evaluations"], theirs["evaluations"]):
            assert abs(a["plus"] - b["plus"]) <= 1e-12 * scale
            assert abs(a["minus"] - b["minus"]) <= 1e-12 * scale


# ------------------------------------------------------------------ the rule

def test_certifying_predicate_boundaries():
    ground, first = -1.0, 0.0
    assert screen.certifying(-0.9, 0.5, ground, first)
    assert not screen.certifying(-0.5, 0.5, ground, first)  # E + sigma == E1
    assert not screen.certifying(-0.2, 0.5, ground, first)
    assert not screen.certifying(-1.0 - 1e-6, 0.1, ground, first)  # below E0
    assert screen.certifying(-1.0 - 1e-15, 0.1, ground, first)  # rounding
    assert screen.temple_lower_bound(0.1, 0.5, first) is None
    assert screen.temple_lower_bound(-0.5, 0.25, first) == pytest.approx(-1.0)


@pytest.mark.parametrize("entries, verdict", [
    ([], "UNREACHED"),
    ([("PROHIBITIVE", False), ("AFFORDABLE", False)], "UNREACHED"),
    ([("AFFORDABLE", True), ("PROHIBITIVE", True)], "OPEN"),
    ([("PROHIBITIVE", True), ("PROHIBITIVE", True), ("AFFORDABLE", False)], "CLOSED"),
    ([("PROHIBITIVE", True), ("GROUPING_SENSITIVE", True)], "GROUPING_SENSITIVE"),
    ([("AFFORDABLE", True), ("INVALID", True)], "INVALID"),
    ([("INVALID", False), ("PROHIBITIVE", True)], "CLOSED"),
    ([("UNRESOLVED", False)], "UNREACHED"),
])
def test_verdict_truth_table(entries, verdict):
    rows = [{"status": status, "in_domain": domain} for status, domain in entries]
    assert screen.verdict_of(rows) == verdict


def test_a_failed_check_is_invalid_whatever_the_ratio():
    checks = {"full_rank": True, "linearization_validated": False}
    assert screen.status_of({"a": 1.0, "b": 1.0}, "a", "b", 10, checks) == "INVALID"
    assert screen.status_of({"a": 1.0, "b": 11.0}, "a", "b", 10,
                            {"full_rank": True}) == "GROUPING_SENSITIVE"


@pytest.mark.parametrize("path, value", [
    (("decision", "verdict"), "OPEN"),
    (("trajectories", "toy", "prefixes", "3", "certifying"), False),
    (("trajectories", "toy", "prefixes", "3", "status"), "AFFORDABLE"),
    (("trajectories", "toy", "prefixes", "2", "role"), "decision"),
    (("trajectories", "toy", "prefixes", "3", "protocols", "qwc_groups", "neyman_ratio"), 1.0),
    (("trajectories", "toy", "prefixes", "3", "deterministic_checks", "partitions_valid"),
     False),
    (("trajectories", "toy", "prefixes", "4", "status"), "PROHIBITIVE"),
    (("lineage", "passes"), False),
    (("spectral_reference", "first_excited_energy"), 1.0),
    (("quantum_advantage_claim",), True),
])
def test_each_forged_field_is_caught(toy, toy_record, path, value):
    forged = copy.deepcopy(toy_record)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert _rederive(toy, forged)


def test_a_self_consistent_variance_forgery_needs_and_fails_the_rebuild(toy, toy_record):
    forged = copy.deepcopy(toy_record)
    price = forged["trajectories"]["toy"]["prefixes"]["3"]["protocols"]["qwc_groups"]
    price["residual_setting_variances"] = [v / 100.0 for v in price["residual_setting_variances"]]
    price["residual_variance_one_shot"] = math.fsum(
        max(v, 0.0) for v in price["residual_setting_variances"])
    price["residual_neyman_sum"] = math.fsum(
        math.sqrt(max(v, 0.0)) for v in price["residual_setting_variances"])
    variance = forged["trajectories"]["toy"]["prefixes"]["3"]["estimator"]["functional_variance"]
    price["cost_ratio"] = checker.q18.derive_ratio(price, variance)
    price["neyman_ratio"] = checker.q18.derive_neyman(price, variance)
    price["allocation_diagnostic"] = checker.allocation_totals(
        price, variance, toy["config"]["allocation_diagnostic"])
    assert checker.recompute_problems(toy["config"], forged, model=toy["model"],
                                      inputs=toy["inputs"], predecessor=toy["predecessor"])


# ------------------------------------------------------------------ the declaration

def test_declaration_is_valid_and_result_free():
    config = declaration.load_config()
    assert declaration.static_problems(config) == []
    assert config["contains_results"] is False
    assert config["evidence"]["system_selection"] == "post_hoc"
    assert config["measured_before_freezing"]["trajectories"]["acase_exact_m25"][
        "prefixes_below_midpoint"] == []


@pytest.mark.parametrize("path, value", [
    (("statistic", "max_ratio"), 20),
    (("statistic", "name"), "uniform_ratio"),
    (("derivative_validation", "steps"), [0.001, 0.0005, 0.00025, 0.000125]),
    (("allocation_diagnostic", "pilot_shots_per_setting"), 16),
    (("grouping", "alternative_protocol"), None),
    (("domain", "rounding_ulps"), 1e6),
    (("decision_rule", "verdict_ladder"), ["OPEN", "CLOSED"]),
    (("evidence", "system_selection"), "preregistered"),
    (("question_id",), "Q18-S3"),
    (("claim_boundary",), "This screen was run once and has been read."),
])
def test_declaration_rejects_drift(path, value):
    config = copy.deepcopy(declaration.load_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert declaration.static_problems(config)


def test_declaration_rejects_a_changed_trajectory():
    config = copy.deepcopy(declaration.load_config())
    rows = config["trajectories"]["rows"]
    rows["acase_level4"]["labels"][12], rows["acase_level4"]["labels"][13] = (
        rows["acase_level4"]["labels"][13], rows["acase_level4"]["labels"][12])
    assert declaration.static_problems(config)
    config = copy.deepcopy(declaration.load_config())
    config["trajectories"]["rows"]["acase_exact_m25"]["prefixes"] = list(range(10, 24))
    assert declaration.static_problems(config)


def test_declaration_loader_rejects_result_fields(tmp_path):
    config = declaration.load_config()
    config["trajectories"]["rows"]["acase_level4"]["neyman_ratio"] = 3.0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="result-shaped"):
        declaration.load_config(path)


def test_premises_reject_a_leaky_image_or_storage_overflow():
    quantities = {"spectrum": {"ground_degeneracy": 1},
                  "premises": {"reference_image_sector_leakage": 0.0,
                               "hamiltonian_sector_leakage": 0.0,
                               "spin_parity_sector_violations": 0,
                               "coefficient_occurrence_bound": 1},
                  "trajectories": {}}
    assert declaration.premise_problems(quantities) == []
    for key, value in (("reference_image_sector_leakage", 1e-6),
                       ("spin_parity_sector_violations", 1),
                       ("coefficient_occurrence_bound", 10 ** 12)):
        broken = copy.deepcopy(quantities)
        broken["premises"][key] = value
        assert declaration.premise_problems(broken)


def test_producer_refuses_another_path_and_a_second_run(tmp_path, monkeypatch):
    class Args:
        out = tmp_path / "elsewhere.json"

    assert producer.refusals(Args()) == ["Q18-S2 writes only its declared record path"]
    Args.out = declaration.RECORD
    monkeypatch.setattr(type(declaration.RECORD), "exists", lambda self: True)
    assert producer.refusals(Args()) == ["Q18-S2 runs once; use its checker to rebuild"]


def test_committed_screen_is_unreached_and_rederives():
    config = declaration.load_config()
    record = json.loads(declaration.RECORD.read_text())
    predecessor = json.loads(producer.PREDECESSOR.read_text())["banks"][declaration.SYSTEM]
    assert checker.rederivation_problems(config, record, declaration.CONFIG.read_bytes(),
                                         predecessor) == []
    assert record["decision"]["verdict"] == "UNREACHED"
    assert record["decision"]["domain"] == []
    assert record["lineage"]["passes"]
    entries = [entry for block in record["trajectories"].values()
               for entry in block["prefixes"].values()]
    assert not any(entry["certifying"] for entry in entries)
    assert [e["status"] for e in entries].count("UNRESOLVED") == 1
    for entry in entries:
        assert all(entry["deterministic_checks"].values())
        for price in entry["protocols"].values():
            assert price["allocation_diagnostic"]["estimator_licensed"] is False


# ------------------------------------------------------------------ checker hardening

def _unresolved(record):
    return record["trajectories"]["toy"]["prefixes"]["4"]


@pytest.mark.parametrize("key, value", [
    ("finite_differences", {"functional": {"passes": True}}),
    ("estimator", {"functional_variance": 0.0}),
    ("universes", {}),
])
def test_an_unresolved_prefix_cannot_carry_priced_fields(toy, toy_record, key, value):
    forged = copy.deepcopy(toy_record)
    _unresolved(forged)[key] = value
    assert _rederive(toy, forged)


@pytest.mark.parametrize("where", ["top", "block", "evaluation", "price"])
def test_every_key_set_is_closed(toy, toy_record, where):
    forged = copy.deepcopy(toy_record)
    certifying = forged["trajectories"]["toy"]["prefixes"]["3"]
    target = {"top": forged, "block": forged["trajectories"]["toy"],
              "evaluation": certifying["finite_differences"]["functional"]["evaluations"][0],
              "price": certifying["protocols"]["qwc_groups"]}[where]
    target["summary"] = {"verdict": "OPEN"}
    assert _rederive(toy, forged)


@pytest.mark.parametrize("key, value", [("variance", -1e-6), ("ground_energy", -100.0)])
def test_the_checker_sanity_rules_fail_a_broken_unresolved_prefix(toy, toy_record, key, value):
    forged = copy.deepcopy(toy_record)
    _unresolved(forged)[key] = value
    assert any("checker sanity rule" in p for p in _rederive(toy, forged))


def test_a_priced_prefix_below_e0_is_left_to_its_declared_check(toy, toy_record):
    entry = copy.deepcopy(toy_record["trajectories"]["toy"]["prefixes"]["3"])
    spectrum = toy_record["spectral_reference"]
    entry["ground_energy"] = spectrum["ground_energy"] - 1.0
    spec = toy["config"]["trajectories"]["rows"]["toy"]
    problems, _, _ = checker.prefix_problems(toy["config"], "toy", 3, entry, spectrum,
                                             spec["energy_history"])
    assert not any("checker sanity rule" in p for p in problems)
    assert any("deterministic checks do not derive" in p for p in problems)


@pytest.mark.parametrize("key, value", [("role", "alternative"),
                                        ("energy_partitioned_words", 1)])
def test_protocol_role_and_partition_size_are_restated(toy, toy_record, key, value):
    forged = copy.deepcopy(toy_record)
    forged["trajectories"]["toy"]["prefixes"]["3"]["protocols"]["qwc_groups"][key] = value
    assert _rederive(toy, forged)


def test_the_checker_restates_lineage_without_the_producer(toy, toy_record):
    entry = toy_record["trajectories"]["toy"]["prefixes"][str(LINEAGE_SIZE)]
    rule = toy["config"]["lineage_check"]
    assert checker.restate_lineage(entry, toy["predecessor"], rule) == producer.lineage_entry(
        entry, toy["predecessor"])
    forged = copy.deepcopy(toy["predecessor"])
    forged["protocols"]["qwc_basis_cover"]["neyman_ratio"] *= 1.01
    assert not checker.restate_lineage(entry, forged, rule)["qwc_basis_cover_prices"]


def test_a_negative_group_variance_is_derived_as_invalid_not_rejected(toy, toy_record):
    entry = copy.deepcopy(toy_record["trajectories"]["toy"]["prefixes"]["3"])
    price = entry["protocols"]["qwc_basis_cover"]
    price.update(energy_setting_variances=[], residual_setting_variances=[],
                 energy_variance_one_shot=0.0, residual_variance_one_shot=0.0,
                 energy_neyman_sum=0.0, residual_neyman_sum=0.0, cost_ratio=None,
                 neyman_ratio=None, variances_nonnegative=False, allocation_diagnostic=None)
    entry["deterministic_checks"]["group_variances_nonnegative"] = False
    entry["status"] = "INVALID"
    spec = toy["config"]["trajectories"]["rows"]["toy"]
    problems, status, certifying = checker.prefix_problems(
        toy["config"], "toy", 3, entry, toy_record["spectral_reference"], spec["energy_history"])
    assert problems == [] and status == "INVALID" and certifying


def test_rebuild_tolerates_the_sign_of_a_rounding_level_variance(toy, toy_record):
    forged = copy.deepcopy(toy_record)
    entry = _unresolved(forged)
    entry["variance"] = abs(entry["variance"]) + 4e-14
    entry["residual_norm"] = math.sqrt(entry["variance"])
    entry["weinstein_upper"] = entry["ground_energy"] + entry["residual_norm"]
    first = forged["spectral_reference"]["first_excited_energy"]
    entry["temple_lower_bound"] = entry["ground_energy"] - entry["variance"] / (
        first - entry["ground_energy"])
    assert checker.recompute_problems(toy["config"], forged, model=toy["model"],
                                      inputs=toy["inputs"], predecessor=toy["predecessor"]) == []
    entry["residual_norm"] = 1e-3
    assert checker.recompute_problems(toy["config"], forged, model=toy["model"],
                                      inputs=toy["inputs"], predecessor=toy["predecessor"])


@pytest.mark.parametrize("execution, expected", [
    ("declared", []),
    ("between", []),
    ("before", ["the record was executed at a commit that does not contain the "
                "declaration's last change"]),
    ("record", ["the record's execution commit must strictly precede its first commit"]),
    (None, ["the record names no execution commit"]),
])
def test_execution_commit_sits_between_declaration_and_record(execution, expected):
    order = ["before", "declared", "between", "record"]
    sha = {name: f"{index + 1:x}" * 40 for index, name in enumerate(order)}
    rank = {value: order.index(name) for name, value in sha.items()}

    def is_ancestor(older, newer):
        return rank[older] <= rank[newer]

    assert declaration.execution_order_problems(
        sha["declared"], sha.get(execution, execution), sha["record"],
        is_ancestor=is_ancestor) == expected


@pytest.mark.parametrize("where, value", [
    ("provenance", "OPEN"), ("elapsed_seconds", {"verdict": "OPEN"}),
    ("seconds", {"status": "AFFORDABLE"}), ("elapsed_seconds", True)])
def test_metadata_slots_carry_no_passengers(toy, toy_record, where, value):
    forged = copy.deepcopy(toy_record)
    if where == "provenance":
        forged["provenance"]["verdict"] = value
    elif where == "seconds":
        forged["trajectories"]["toy"]["prefixes"]["2"]["seconds"] = value
    else:
        forged[where] = value
    assert _rederive(toy, forged)


def test_a_duplicated_key_or_non_finite_constant_is_refused(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"decision": {"verdict": "OPEN", "verdict": "UNREACHED"}}')
    with pytest.raises(ValueError, match="duplicated"):
        checker.load_record(path)
    path.write_text('{"elapsed_seconds": NaN}')
    with pytest.raises(ValueError, match="non-finite"):
        checker.load_record(path)


def test_booleans_do_not_pass_for_integers_or_back(toy, toy_record):
    forged = copy.deepcopy(toy_record)
    forged["lineage"]["toy"]["ground_energy"] = 1
    assert _rederive(toy, forged)


def test_rederivation_restates_lineage_rather_than_calling_the_producer(
        toy, toy_record, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the checker must not reuse the producer's lineage")

    monkeypatch.setattr(producer, "lineage_entry", refuse)
    assert _rederive(toy, toy_record) == []


@pytest.mark.parametrize("execution", ["record"[:4], "HEAD", ":/Record Q18-S2"])
def test_an_abbreviated_or_symbolic_execution_commit_is_refused(execution):
    assert declaration.execution_order_problems(
        "a" * 40, execution, "b" * 40, is_ancestor=lambda older, newer: True) == [
        "the record's execution commit is not a full commit SHA"]


def test_post_record_krylov_basis_certifies_outside_the_screen():
    """PLAN's post-record check: the ladder's fixed Krylov basis certifies.

    {H^k|psi>, k <= 8} on the hubbard_2x2 rung, solved at the package's default
    thresholds. Outside the Q18-S2 screen; it is why the ledger scopes UNREACHED
    to the screened trajectories.
    """
    from clifford_qc.subspace.linalg import solve_projected

    model = declaration.system_model()
    spectrum = screen.sector_spectrum(model)
    from benchmarks.run_phase15_h2_preflight import reference_vector
    from clifford_qc.pauli_action import PauliLinearOperator

    hamiltonian = PauliLinearOperator(model.hamiltonian.to_mv())
    vectors = [reference_vector(model)]
    for _ in range(8):
        vectors.append(hamiltonian.matvec(vectors[-1]))
    basis = np.column_stack(vectors)
    images = np.column_stack([hamiltonian.matvec(v) for v in vectors])
    overlap = basis.conj().T @ basis
    projected = basis.conj().T @ images
    result = solve_projected(0.5 * (overlap + overlap.conj().T),
                             0.5 * (projected + projected.conj().T))
    state = basis @ result.coefficients[:, 0]
    state = state / np.linalg.norm(state)
    energy = float(np.vdot(state, hamiltonian.matvec(state)).real)
    sigma = float(np.linalg.norm(hamiltonian.matvec(state) - energy * state))
    assert energy == pytest.approx(-10.100105016587381, abs=1e-8)
    assert sigma == pytest.approx(0.0856, abs=1e-4)
    assert energy + sigma - spectrum["first_excited_energy"] == pytest.approx(-0.2081, abs=1e-4)

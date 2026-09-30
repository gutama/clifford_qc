"""The Phase 15 variance-extrapolation declaration, held to its gate.

Nothing here computes a prefix variance on a declared bank. The fit is checked
on synthetic trajectories, including the two-level contamination picture its
rationale rests on, and on a real trajectory on the undeclared Hubbard dimer.
Each clause of the gate must reject a deliberate mutation of the config.
"""

from __future__ import annotations

import copy
import json
import math

import pytest

import benchmarks.check_phase15_extrapolation_preregistration as gate
from benchmarks.run_mapping_axis import _raw_pool
from clifford_qc.backends import ExactMVBackend
from clifford_qc.ir import PauliWord
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import MatrixElementBank, SecondMomentBank
from clifford_qc.subspace.generator_core import Generator
from clifford_qc.subspace.generators import identity_generator


@pytest.fixture(scope="module")
def config():
    return gate.load_config()


@pytest.fixture(scope="module")
def recomputed(config):
    computed: dict = {}
    problems = gate.structural_problems(config, [], computed)
    return problems, computed


def test_the_committed_declaration_passes_every_static_clause(config):
    assert gate.static_problems(config) == []


def test_the_committed_numbers_recompute(recomputed, config):
    problems, computed = recomputed
    assert problems == []
    assert sorted(n for n, row in computed.items() if row["in_domain"]) == sorted(
        config["required_banks"])


def test_a_result_field_is_refused_at_load(tmp_path, config):
    broken = copy.deepcopy(config)
    broken["measured_before_freezing"]["h4"]["extrapolated_energy"] = -2.18
    path = tmp_path / "config.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="result field"):
        gate.load_config(path)


# ------------------------------------------------------------------ the rule

def test_a_straight_line_recovers_its_intercept():
    fit = gate.extrapolate([0.3, 0.2, 0.1], [1.3, 1.2, 1.1], resolved=[True] * 3)
    assert fit["extrapolable"]
    assert fit["intercept"] == pytest.approx(1.0)
    assert fit["slope"] == pytest.approx(1.0)
    assert fit["residual_rms"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("variances, energies, resolved, reason", [
    ([0.3, 0.2, 0.1], [1.1, 1.2, 1.3], [True] * 3, "slope"),
    ([0.2, 0.2, 0.2], [1.3, 1.2, 1.1], [True] * 3, "only by rounding"),
    ([0.3, 0.2, 0.1], [1.3, 1.2, 1.1], [True, True, False], "not resolved"),
])
def test_the_fit_refuses_what_the_model_cannot_explain(variances, energies, resolved,
                                                       reason):
    fit = gate.extrapolate(variances, energies, resolved=resolved)
    assert not fit["extrapolable"]
    assert reason in fit["reason"]


def test_variances_that_differ_only_by_rounding_do_not_extrapolate():
    """Revision 1: the undeclared LiH toy's Brillouin case, where singles leave the
    Hartree-Fock Ritz state unchanged and the variances differ at 1e-13."""
    variances = [0.0013969672737914607, 0.0013969672735782979, 0.001396967273365135]
    energies = [-7.862026959394136, -7.862026959394136, -7.862026959394137]
    fit = gate.extrapolate(variances, energies, resolved=[True] * 3, scales=[61.8] * 3)
    assert not fit["extrapolable"] and "only by rounding" in fit["reason"]
    unguarded = gate.extrapolate(variances, energies, resolved=[True] * 3)
    assert "intercept" in unguarded  # without the floor the noise would be fitted


def test_the_rationale_holds_on_a_two_level_contamination():
    """E = E0 + w gap and sigma^2 = w(1 - w) gap^2: the window lands near E0."""
    ground, gap = -1.0, 0.3
    weights = [0.04, 0.02, 0.01]
    energies = [ground + w * gap for w in weights]
    variances = [w * (1 - w) * gap ** 2 for w in weights]
    fit = gate.extrapolate(variances, energies, resolved=[True] * 3)
    final_error = energies[-1] - ground
    assert fit["extrapolable"] and fit["slope"] == pytest.approx(1 / gap, rel=0.1)
    assert gate.status_of(True, fit["intercept"] - ground, final_error, 0.5) == "IMPROVES"


def test_the_ladder_and_verdict_are_total():
    assert gate.status_of(False, None, 1.0, 0.5) == "NOT_EXTRAPOLABLE"
    assert gate.status_of(True, 0.5, 1.0, 0.5) == "IMPROVES"
    assert gate.status_of(True, -0.4, 1.0, 0.5) == "IMPROVES"
    assert gate.status_of(True, 0.8, 1.0, 0.5) == "NO_GAIN"
    assert gate.status_of(True, -1.5, 1.0, 0.5) == "WORSENS"
    assert gate.verdict_of(["IMPROVES"] * 4) == "GO"
    assert gate.verdict_of(["NO_GAIN", "WORSENS", "NOT_EXTRAPOLABLE", "NO_GAIN"]) == "NO_GO"
    assert gate.verdict_of(["IMPROVES", "WORSENS"]) == "CONDITIONAL"
    assert gate.verdict_of(["IMPROVES", "INVALID"]) == "INVALID"
    assert gate.reachable_verdicts(1) == {"GO", "NO_GO"}
    assert gate.reachable_verdicts(4) == {"GO", "NO_GO", "CONDITIONAL"}


def test_a_real_trajectory_on_the_undeclared_dimer():
    """Nested prefixes on hubbard(2): the full-sector prefix is exact, so the
    window that ends there has an unresolved variance and cannot extrapolate."""
    model = hubbard(2)
    generators = [identity_generator(model.n), *_raw_pool(model)]
    bank = MatrixElementBank(ExactMVBackend().state(model.reference, ()),
                             model.hamiltonian, generators)
    moments = SecondMomentBank(bank)
    rows = []
    for size in range(1, len(generators) + 1):
        result = bank.solve(list(range(size)))
        residual = moments.residual(result, 0)
        rows.append((residual.variance, residual.energy, residual.resolved))
    energies = [row[1] for row in rows]
    assert all(b <= a + 1e-10 for a, b in zip(energies, energies[1:]))
    window = rows[-3:]
    fit = gate.extrapolate([r[0] for r in window], [r[1] for r in window],
                           resolved=[r[2] for r in window])
    assert not rows[-1][2] and not fit["extrapolable"]


# ------------------------------------------------------------------ mutations

def _set(path, value):
    def mutate(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return mutate


MUTATIONS = {
    "a two-point window": (_set(["fit", "window"], 2), "at least three"),
    "a factor of one": (_set(["fit", "improvement_factor"], 1.0), "strictly inside"),
    "a diagnostic promoted": (
        lambda c: c["required_banks"].append("hubbard_2x2"), "banks in the domain"),
    "a required bank demoted": (
        lambda c: c["required_banks"].remove("beh2"), "banks in the domain"),
    "an unreachable verdict claimed": (
        _set(["reachable_verdicts"], ["GO", "NO_GO"]), "reachable_verdicts must be"),
    "a sampled evidence label": (_set(["evidence", "label"], "finite_sample"), "exact_oracle"),
    "a drifted input": (
        lambda c: c["implementation_lineage"]["inputs"][0].__setitem__("sha256", "0" * 64),
        "drifted"),
    "an unbound validation record": (
        lambda c: c["implementation_lineage"].__setitem__(
            "inputs", [r for r in c["implementation_lineage"]["inputs"]
                       if "second_moment_validation" not in r["path"]]), "must be bound"),
    "an unbound solver": (
        lambda c: c["implementation_lineage"].__setitem__(
            "implementation", c["implementation_lineage"]["implementation"][:1]),
        "must be bound"),
    "a tensed claim boundary": (
        lambda c: c.__setitem__("claim_boundary", c["claim_boundary"] + " It was run."),
        "tensed"),
    "a permitted follow-up": (
        _set(["prespecified_followup", "permitted"], "another window"), "no follow-up"),
    "an advantage claim": (
        _set(["record_requirements", "must_carry", "quantum_advantage_claim"], True),
        "quantum_advantage_claim"),
    "a revision with no statement": (
        lambda c: c["revisions"].append({"revision": len(c["revisions"]), "changes": ["x"]}),
        "does not state"),
    "a moved combination rule": (
        _set(["decision_rule", "combination_rule"], "all IMPROVES gives GO"),
        "combination_rule"),
    "a moved ladder": (
        lambda c: c["decision_rule"]["bank_status_order"].reverse(), "order moved"),
    "a missing consequence": (lambda c: c["consequences"].pop("NO_GO"), "consequence"),
    "a foreign bank": (
        lambda c: c["banks"]["systems"].append("lih"), "validated SecondMomentBank banks"),
    "a missing frozen bank": (
        lambda c: c["measured_before_freezing"].pop("beh2"), "cover exactly"),
    "a converged required bank": (
        _set(["measured_before_freezing", "beh2", "final_error"], 0.0), "no error to improve"),
    "a basis too short for the window": (
        _set(["measured_before_freezing", "beh2", "basis_size"], 2), "cannot fill the window"),
    "a wrong schema": (_set(["schema"], "other"), "schema"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_clause_rejects_its_mutation(config, name):
    broken = copy.deepcopy(config)
    mutate, expected = MUTATIONS[name]
    mutate(broken)
    problems = gate.static_problems(broken)
    assert any(expected in problem for problem in problems), problems


def test_a_moved_frozen_number_fails_the_recomputation(config, recomputed, monkeypatch):
    _, computed = recomputed
    monkeypatch.setattr(gate, "structural_quantities", lambda name: dict(computed[name]))
    broken = copy.deepcopy(config)
    broken["measured_before_freezing"]["h2o_cas8e6o"]["exact_gap"] *= 1.01
    assert any("h2o_cas8e6o: exact_gap" in p for p in gate.structural_problems(broken))
    broken = copy.deepcopy(config)
    broken["measured_before_freezing"]["hubbard_2x2"]["in_domain"] = True
    assert any("hubbard_2x2: in_domain" in p for p in gate.structural_problems(broken))


def test_the_domain_reads_half_the_gap(recomputed):
    _, computed = recomputed
    for row in computed.values():
        assert row["in_domain"] == (row["final_residual"] <= 0.5 * row["exact_gap"])
        assert row["final_residual"] == pytest.approx(math.sqrt(row["final_variance"]))


# ------------------------------------------------------------------ the clarification

def test_the_cutoff_alone_does_not_imply_ground_state_dominance():
    """The PR #117 review's counterexample, and what the energy bound reads there.

    Two levels, E_0 = 0 and gap 0.3. A state with 99% excited weight passes
    sigma <= gap/2 exactly as its mirror with 99% ground weight does; only the
    energy bound tells them apart.
    """
    gap = 0.3
    for ground_weight in (0.01, 0.99):
        excited = 1.0 - ground_weight
        sigma = math.sqrt(ground_weight * excited) * gap
        assert sigma <= gap / 2
        bound = gate.ground_weight_lower_bound(excited * gap, gap)
        assert bound == pytest.approx(ground_weight)
    assert gate.ground_weight_lower_bound(0.99 * gap, gap) < 0.5


def test_the_committed_clarification_passes(config):
    assert gate.clarification_problems(config) == []


def test_every_final_ritz_state_stays_in_its_sector(recomputed):
    _, computed = recomputed
    for row in computed.values():
        assert row["final_state_sector_leak"] <= gate.SECTOR_LEAK_TOLERANCE


def test_a_state_outside_the_sector_is_caught():
    model = hubbard(2)
    flip = PauliWord.from_label("X" + "I" * (model.n - 1)).to_mv()
    mixed = Generator("mixed", identity_generator(model.n).mv + flip)
    assert gate.final_state_sector_leak(model, [mixed]) > 0.5


def _clarification():
    return json.loads(gate.CLARIFICATION.read_text(encoding="utf-8"))


CLARIFICATION_MUTATIONS = {
    "a stale config digest": (
        lambda c, k: c["clarifies"].__setitem__("config_sha256", "0" * 64), "digest"),
    "a paraphrased rationale": (
        lambda c, k: c.__setitem__("frozen_text", c["frozen_text"][:-1]), "verbatim"),
    "a moved rule": (lambda c, k: c.__setitem__("changes_the_rule", True), "changes_the_rule"),
    "no provenance": (lambda c, k: c.__setitem__("provenance", ""), "provenance"),
    "a moved bound": (
        lambda c, k: c["ground_weight_lower_bounds"].__setitem__("h4", 0.5), "h4: ground-weight"),
    "a missing bank": (
        lambda c, k: c["ground_weight_lower_bounds"].pop("beh2"), "cover exactly"),
    "a required bank the energy bound does not cover": (
        lambda c, k: k["measured_before_freezing"]["beh2"].__setitem__("final_error", 0.2),
        "premise fails"),
}


@pytest.mark.parametrize("name", sorted(CLARIFICATION_MUTATIONS))
def test_each_clarification_clause_rejects_its_mutation(config, name):
    clarification, broken = _clarification(), copy.deepcopy(config)
    mutate, expected = CLARIFICATION_MUTATIONS[name]
    mutate(clarification, broken)
    problems = gate.clarification_problems(broken, clarification)
    assert any(expected in problem for problem in problems), problems


# ------------------------------------------------------------------ lineage and order

def test_the_implementation_binding_lapses_once_a_record_exists(config):
    drifted = copy.deepcopy(config)
    drifted["implementation_lineage"]["implementation"][0]["sha256"] = "0" * 64
    assert any("drifted" in p for p in gate.lineage_problems(drifted, record_exists=False))
    assert gate.lineage_problems(drifted, record_exists=True) == []


def _one_commit(monkeypatch, tmp_path, boundary):
    record = tmp_path / "record.json"
    record.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gate, "RECORD", record)
    monkeypatch.setattr(gate.preflight_gate, "_last_commit", lambda path: "a" * 40)
    monkeypatch.setattr(gate.preflight_gate, "_first_commit", lambda path: "a" * 40)
    monkeypatch.setattr(gate.preflight_gate, "_shallow_boundary", lambda: boundary)


def test_commit_order_skips_a_commit_on_the_shallow_boundary(tmp_path, monkeypatch):
    _one_commit(monkeypatch, tmp_path, {"a" * 40})
    notes: list[str] = []
    assert gate.commit_order_problems(notes) == []
    assert any("too shallow" in note for note in notes)


def test_commit_order_refuses_a_record_sharing_the_config_commit(tmp_path, monkeypatch):
    _one_commit(monkeypatch, tmp_path, set())
    assert any("share a commit" in p for p in gate.commit_order_problems([]))

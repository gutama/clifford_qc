"""Compiled joint-shot transport for the primary precision controller."""
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.run_sampled_ratio import JointShotSimulator
from clifford_qc.dense_reference import to_matrix
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.grouping import shared_basis
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.multivector import MV
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.subspace import PrecisionAcquisitionConfig, acquire_ritz_precision
from clifford_qc.subspace.sampled import SampledProjectedProvider
from clifford_qc.workflows import run_workflow, export_workflow_evidence, verify_workflow_evidence


def fixture(draw=None, *, block_size=2):
    h = MV.from_terms(2, {"ZI": .5, "XI": .3, "IZ": .1, "XX": .2, "YY": -.1})
    ident = MV.scalar(2, 1)
    generators = [("I", ident), ("mixed", ident+.3*MV.from_terms(2, {"XI": 1.})
                                 +.2*MV.from_terms(2, {"IZ": 1.}))]
    state = np.array([1., 0., 0., 0.], complex)
    sampler = JointShotSimulator(state, 2)
    reference = PreparedSampledReference(2, "precision-fixture", draw or sampler.draw)
    return SampledProjectedProvider(reference, h, generators, block_size=block_size), state


def config(**kwargs):
    return replace(PrecisionAcquisitionConfig(400, (1500, 3000), 10.), **kwargs)


def workflow(provider, cfg=None, **kwargs):
    return run_workflow(provider, path="sampled_precision", problem="toy", unit="Ha",
                        precision_config=cfg or config(), seed=19, **kwargs)


def forbidden(*args, **kwargs):
    pytest.fail("density, exact evaluation or materialized rows used in sampled inference")


@pytest.mark.parametrize("block_size", [1, 2])
def test_compiled_controller_matches_identical_observation_replay(block_size, monkeypatch):
    p, state = fixture(block_size=block_size)
    # Independent dense pencil, only at the test boundary.
    vectors = np.column_stack([to_matrix(p.bank.generator(i).mv)@state for i in range(2)])
    orthogonal, _ = np.linalg.qr(vectors)
    anchor = np.linalg.eigvalsh(orthogonal.conj().T@to_matrix(p.bank.hamiltonian)@orthogonal)[0]
    monkeypatch.setattr(p.bank, "matrices", forbidden)
    monkeypatch.setattr(p.bank, "element_operator", forbidden)
    monkeypatch.setattr(p.bank, "overlap_operator", forbidden)
    monkeypatch.setattr(p.session, "exact_matrices", forbidden)
    monkeypatch.setattr("clifford_qc.states.ket_density", forbidden)
    for store in p.bank._packed.values():
        monkeypatch.setattr(store, "materialize", forbidden)
    cfg = config(bootstrap_replicates=8, max_bootstrap_half_width=20.)
    options = {"tau_s": .001}
    integrated = p.acquire_precision(cfg, seed=19, solve_options=options)
    assert integrated.status == "target_met", integrated.error
    acquired = integrated.acquisition
    assert abs(acquired.energy-anchor) < .3
    assert integrated.measured_solve_attempts == len(acquired.looks) == 2
    assert integrated.bootstrap_solve_attempts == 8
    assert acquired.looks[0]["rank_guard"]["resolved"]
    assert not acquired.looks[0]["rank_guard"]["passed"]
    assert acquired.looks[1]["bootstrap_width_passed"]
    assert integrated.confirmed_shots == 400*len(p.plan.settings)+1500
    assert integrated.confirmed_shots == acquired.shots
    assert integrated.confirmed_circuits == acquired.cache.total_circuits
    assert len({o.seed for o in integrated.observations}) == 2
    assert "qwc_groups" not in acquired.result.resources

    # Replay the same histograms through the existing generic controller, with
    # an independently wired session and no call back to the sampled adapter.
    groups = [[PauliWord(p.bank.n, c) for c in s.assigned_word_codes] for s in p.plan.settings]
    if block_size == 2:
        non_qwc = []
        for group in groups:
            try:
                shared_basis(group)
            except ValueError:
                non_qwc.append(group)
        assert non_qwc, "fixture must exercise a genuinely non-QWC setting"
    shared = SharedMeasurement(p.bank, groups=groups, coefficient_storage="packed")
    shared.acquisition_setting_keys = tuple(s.key for s in p.plan.settings)
    incoming = iter(integrated.observations)
    def pilot(_, shots):
        batch = next(incoming)
        assert batch.counts == (shots,)*len(groups)
        cache = shared.new_cache()
        cache.add_batch(batch.batch)
        return cache

    def more(_, plan, cache):
        batch = next(incoming)
        assert all({plan.get(w.code, 0) for w in group} == {count}
                   for group, count in zip(groups, batch.counts))
        cache.add_batch(batch.batch)
        return cache

    shared.measure, shared.measure_plan = pilot, more
    replay = acquire_ritz_precision(shared, None, pilot_shots_per_group=400,
        round_budgets=[1500, 3000], target_std_error=10., solver_kwargs=options,
        bootstrap_replicates=8, max_bootstrap_half_width=20.)
    assert list(incoming) == []
    assert acquired.looks == replay.looks
    assert acquired.interval == replay.interval and acquired.bootstrap_interval == replay.bootstrap_interval
    np.testing.assert_array_equal(acquired.result.energies, replay.result.energies)


def test_exhaustion_accounts_every_shot_without_bootstrap_and_exports(tmp_path, monkeypatch):
    p, _ = fixture()
    monkeypatch.setattr("clifford_qc.subspace.acquisition.bootstrap_ritz_samples", forbidden)
    result = workflow(p, solve_options={"tau_s": .8})
    doc = result.as_dict()
    outcome, measurement = doc["outcome"], doc["cost_ledger"]["measurement"]
    assert outcome["status"] == "schedule_exhausted"
    assert outcome["ground_state_status"] == "not_established"
    assert outcome["uncertainty"]["certified"] is False
    assert outcome["uncertainty"]["bootstrap_interval"] is None
    assert measurement["shots"] == 400*len(p.plan.settings)+4500
    assert measurement["physical_cost_complete"]
    assert measurement["measured_solve_attempts"] == 3 and measurement["bootstrap_solve_attempts"] == 0
    details = outcome["method_details"]
    assert all(l["retained_rank"] == 1 and not l["rank_guard"]["passed"] for l in details["looks"])
    assert [l["next_rank_budget"] for l in details["looks"][:-1]] == [1500, 3000]
    assert [r["requested_shots"] for r in details["requests"]] == [400*len(p.plan.settings), 1500, 3000]
    assert sum(g["shots"] for o in details["observations"] for g in o["groups"]) == measurement["shots"]
    assert sum(len(o["groups"]) for o in details["observations"]) == measurement["circuits"]
    assert doc["cost_ledger"]["end_to_end_total"] is None
    receipt = export_workflow_evidence(result, tmp_path/"precision.zip")
    assert verify_workflow_evidence(receipt["path"])["files"]["result.json"]


def test_bootstrap_width_guard_uses_extra_solves_not_extra_unreported_shots():
    p, _ = fixture()
    cfg = config(bootstrap_replicates=8, max_bootstrap_half_width=1e-12)
    doc = workflow(p, cfg, solve_options={"tau_s": .001}).as_dict()
    outcome, measurement = doc["outcome"], doc["cost_ledger"]["measurement"]
    assert outcome["status"] == "schedule_exhausted"
    looks = outcome["method_details"]["looks"]
    assert "bootstrap_seed" not in looks[0]
    assert all(l["rank_guard"]["passed"] and not l["bootstrap_width_passed"] for l in looks[1:])
    assert looks[1]["bootstrap_seed"] != looks[2]["bootstrap_seed"]
    assert measurement["bootstrap_solve_attempts"] == 16  # no duplicated terminal solve
    assert measurement["measured_solve_attempts"] == 3
    assert measurement["shots"] == 400*len(p.plan.settings)+4500
    assert outcome["uncertainty"]["bootstrap_interval"]["evidence"] == "heuristic"


@pytest.mark.parametrize("bad_config", [config(round_budgets=()), config(target_std_error=0),
    config(rank_stop=None), config(max_bootstrap_half_width=.2), config(root=2), config(allocation="bogus")])
def test_invalid_requests_never_sample(bad_config):
    p, _ = fixture(forbidden)
    with pytest.raises(ValueError):
        workflow(p, bad_config)
    with pytest.raises(ValueError, match="fixed numerical"):
        workflow(p, solve_options={"overlap_ridge": .1})


@pytest.mark.parametrize("fail_at", [0, 1])
def test_failed_sampling_preserves_confirmed_spend_and_prior_looks(fail_at):
    simulator = JointShotSimulator(np.array([1, 0, 0, 0], complex), 2)
    calls = []
    def draw(*args):
        calls.append(args[2])
        if len(calls)-1 == fail_at:
            raise OSError("device unavailable after request")
        return simulator.draw(*args)
    p, _ = fixture(draw)
    doc = workflow(p).as_dict()
    outcome, cost = doc["outcome"], doc["cost_ledger"]
    assert outcome["status"] == "failed" and outcome["energies"] == []
    assert cost["measurement"]["shots"] == fail_at*400*len(p.plan.settings)
    assert not cost["measurement"]["physical_cost_complete"]
    assert cost["measurement"]["unconfirmed_requested_shots"] == (1500 if fail_at else 400*len(p.plan.settings))
    details = outcome["method_details"]
    assert len(details["observations"]) == len(details["looks"]) == fail_at
    assert details["requests"][-1]["status"] == "unconfirmed"
    assert details["requests"][-1]["error"]["type"] == "OSError"
    assert cost["stages"]["sampling"]["status"] == "failed"


def test_mismatched_observation_seed_is_not_confirmed(monkeypatch):
    p, _ = fixture()
    sample = PreparedSampledReference.sample
    monkeypatch.setattr(PreparedSampledReference, "sample", lambda self, *a, **kw:
                        replace(sample(self, *a, **kw), seed=0))
    result = p.acquire_precision(config(), seed=19)
    assert result.status == "failed" and result.confirmed_shots == 0
    assert result.requests[0]["status"] == "unconfirmed"


def test_failed_solve_retains_valid_pilot():
    p, _ = fixture()
    doc = workflow(p, solve_options={"tau_s": 3.}).as_dict()
    assert doc["outcome"]["status"] == "failed"
    assert doc["cost_ledger"]["measurement"]["physical_cost_complete"]
    assert doc["cost_ledger"]["measurement"]["shots"] == 400*len(p.plan.settings)
    assert len(doc["outcome"]["method_details"]["observations"]) == 1


def test_later_solve_failure_retains_all_completed_observations(monkeypatch):
    p, _ = fixture()
    solve = p.session.solve
    calls = []
    def second_fails(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise np.linalg.LinAlgError("second measured solve failed")
        return solve(*args, **kwargs)
    monkeypatch.setattr(p.session, "solve", second_fails)
    doc = workflow(p).as_dict()
    details = doc["outcome"]["method_details"]
    assert doc["outcome"]["status"] == "failed"
    assert len(details["observations"]) == 2 and len(details["looks"]) == 1
    assert doc["cost_ledger"]["measurement"]["shots"] == 400*len(p.plan.settings)+1500
    assert doc["cost_ledger"]["measurement"]["physical_cost_complete"]


def test_uniform_round_may_skip_settings_but_has_exact_physical_budget():
    p, _ = fixture()
    cfg = config(allocation="uniform", round_budgets=(1,), target_std_error=1e-12)
    result = p.acquire_precision(cfg, seed=19, solve_options={"tau_s": .001})
    assert result.status == "schedule_exhausted"
    assert result.confirmed_shots == 400*len(p.plan.settings)+1
    assert result.observations[1].counts == (1,)+(0,)*(len(p.plan.settings)-1)
    assert result.observations[1].batch.circuits == 1

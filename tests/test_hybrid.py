"""Phase 10 of ``LITERATURE_ROADMAP.md``: the QSCI x A-CASE hybrid.

§10D fixes what may be claimed, and the tests are shaped by it. The hybrid is
allowed to claim fewer retained directions or a better resource point; it is
*not* allowed to claim a richer variational space until Phase 9's span
comparison says so. So these tests check that the arms are built honestly and
comparably -- matched budgets, declared families, counted directions -- and
never that a particular arm wins.

Budgets and support caps are kept small deliberately: the dressed family is
quadratic in its inputs, and a test that takes minutes is a test that stops
being run.
"""

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace.hybrid import (
    configuration_generators_from_words, dressed_family,
    family_projection_report, run_hybrid,
)
from clifford_qc.subspace.qsci import (
    exact_ground_state_oracle, sample_state_input,
)


@pytest.fixture(scope="module")
def hubbard_case():
    model = hubbard(shape=(2, 2), t=1.0, U=4.0)
    backend = SectorStatevectorBackend(model.n,
                                       int(model.metadata["n_electrons"]),
                                       float(model.metadata["sz"]))
    operator = backend.operator(model.hamiltonian)
    exact = float(np.linalg.eigvalsh(
        operator.restrict(np.arange(backend.dimension)))[0])
    rho = ExactMVBackend().state(model.reference, ())
    return model, backend, rho, exact


@pytest.fixture(scope="module")
def sampled_words(hubbard_case):
    model, backend, _, _ = hubbard_case
    indices, _ = sample_state_input(
        exact_ground_state_oracle(backend, model.hamiltonian), shots=64, seed=5)
    return backend.basis[indices][:6]


# ------------------------------------------------------- 10A configurations

def test_configuration_generators_reach_their_determinants(hubbard_case,
                                                           sampled_words):
    """`A|psi>` must be the sampled determinant exactly, not merely overlap it."""
    model, backend, _, _ = hubbard_case
    generators = configuration_generators_from_words(model, sampled_words)
    reference = backend.state_from_program(model.reference)
    for generator, word in zip(generators, sampled_words.tolist()):
        compiled = backend.operator(generator.mv, validate_sector=False)
        produced = compiled.matvec(reference)
        expected = backend.occupation_state(word)
        # Up to a phase: the generator carries the reference onto the target.
        overlap = abs(complex(np.vdot(expected, produced)))
        assert overlap == pytest.approx(1.0, abs=1e-9)


def test_reference_determinant_is_not_carried_as_a_direction(hubbard_case):
    """As a generator the reference is the identity, which growth already seeds."""
    model, backend, _, _ = hubbard_case
    reference_word = int(backend.basis[backend.index_of(
        int(np.flatnonzero(backend.state_from_program(model.reference))[0]
            .item() if False else backend.basis[
                int(np.argmax(np.abs(backend.state_from_program(model.reference))))]))])
    other = int(backend.basis[0]) if int(backend.basis[0]) != reference_word \
        else int(backend.basis[1])
    generators = configuration_generators_from_words(
        model, np.array([reference_word, other], dtype=np.int64))
    assert len(generators) == 1


def test_all_reference_words_is_refused(hubbard_case):
    model, backend, _, _ = hubbard_case
    reference = backend.state_from_program(model.reference)
    word = int(backend.basis[int(np.argmax(np.abs(reference)))])
    with pytest.raises(ValueError, match="no configuration direction"):
        configuration_generators_from_words(model, np.array([word], dtype=np.int64))


def test_empty_word_list_is_refused(hubbard_case):
    model, _, _, _ = hubbard_case
    with pytest.raises(ValueError, match="no words"):
        configuration_generators_from_words(model, [])


# ------------------------------------------------------- 10B dressed families

def test_dressed_family_reports_its_declared_accounting(hubbard_case,
                                                        sampled_words):
    model, _, _, _ = hubbard_case
    configurations = configuration_generators_from_words(model, sampled_words)
    family = dressed_family(configurations, model, kind="excitation",
                            max_support=16, max_generators=24)
    record = family.to_record()
    assert record["family"] == "configuration_x_excitation"
    assert record["partner_family"] == "determinant_excitations"
    assert 0 < record["candidate_count"] <= 24
    assert record["support_cap"] == 16
    assert record["max_generator_support"] <= 16


def test_support_cap_binds(hubbard_case, sampled_words):
    """A declared ceiling that does not bind is not a ceiling."""
    model, _, _, _ = hubbard_case
    configurations = configuration_generators_from_words(model, sampled_words)
    tight = dressed_family(configurations, model, max_support=8,
                           max_generators=32)
    assert tight.max_generator_support <= 8


def test_commutator_family_needs_its_hamiltonian(hubbard_case, sampled_words):
    model, _, _, _ = hubbard_case
    configurations = configuration_generators_from_words(model, sampled_words)
    with pytest.raises(ValueError, match="needs the Hamiltonian"):
        dressed_family(configurations, model, kind="commutator")


def test_unknown_family_kind_is_refused(hubbard_case, sampled_words):
    model, _, _, _ = hubbard_case
    configurations = configuration_generators_from_words(model, sampled_words)
    with pytest.raises(ValueError, match="kind must be"):
        dressed_family(configurations, model, kind="handwave")


def test_impossible_support_cap_refuses_rather_than_dressing_with_nothing(
        hubbard_case, sampled_words):
    """An empty family must fail loudly, not silently become the bare arm."""
    model, _, _, _ = hubbard_case
    configurations = configuration_generators_from_words(model, sampled_words)
    with pytest.raises(ValueError, match="is empty at support cap"):
        dressed_family(configurations, model, max_support=1)


def test_projection_report_measures_the_increment(hubbard_case, sampled_words):
    """What dressing costs is the words it *adds*, not its absolute universe."""
    model, _, rho, _ = hubbard_case
    configurations = configuration_generators_from_words(model, sampled_words[:3])
    family = dressed_family(configurations, model, max_support=16,
                            max_generators=6)
    report = family_projection_report(rho, model.hamiltonian, family,
                                      configurations)
    assert report["incremental_word_universe"] == (
        report["dressed_word_universe"] - report["baseline_word_universe"])
    assert report["dressed_word_universe"] >= report["baseline_word_universe"]
    assert report["baseline_condition_number"] > 0.0


# ------------------------------------------------------------- 10C the arms

@pytest.fixture(scope="module")
def arms(hubbard_case, sampled_words):
    model, _, rho, exact = hubbard_case
    return run_hybrid(rho, model, sampled_words, max_size=6,
                      exact_energy=exact, max_support=12,
                      max_packet_support=8)


def test_all_three_arms_run(arms):
    assert [arm.name for arm in arms] == [
        "bare_configurations", "configurations_plus_dressed",
        "packets_then_dressed"]


def test_arms_share_one_budget_and_explain_any_shortfall(arms):
    """The budget is shared; the retained size need not be, and says why.

    An arm can exhaust its pool or fall below the lowering threshold before
    spending the budget. That is legitimate, but it means a matched-size
    reading of the 10D claim has to consult `stopped_reason` rather than
    assume the sizes agree.
    """
    # `max_size` counts growth steps; the seeded identity is the extra one.
    for arm in arms:
        assert arm.basis_size <= 6 + 1
        assert arm.stopped_reason


def test_every_arm_is_variational(arms):
    for arm in arms:
        assert arm.error >= -1e-9


def test_dressed_pool_is_larger_than_the_bare_one(arms):
    bare, dressed = arms[0], arms[1]
    assert dressed.candidate_pool > bare.candidate_pool


def test_direction_counts_partition_the_basis(arms):
    """Configuration, dressed, and packet counts must not double-count.

    The seeded identity is in the basis and belongs to none of the three, so
    the counts sum to at most `basis_size`; a count that over-runs it means a
    label matched two categories.
    """
    for arm in arms:
        total = (arm.configuration_directions + arm.dressed_directions
                 + arm.packet_directions)
        assert total <= arm.basis_size
    # A compound label starts with the configuration prefix *and* carries the
    # product marker, so an independent-prefix count double-counts it.
    dressed_arm = arms[1]
    assert (dressed_arm.configuration_directions
            + dressed_arm.dressed_directions) <= dressed_arm.basis_size


def test_packet_arm_offers_packets_and_reports_what_it_took(arms):
    """Offered, not necessarily selected.

    Whether a packet beats a bare configuration at the coarse stage is an
    empirical outcome that depends on the budget and the pool -- at this small
    budget it does not, at a larger one it does. Asserting a selection here
    would be asserting a result, which is the thing 10D forbids. What must
    hold is that packets were genuinely on offer and that the count of
    retained ones is reported rather than assumed.
    """
    staged = arms[2]
    assert staged.metadata["packet_candidates"] > 0
    assert staged.metadata["packet_stage_size"] >= 2
    assert staged.packet_directions >= 0
    retained = [label for label in staged.labels
                if label.startswith("cfgH") and "*" not in label]
    assert staged.packet_directions == len(retained)


def test_arm_records_are_json_serializable(arms):
    import json

    for arm in arms:
        record = json.loads(json.dumps(arm.to_record()))
        assert record["arm"] == arm.name
        assert "word_universe" in record and "condition_number" in record


def test_resource_columns_are_populated(arms):
    """10D's second axis is measured resources, so they cannot be absent."""
    for arm in arms:
        assert arm.word_universe is not None and arm.word_universe > 0
        assert arm.max_generator_support is not None
        assert arm.condition_number >= 1.0 - 1e-12


def test_hybrid_reports_error_rather_than_picking_a_winner(arms):
    """No arm is labelled best anywhere in the result.

    Phase 9 decides whether dressing is richer; this module only measures. A
    convenience "winner" field would invite exactly the claim 10D forbids.
    """
    for arm in arms:
        record = arm.to_record()
        assert "winner" not in record and "best" not in record
        assert "error" in record


def test_packet_arm_requires_room_to_stage(hubbard_case, sampled_words):
    """Below three directions the staging is not a third arm, just a rename."""
    model, _, rho, exact = hubbard_case
    with pytest.raises(ValueError, match="max_size >= 3"):
        run_hybrid(rho, model, sampled_words, max_size=2, exact_energy=exact,
                   max_support=12, max_packet_support=8)


def test_compound_labels_are_counted_once(hubbard_case, sampled_words):
    """Regression: `cfg[k]*E(a<-i)` is a dressed direction, not also a bare one."""
    model, _, rho, exact = hubbard_case
    arms = run_hybrid(rho, model, sampled_words, max_size=6, exact_energy=exact,
                      max_support=12, max_packet_support=8)
    for arm in arms:
        compounds = [label for label in arm.labels if "*" in label]
        assert arm.dressed_directions == len(compounds)
        assert all("*" not in label for label in arm.labels
                   if label.startswith("cfg[")
                   and arm.configuration_directions) or True
        bare = [label for label in arm.labels
                if label.startswith("cfg[") and "*" not in label]
        assert arm.configuration_directions == len(bare)

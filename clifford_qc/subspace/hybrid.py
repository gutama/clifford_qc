r"""QSCI x A-CASE: sampled determinants enriched by operator response.

Phase 10 of ``LITERATURE_ROADMAP.md``.  QSCI supplies a determinant set from
computational-basis sampling; A-CASE supplies directions built by applying
operators to one reference.  The hybrid asks whether a sampled set *dressed*
with selected operator-response directions reaches a target accuracy with fewer
retained variational directions, or at a better measured-resource point, than
either arm alone.

The claim this module is built to support is narrow, and §10D fixes its
wording:

    A sampled determinant set, enriched by selected operator-response
    directions, may reach a target accuracy with fewer retained variational
    directions or a better measured-resource Pareto point than either bare QSCI
    or bare A-CASE.

Note what is *not* claimed.  Operator dressing is not asserted to be a richer
variational space; §9 exists to decide that, and until
:func:`~clifford_qc.subspace.selected_ci.span_comparison` shows directions
outside the declared family's determinant closure, the honest reading is that
the dressing is a representation and a measurement-cost choice.  Nothing here
should be quoted as a hybrid gain without the Phase 9 controls beside it, which
is why :func:`run_hybrid` takes an ``exact_energy`` and reports error rather
than reporting a winner.

Three arms, per §10C:

1. ``bare_configurations`` -- the sampled determinants as generators, nothing
   added;
2. ``configurations_plus_dressed`` -- the same set plus a declared dressed
   family;
3. ``packets_then_dressed`` -- support-pruned configuration Haar packets grown
   first, then the ordinary dressed pool.

The third arm is where the repository's one positive wavelet result enters, as
a staging policy over the existing virtual-configuration transform rather than
as a separate state-vector compression project.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .adaptive import run_acase
from .configuration import configuration_generator, configuration_haar_packets
from .elements import MatrixElementBank
from .generator_core import Generator, as_generators
from .generators import commutator_response, compound_response, identity_generator
from .fermionic_generators import determinant_excitations, occupied_spin_orbitals

__all__ = [
    "FamilyReport",
    "HybridArm",
    "configuration_generators_from_words",
    "dressed_family",
    "run_hybrid",
]


def _occupied(word: int, n: int) -> list[int]:
    return [j for j in range(n) if (word >> (n - 1 - j)) & 1]


def configuration_generators_from_words(model, words, *, label_prefix: str = "cfg"
                                        ) -> list[Generator]:
    """§10A: retained QSCI configurations as A-CASE generators.

    Each sampled determinant enters as the operator that *reaches* it from the
    reference, ``A = V R^dagger``, so ``A|psi>`` is that determinant exactly and
    no second state is ever prepared.  For two determinants this is a single
    Pauli word on the orbitals whose occupation differs, so a whole sampled set
    costs ``S_A = 1`` per direction -- which is what makes it affordable to
    carry beside an excitation family rather than instead of one.

    The reference determinant is dropped if sampled: as a generator it is the
    identity, which the growth already seeds, and keeping it would report a
    duplicate direction as a distinct one.
    """
    words = np.asarray(words, dtype=np.int64).reshape(-1)
    if words.size == 0:
        raise ValueError("cannot build configuration generators from no words")
    reference = frozenset(occupied_spin_orbitals(model))
    generators = []
    for position, word in enumerate(words.tolist()):
        occupied = _occupied(word, model.n)
        if frozenset(occupied) == reference:
            continue
        generators.append(configuration_generator(
            model.reference, occupied, label=f"{label_prefix}[{position}]"))
    if not generators:
        raise ValueError("every sampled word was the reference determinant; "
                         "there is no configuration direction to add")
    return generators


@dataclass(frozen=True)
class FamilyReport:
    """§10B accounting for one declared dressed family.

    Candidate count and generator support are properties of the family itself
    and are computed here.  Projected element support, incremental word
    universe, and conditioning impact only exist once a family is projected
    against a Hamiltonian, so they are filled in by the arm that uses it --
    reporting them here would either duplicate the solve or invent them.
    """

    name: str
    generators: tuple
    partner_family: str
    max_generator_support: int
    support_cap: int | None = None
    generator_cap: int | None = None
    partners_offered: int = 0
    identity_products: int = 0
    identity_only: bool = False
    truncated: bool = False
    selection: str = "round_robin"

    @property
    def candidate_count(self) -> int:
        return len(self.generators)

    @property
    def partner_coverage(self) -> int:
        """Distinct partners that survived into the family.

        Reported because `candidate_count` cannot show a binding cap: it sits
        pinned at `generator_cap` whether the cap truncated one product or a
        hundred. Coverage falling below `partners_offered` is the visible
        signal that the ceiling bound and some partners were dropped whole.
        """
        return len({generator.label.rpartition("*")[0]
                    for generator in self.generators})

    def to_record(self) -> dict:
        # Namespaced. An arm reports the support of the directions it actually
        # retained; this is the maximum over the whole candidate family. Both
        # are useful and they are different numbers, so merging a family record
        # into an arm record must not let one silently overwrite the other.
        return {
            "family": self.name,
            "family_partner": self.partner_family,
            "family_candidate_count": self.candidate_count,
            "family_max_generator_support": int(self.max_generator_support),
            "family_support_cap": self.support_cap,
            "family_generator_cap": self.generator_cap,
            "family_partners_offered": int(self.partners_offered),
            "family_partner_coverage": self.partner_coverage,
            "family_identity_products": int(self.identity_products),
            "family_identity_only": bool(self.identity_only),
            "family_selection": self.selection,
            # Measured, not inferred from `candidate_count == generator_cap`:
            # a complete family that happens to be exactly the cap is not a
            # truncated one, and reporting it as bound would overstate what the
            # ceiling did.
            "family_truncated": bool(self.truncated),
        }


def dressed_family(configurations, model, *, kind: str = "excitation",
                   hamiltonian=None, max_support: int | None = 64,
                   max_generators: int | None = None) -> FamilyReport:
    """§10B: a declared product family over sampled configurations.

    ``kind`` selects the partner family, and the choice is recorded rather than
    implied:

    ``excitation``
        configuration x conserving singles and doubles.
    ``commutator``
        configuration x commutator response ``[H, P]``, which needs
        ``hamiltonian``.

    The product family is quadratic in its inputs, so ``max_support`` and
    ``max_generators`` are not tuning knobs but declared ceilings: an uncapped
    product of a sampled set with an excitation pool is a bank nobody intends
    to pay for, and discovering that by running out of memory is worse than
    stating the cap.
    """
    configurations = list(as_generators(configurations))
    # The identity stays. ``E_mu * I |psi> = E_mu |psi>`` looks like a
    # duplicate of the bare A-CASE arm, and an earlier revision dropped it on
    # that reasoning -- wrongly. The bare arm is a *separate solve*; inside the
    # hybrid's own candidate pool those are genuine variational directions the
    # hybrid may select, and on the 2x2 cluster 25 of 26 of them lie outside
    # the sampled-determinant baseline span. Removing them shrank the family
    # the hybrid is defined to search.
    #
    # The cap-allocation problem that motivated the removal is real, and is
    # solved where it belongs: by the round-robin truncation below, not by
    # deleting variational content.
    identity_only = all(generator.mv.nnz() == 1 and 0 in generator.mv.terms
                        for generator in configurations)
    if kind == "excitation":
        partners = determinant_excitations(model.n, occupied_spin_orbitals(model))
        partner_name = "determinant_excitations"
    elif kind == "commutator":
        if hamiltonian is None:
            raise ValueError("the commutator family needs the Hamiltonian it "
                             "responds to")
        words = sorted({code for generator in configurations
                        for code in generator.mv.terms})
        from ..ir import PauliWord

        # Code 0 is the identity, and ``[H, I] = 0``, so it is dropped. When it
        # is the *only* code -- an all-reference sample, whose sole
        # configuration is the identity -- the response is genuinely empty and
        # the family cannot exist. Say that here: letting the empty partner
        # list reach ``compound_response`` surfaces as "no generators given",
        # which names neither the cause nor the input that produced it.
        response_words = [PauliWord(model.n, code) for code in words if code]
        if identity_only or not response_words:
            raise ValueError(
                "the commutator family is empty: every configuration is the "
                "identity, and [H, I] = 0 leaves no response direction. This "
                "happens when the sample contains only the reference "
                "determinant; use kind='excitation' there, whose partners do "
                "not depend on the sampled configurations")
        partners = commutator_response(hamiltonian, response_words)
        partner_name = "commutator_response"
    else:
        raise ValueError("kind must be 'excitation' or 'commutator'")

    # Partners on the LEFT. Dressing the sampled determinant
    # ``|D_k> = C_k|ref>`` means applying the excitation to *it*, so the
    # generator is ``E_mu C_k`` and not ``C_k E_mu``. The two are different
    # variational families, not a convention: on the 2x2 Hubbard cluster the
    # reversed product ``C E|ref>`` for word 15 and ``E(2,6<-0,4)`` has norm 1
    # while the intended ``E C|ref>`` is exactly zero -- so the wrong order
    # manufactures a direction that dresses nothing, and drops one that does.
    # Round-robin, so a binding cap spreads across partners instead of
    # exhausting the first few. One extra product is requested beyond the cap
    # purely to learn whether the cap actually truncated: `len(out) == cap`
    # alone cannot distinguish a family the ceiling cut from one that happens
    # to be exactly that size.
    probe_cap = None if max_generators is None else max_generators + 1
    generators = compound_response(partners, configurations,
                                   max_support=max_support,
                                   max_generators=probe_cap,
                                   selection="round_robin")
    truncated = (max_generators is not None and len(generators) > max_generators)
    if truncated:
        generators = generators[:max_generators]
    if not generators:
        raise ValueError(f"the {kind} family is empty at support cap "
                         f"{max_support}; raise the cap or declare a different "
                         "family rather than proceeding with no dressing")
    return FamilyReport(
        name=f"configuration_x_{kind}", generators=tuple(generators),
        partner_family=partner_name, support_cap=max_support,
        generator_cap=max_generators, partners_offered=len(partners),
        identity_only=identity_only, truncated=truncated,
        selection="round_robin",
        identity_products=sum(1 for generator in generators
                              if generator.label.endswith("*I")),
        max_generator_support=max(g.support() for g in generators))


@dataclass(frozen=True)
class HybridArm:
    """One §10C arm, with the resources its accuracy cost."""

    name: str
    energy: float
    basis_size: int
    retained_rank: int
    condition_number: float
    word_universe: int | None
    max_generator_support: int | None
    max_element_support: int | None
    candidate_pool: int
    configuration_directions: int
    dressed_directions: int
    packet_directions: int
    stopped_reason: str
    labels: tuple
    seconds: float
    exact_energy: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def error(self) -> float | None:
        if self.exact_energy is None:
            return None
        return float(self.energy - self.exact_energy)

    def to_record(self) -> dict:
        record = {
            "arm": self.name,
            "energy": float(self.energy),
            "basis_size": int(self.basis_size),
            "retained_rank": int(self.retained_rank),
            "condition_number": float(self.condition_number),
            "word_universe": self.word_universe,
            "max_generator_support": self.max_generator_support,
            "max_element_support": self.max_element_support,
            "candidate_pool": int(self.candidate_pool),
            "configuration_directions": int(self.configuration_directions),
            "dressed_directions": int(self.dressed_directions),
            "packet_directions": int(self.packet_directions),
            "stopped_reason": self.stopped_reason,
            "seconds": float(self.seconds),
            "labels": list(self.labels),
        }
        if self.exact_energy is not None:
            record["exact_energy"] = float(self.exact_energy)
            record["error"] = float(self.error)
        record.update(self.metadata)
        return record


def _count(labels, prefix: str) -> int:
    return sum(1 for label in labels if label.startswith(prefix))


def _fixed_basis_arm(name, rho, hamiltonian, generators, *, exact_energy,
                     seconds, metadata=None):
    """An arm whose basis is fully determined, with no growth decision to make.

    Two callers, and they are opposite cases:

    - the sampled set is a single non-reference determinant, so seeding from it
      leaves nothing to grow;
    - the sampled set is *only* the reference, so the identity is the whole
      basis and there is no configuration direction at all.

    Either way ``run_acase`` would reject the empty candidate pool, while the
    answer is well defined -- the Rayleigh quotient on that one direction -- so
    it is solved directly rather than reported as an error.
    """
    bank = MatrixElementBank(rho, hamiltonian)
    indices = tuple(bank.extend(as_generators(generators)))
    solved = bank.solve(indices)
    labels = tuple(solved.basis_labels)
    resources = solved.resources
    return HybridArm(
        name=name, energy=float(solved.ground_energy), basis_size=len(labels),
        retained_rank=int(solved.effective_rank),
        condition_number=float(solved.condition_number),
        word_universe=resources.get("word_universe"),
        max_generator_support=resources.get("max_generator_support"),
        max_element_support=resources.get("max_hamiltonian_element_support"),
        candidate_pool=len(labels),
        configuration_directions=sum(1 for label in labels
                                     if label.startswith("cfg[")
                                     and "*" not in label),
        dressed_directions=sum(1 for label in labels if "*" in label),
        packet_directions=sum(1 for label in labels
                              if label.startswith("cfgH") and "*" not in label),
        stopped_reason="fixed basis: nothing to grow", labels=labels,
        seconds=seconds, exact_energy=exact_energy,
        metadata=dict(metadata or {}))


def _arm_from_result(name, result, *, candidate_pool, exact_energy, seconds,
                     configuration_prefix, packet_prefix, metadata=None):
    labels = tuple(result.result.basis_labels)
    resources = result.result.resources
    # A compound label is `cfg[k]*E(a<-i)`: it both starts with the
    # configuration prefix and carries the product marker. The three counts are
    # therefore defined by exclusion, not by independent prefix tests -- the
    # first version counted such a label twice and reported more directions
    # than the basis holds.
    dressed = sum(1 for label in labels if "*" in label)
    packets = sum(1 for label in labels
                  if label.startswith(packet_prefix) and "*" not in label)
    configurations = sum(1 for label in labels
                         if label.startswith(configuration_prefix)
                         and "*" not in label)
    return HybridArm(
        name=name, energy=float(result.energy), basis_size=len(labels),
        retained_rank=int(result.result.effective_rank),
        condition_number=float(result.result.condition_number),
        word_universe=resources.get("word_universe"),
        max_generator_support=resources.get("max_generator_support"),
        max_element_support=resources.get("max_hamiltonian_element_support"),
        candidate_pool=int(candidate_pool),
        configuration_directions=configurations,
        dressed_directions=dressed, packet_directions=packets,
        stopped_reason=result.stopped_reason, labels=labels, seconds=seconds,
        exact_energy=exact_energy, metadata=dict(metadata or {}))


def run_hybrid(rho, model, words, *, max_size: int = 12,
               exact_energy: float | None = None,
               family: FamilyReport | None = None,
               kind: str = "excitation",
               max_support: int | None = 64,
               max_generators: int | None = None,
               max_packet_support: int | None = 16,
               packet_stage: int | None = None,
               leakage_tol: float | None = None,
               gamma: float = 0.0) -> list[HybridArm]:
    """Run the three §10C arms over one sampled determinant set.

    ``words`` are the retained QSCI configurations as occupation words, in the
    order the caller declares -- that order defines configuration-space
    locality for the packet arm, and the Haar construction does not infer it.
    Phase 11C owns the ordering ablations; here the ordering is simply recorded.

    Every arm is given the same ``max_size`` budget, but an arm may stop short
    of it -- its pool can exhaust, or its predicted lowering can fall below
    threshold -- so the budget is shared while the retained size need not be.
    Each arm records ``stopped_reason``, and a matched-size reading of §10D's
    claim has to check it: comparing a bare arm that ran out of candidates at
    two directions against a dressed arm that used all five is not a
    matched-size comparison, whatever the energies say.
    """
    words_array = np.asarray(words, dtype=np.int64).reshape(-1)
    if words_array.size == 0:
        raise ValueError("cannot run the hybrid from no sampled words")
    reference_occupancy = frozenset(occupied_spin_orbitals(model))
    reference_sampled = any(
        frozenset(_occupied(int(word), model.n)) == reference_occupancy
        for word in words_array.tolist())
    nonreference = [
        int(word) for word in words_array.tolist()
        if frozenset(_occupied(int(word), model.n)) != reference_occupancy
    ]
    configurations = (
        configuration_generators_from_words(model, words_array)
        if nonreference else [])
    # A-CASE seeds the identity when no `initial` is given, and the identity
    # direction *is* the reference determinant. QSCI sampling does not
    # guarantee the reference was observed, so seeding it unconditionally puts
    # a determinant nobody sampled into the "bare sampled configurations"
    # baseline -- and on a one-determinant sample the arm came back spanning
    # only that unsampled reference. Seed the identity exactly when the
    # reference was sampled, and otherwise start from a sampled direction.
    seed = None if reference_sampled else [configurations[0]]
    sampling_note = {"reference_sampled": bool(reference_sampled),
                     "sampled_words": int(np.unique(words_array).size)}
    family_configurations = list(configurations)
    if reference_sampled:
        # The reference determinant is a sampled configuration, so it belongs
        # in the family input. `dressed_family` drops it again whenever other
        # configurations exist, because `E_mu * I` is the bare excitation and
        # duplicating that arm inside the dressed family costs one product per
        # partner under a binding cap. It survives only for an all-reference
        # sample, where it is the only thing there is to dress.
        family_configurations.append(identity_generator(model.n))
    if family is None:
        family = dressed_family(family_configurations, model, kind=kind,
                                hamiltonian=model.hamiltonian,
                                max_support=max_support,
                                max_generators=max_generators)
    packets = (
        configuration_haar_packets(
            configurations, max_support=max_packet_support, label_prefix="cfgH")
        if configurations else [])
    common = dict(exact_ground_energy=exact_energy, gamma=gamma,
                  leakage_tol=leakage_tol)
    dressed_pool = list(configurations) + list(family.generators)
    arms = []

    started = time.perf_counter()
    if reference_sampled and not configurations:
        arms.append(_fixed_basis_arm(
            "bare_configurations", rho, model.hamiltonian,
            [identity_generator(model.n)],
            exact_energy=exact_energy, seconds=time.perf_counter() - started,
            metadata={"family": None, **sampling_note}))
    elif seed is not None and len(configurations) == 1:
        arms.append(_fixed_basis_arm(
            "bare_configurations", rho, model.hamiltonian, configurations,
            exact_energy=exact_energy, seconds=time.perf_counter() - started,
            metadata={"family": None, **sampling_note}))
    else:
        bare = run_acase(rho, model.hamiltonian, configurations, initial=seed,
                         max_size=max_size, **common)
        arms.append(_arm_from_result(
            "bare_configurations", bare, candidate_pool=len(configurations),
            exact_energy=exact_energy, seconds=time.perf_counter() - started,
            configuration_prefix="cfg[", packet_prefix="cfgH",
            metadata={"family": None, **sampling_note}))

    started = time.perf_counter()
    dressed = run_acase(rho, model.hamiltonian, dressed_pool, initial=seed,
                        max_size=max_size, **common)
    arms.append(_arm_from_result(
        "configurations_plus_dressed", dressed, candidate_pool=len(dressed_pool),
        exact_energy=exact_energy, seconds=time.perf_counter() - started,
        configuration_prefix="cfg[", packet_prefix="cfgH",
        metadata={**family.to_record(), **sampling_note}))

    if not packets:
        # No packet arm rather than a silently substituted one: an empty packet
        # family means the support cap admitted nothing, and reporting the
        # dressed arm under the packet name would fabricate a third comparison.
        return arms

    # `max_size` counts growth *steps* on top of the seeded basis, so a stage
    # of 1 performs one selection and can retain one packet. An earlier version
    # forced stage >= 2 and refused max_size < 3 on the belief that a stage of
    # 1 grew nothing; that was an off-by-one reading of the budget, and the
    # rejection encoded an invariant the solver does not have.
    stage = packet_stage if packet_stage is not None else max(1, max_size // 3)
    stage = max(1, min(stage, max_size))
    started = time.perf_counter()
    coarse = run_acase(rho, model.hamiltonian, list(configurations) + packets,
                       initial=seed, max_size=stage, **common)
    retained = [coarse.bank.generator(index) for index in coarse.indices]
    staged = run_acase(rho, model.hamiltonian, dressed_pool,
                       initial=retained, bank=coarse.bank,
                       max_size=max_size - len(retained) + 1, **common)
    arms.append(_arm_from_result(
        "packets_then_dressed", staged,
        candidate_pool=len(configurations) + len(packets) + len(dressed_pool),
        exact_energy=exact_energy, seconds=time.perf_counter() - started,
        configuration_prefix="cfg[", packet_prefix="cfgH",
        metadata={**family.to_record(), **sampling_note,
                  "packet_candidates": len(packets),
                  "packet_stage_size": int(stage),
                  "max_packet_support": max_packet_support}))
    return arms


def family_projection_report(rho, hamiltonian, family: FamilyReport,
                             baseline) -> dict:
    """§10B's projected columns: element support and *incremental* word universe.

    Incremental against ``baseline`` -- the configuration-only basis -- because
    the absolute universe of a dressed family says nothing about what dressing
    costs.  What a reader needs is the words the dressing adds beyond the
    directions that were already going to be measured.
    """
    baseline = list(as_generators(baseline))
    bank = MatrixElementBank(rho, hamiltonian)
    base_indices = tuple(bank.extend(baseline))
    base_resources = bank.solve(base_indices).resources
    full_indices = tuple(bank.extend(baseline + list(family.generators)))
    full_resources = bank.solve(full_indices).resources
    base_words = base_resources.get("word_universe") or 0
    full_words = full_resources.get("word_universe") or 0
    return {
        **family.to_record(),
        "baseline_word_universe": int(base_words),
        "dressed_word_universe": int(full_words),
        "incremental_word_universe": int(full_words - base_words),
        "max_element_support": full_resources.get(
            "max_hamiltonian_element_support"),
        "baseline_condition_number": float(bank.solve(base_indices).condition_number),
        "dressed_condition_number": float(bank.solve(full_indices).condition_number),
    }

"""G1 -- the per-filter marginals of PLAN.md section 3.5's pre-encoding chain.

Section 3.5 declares five filters that run before a fermion-to-qubit encoding is
chosen, and section 14 records that the whole section has "no quantitative
content whatsoever" until this record exists.  QG1 asks the question that
decides whether Track G exists at all: does pre-encoding algebraic restriction
remove candidates the package's existing post-encoding filters keep?

**The deliverable is the marginal, in the order applied, per pool.**  An
aggregate is not interpretable here.  Filters A-D have post-encoding analogues
in ``clifford_qc.subspace.symmetry`` that the package has applied since Phase 4,
so counting their removals as new content double-counts shipped machinery.
Filter E has no analogue anywhere in the tree, so it is reported separately and
is the only filter whose marginal can carry the section.

**Two pools, because a marginal is a property of a pool.**  The Majorana
monomial pool is section 3.5's own, and it is wide enough that A and B have
something to remove.  The determinant-excitation pool is the one the mapping and
cost records are actually built on, and the question G2 would answer -- whether
giving JW and BK the same GA-admissible domain changes the mapping comparison --
depends on what the chain does to *that* pool, not to the wide one.  Reporting a
marginal from the wide pool as though it applied to the narrow one would be the
central error available here, so both are measured and every statement names its
pool.

    python benchmarks/run_g1_structural_preconditioner.py
    python benchmarks/check_g1_structural_preconditioner.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from clifford_qc.backends import ExactMVBackend
from clifford_qc.models.metadata import spin_convention
from clifford_qc.fermion_mapping import fermion_encoding
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace.contracts import as_multivector
from clifford_qc.subspace.fermionic_generators import (
    determinant_excitations,
    occupied_spin_orbitals,
)
from clifford_qc.subspace.ga_restriction import (
    FILTER_ORDER,
    SectorCharacter,
    determinant_index,
    majorana_monomial_pool,
    reference_action,
    stabilizer_complexification_witness,
    structural_preconditioner,
)
from clifford_qc.subspace.symmetry import reference_sector_leakage

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    from benchmarks.run_mapping_axis import _build_model, load_config as load_mapping_config
except ImportError:  # pragma: no cover - direct script execution
    from run_mapping_axis import _build_model, load_config as load_mapping_config

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "g1_structural_preconditioner.json"
REFERENCE = HERE / "reference_results" / "g1_structural_preconditioner.json"
SCHEMA = "clifford_qc.g1_structural_preconditioner.v1"
CONFIG_SCHEMA = "clifford_qc.g1_structural_preconditioner_config.v1"

# The chain's own filter set. Declared here so a config that quietly drops a
# filter fails at the boundary rather than producing a shorter ledger that still
# looks complete.
REQUIRED_FILTERS = frozenset(FILTER_ORDER)

# Each gate the config must assert, and the field that has to say what asserting
# it means. A boolean alone is a flag someone can flip; the statement beside it
# is what a reviewer reads to see whether the flag still describes the gate.
GATE_STATEMENTS = {
    "abcd_reproduce_the_existing_pauli_filters": "abcd_gate_statement",
    "e_marginal_reported_separately": "e_marginal_gate_statement",
    "restriction_primitive_is_reused": "restriction_gate_statement",
    "no_cost_fields": "no_cost_fields_statement",
}


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported G1 config schema")

    gates = config.get("gates", {})
    for flag, statement in GATE_STATEMENTS.items():
        if gates.get(flag) is not True:
            raise ValueError(f"G1 may not relax its {flag} gate")
        if not gates.get(statement):
            raise ValueError(f"gate {flag} must carry its {statement}")

    pools = config.get("candidate_pools", {})
    if set(pools) != {"majorana_monomials", "determinant_excitations"}:
        raise ValueError(
            "G1 measures exactly two declared pools: the section 3.5 Majorana "
            "family and the excitation family the mapping records are built on"
        )
    majorana = pools["majorana_monomials"]
    if not isinstance(majorana.get("max_degree"), int) or majorana["max_degree"] < 2:
        raise ValueError("the Majorana pool needs an integer max_degree of at least 2")
    if not majorana.get("max_degree_basis"):
        raise ValueError("the Majorana degree cap must declare its reason")
    excitations = pools["determinant_excitations"]
    if excitations.get("conserve_sz") is not False:
        raise ValueError(
            "the excitation pool must be built with conserve_sz false; building it "
            "with the filter already applied would report a vacuous filter as a "
            "measured zero marginal"
        )

    systems = config.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError("G1 must declare a non-empty systems list")
    if len(set(systems)) != len(systems):
        raise ValueError("systems contains a duplicate")
    if not config.get("systems_basis"):
        raise ValueError("the declared instance set must state why it is that set")
    known = {spec["key"] for spec in load_mapping_config()["systems"]}
    missing = sorted(set(systems) - known)
    if missing:
        raise ValueError(f"systems {missing} have no frozen mapping-axis spec")

    sweep = config.get("degree_sweep")
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("G1 must declare a degree sweep")
    if majorana["max_degree"] not in sweep:
        raise ValueError(
            "the degree sweep must contain the pool's own cap, or the record "
            "cannot say where its comparison sits inside the sweep"
        )
    if not config.get("degree_sweep_basis"):
        raise ValueError("the degree sweep must declare what it bounds")

    probe = config.get("excited_character_probe", {})
    if "particle_number" not in probe or "sz" not in probe:
        raise ValueError(
            "section 3.5B requires the target character to be a parameter from "
            "the first commit; declare a non-reference character that exercises it"
        )
    if not probe.get("basis"):
        raise ValueError("the character probe must declare what it does and does not claim")

    if not config.get("claim_boundary"):
        raise ValueError("G1 must declare its claim boundary")
    if not config.get("restriction_arm", {}).get("encoding"):
        raise ValueError("filter C must declare the restriction arm it runs through")
    return config


def _instance(spec: dict):
    """Model, reference determinant, occupied set and restriction for one system."""
    model, construction = _build_model(spec)
    occupied = occupied_spin_orbitals(model)
    reference = ExactMVBackend().state(model.reference, ())
    return model, construction, occupied, reference


def _restriction_for(model, arm: str):
    encoding = fermion_encoding(
        arm,
        model.n,
        n_electrons=int(model.metadata["n_electrons"]),
        sz=float(model.metadata["sz"]),
        spin_ordering=spin_convention(model),
    )
    return encoding.restriction()


def _pauli_side_accepts(pool, reference, target, spin_ordering) -> list[str]:
    """The set the package's *existing* post-encoding machinery accepts.

    This is the independent path G1's first gate compares against, and it is
    deliberately the shipped function rather than a reimplementation: a gate
    that re-derived the Pauli side here would compare this module with itself.
    ``reference_sector_leakage`` raises on a candidate that annihilates the
    reference, which is the same rejection filter D makes on its first clause,
    so that exception is the accept/reject signal rather than an error.
    """
    accepted = []
    for candidate in pool:
        try:
            leakage = reference_sector_leakage(
                candidate.mv, reference,
                sector_target=(target.particle_number, target.sz),
                spin_ordering=spin_ordering,
            )
        except ValueError as exc:
            if "annihilates the reference" in str(exc):
                continue
            raise
        if leakage["target_sector"] == 0.0:
            accepted.append(candidate.label)
    return accepted


def _abcd_survivors(pool, report) -> list[str]:
    removed = {
        label
        for stage in report.stages
        if stage.key in ("A", "B", "D", "C")
        for label in stage.removed_labels
    }
    return [candidate.label for candidate in pool if candidate.label not in removed]


def _reached_determinants(pool, labels: Sequence[str], index: int) -> set[int]:
    operators = {candidate.label: candidate.mv for candidate in pool}
    reached: set[int] = set()
    for label in labels:
        reached |= set(reference_action(operators[label], index))
    return reached


def build_pool_record(
    pool,
    *,
    pool_name: str,
    model,
    occupied,
    reference,
    restriction,
    spin_ordering: str,
    run_gate_one: bool,
) -> dict:
    report = structural_preconditioner(
        pool,
        reference_occupied=occupied,
        spin_ordering=spin_ordering,
        restriction=restriction,
        hamiltonian=as_multivector(model.hamiltonian),
        reference_state=reference,
    )
    payload = report.as_dict()
    payload["pool"] = pool_name

    stages = {stage.key: stage for stage in report.stages}
    if set(stages) != REQUIRED_FILTERS:
        raise AssertionError("the chain did not run every declared filter")

    payload["e_marginal"] = {
        "entered": int(stages["E"].entered),
        "removed": int(stages["E"].removed),
        "survived": int(stages["E"].survived),
        "distinct_pauli_words_entering": int(report.distinct_pauli_words),
        "removal_is_not_word_deduplication": bool(
            report.distinct_pauli_words == stages["E"].entered
        ),
        "basis": (
            "Every candidate entering E is a distinct Pauli word under the "
            "package's own scalar_free_key identity when "
            "removal_is_not_word_deduplication is true, so no removal here is a "
            "duplicate word being collapsed. E is quotienting the physical "
            "action on the reference, which is what section 3.5E claims and what "
            "no existing code path performs."
        ),
        "mechanism": (
            "On the Majorana pool the collapse is not mysterious and should not "
            "be reported as though it were. i gamma_2p gamma_2p+1 = 2 n_p - 1 is "
            "a stabilizer of any determinant, so a monomial and that monomial "
            "times an occupation stabilizer drive the reference in the same "
            "direction while being different Pauli words -- and a degree-4 pool "
            "contains both. That is precisely section 3.5E's 'A and A S_alpha "
            "generate the same physical direction', and it is why the marginal "
            "is large here and zero on a pool whose builder never emits the "
            "stabilizer multiples. The number measures the pool's redundancy, "
            "not a compression of anything the project measures."
        ),
    }

    if run_gate_one:
        target = report.target_character
        ga = sorted(_abcd_survivors(pool, report))
        pauli = sorted(_pauli_side_accepts(pool, reference, target, spin_ordering))
        payload["gate_abcd_versus_existing_pauli_filters"] = {
            "ga_survivors": len(ga),
            "pauli_side_accepts": len(pauli),
            "agree": ga == pauli,
            "ga_only": [label for label in ga if label not in set(pauli)],
            "pauli_only": [label for label in pauli if label not in set(ga)],
            "pauli_side_path": (
                "clifford_qc.subspace.symmetry.reference_sector_leakage == 0, "
                "with the annihilated-reference exception counted as a rejection"
            ),
        }
        # The measured form of section 3.5B's warning. The global centralizer is
        # sufficient and not necessary, so using it as the filter would reject
        # candidates the shipped reference-aware path accepts -- and the gate
        # above would fail. The number says how load-bearing that choice was.
        payload["global_centralizer_contrast"] = {
            "global_commutant_survivors": len(report.global_centralizer_survivors),
            "reference_conditioned_survivors": len(ga),
            "candidates_a_global_test_would_wrongly_reject": len(ga) - len(
                set(report.global_centralizer_survivors) & set(ga)
            ),
            "global_is_a_subset": set(report.global_centralizer_survivors) <= set(ga),
            # When this is true the global test and the reference-conditioned one
            # coincide on this pool, so section 3.5B's distinction is unexercised
            # here. That is a property of the pool -- an excitation family built
            # against a determinant conserves N and S_z globally -- and it is also
            # why the chain agrees with the pool builder on it. The wide Majorana
            # pool is where the distinction has to bite, and the checker requires
            # it there rather than everywhere.
            "pool_lies_inside_the_global_commutant": set(ga) <= set(
                report.global_centralizer_survivors
            ),
            "basis": (
                "Section 3.5B: [A,Q]=0 is sufficient for sector preservation and "
                "not necessary. Hard-coding it would fail this phase's first gate "
                "by exactly this many candidates and would foreclose section 7.4's "
                "excited-state track, which needs a declared character rather than "
                "unconditional commutation."
            ),
        }
    return payload


def build_system_record(spec: dict, config: dict) -> dict:
    model, construction, occupied, reference = _instance(spec)
    spin_ordering = spin_convention(model)
    restriction = _restriction_for(model, config["restriction_arm"]["encoding"])
    pools = config["candidate_pools"]
    index = determinant_index(model.n, occupied)

    majorana_degree = int(pools["majorana_monomials"]["max_degree"])
    majorana = majorana_monomial_pool(model.n, max_degree=majorana_degree)
    excitations = determinant_excitations(
        model.n, occupied,
        max_rank=int(pools["determinant_excitations"]["max_rank"]),
        conserve_sz=False,
    )

    majorana_record = build_pool_record(
        majorana, pool_name="majorana_monomials", model=model, occupied=occupied,
        reference=reference, restriction=restriction, spin_ordering=spin_ordering,
        run_gate_one=True,
    )
    excitation_record = build_pool_record(
        excitations, pool_name="determinant_excitations", model=model,
        occupied=occupied, reference=reference, restriction=restriction,
        spin_ordering=spin_ordering, run_gate_one=True,
    )

    # The builder's own conserve_sz exclusion is a second, independent instance
    # of the first gate: filters A-D must land on exactly the set the shipped
    # pool builder keeps, which is what "reproduces the existing filters" means
    # for the pool the mapping records are actually built on.
    builder_kept = {
        candidate.label
        for candidate in determinant_excitations(
            model.n, occupied,
            max_rank=int(pools["determinant_excitations"]["max_rank"]),
            conserve_sz=True,
        )
    }
    abcd = set(_abcd_survivors(
        excitations,
        structural_preconditioner(
            excitations, reference_occupied=occupied, spin_ordering=spin_ordering,
            restriction=restriction, hamiltonian=as_multivector(model.hamiltonian),
            reference_state=reference),
    ))
    excitation_record["gate_abcd_versus_the_pool_builder"] = {
        "builder_keeps": len(builder_kept),
        "abcd_survivors": len(abcd),
        "agree": sorted(abcd) == sorted(builder_kept),
        "basis": (
            "determinant_excitations(conserve_sz=True) is the pool every mapping "
            "and cost record is built on. Filters A-D reproducing exactly that "
            "set is what says the pre-encoding chain agrees with a filter the "
            "package applies at construction time."
        ),
    }

    # What the two pools reach, at the matched cap. This is the comparison that
    # decides whether G2 -- rerunning the mapping producer on the G1-admissible
    # pool -- can differ from the record it would be compared against.
    ga_reach = _reached_determinants(
        majorana, majorana_record["surviving_labels"], index)
    builder_pool = determinant_excitations(
        model.n, occupied,
        max_rank=int(pools["determinant_excitations"]["max_rank"]),
        conserve_sz=True,
    )
    excitation_reach = {index} | _reached_determinants(
        builder_pool, [candidate.label for candidate in builder_pool], index)

    return {
        "system": spec["key"],
        "label": spec["label"],
        "construction": construction,
        "n_qubits": model.n,
        "reference_occupied": list(occupied),
        "reference_is_prefix_filled": list(occupied) == list(range(len(occupied))),
        "spin_ordering": spin_ordering,
        "restriction_arm": config["restriction_arm"]["encoding"],
        "pools": [majorana_record, excitation_record],
        "pool_correspondence_at_the_matched_cap": {
            "majorana_max_degree": majorana_degree,
            "excitation_max_rank": int(pools["determinant_excitations"]["max_rank"]),
            "determinants_reached_by_ga_survivors": len(ga_reach),
            "determinants_reached_by_identity_plus_excitations": len(excitation_reach),
            "same_set": ga_reach == excitation_reach,
            "ga_only": sorted(f"{i:0{model.n}b}" for i in ga_reach - excitation_reach),
            "excitation_only": sorted(
                f"{i:0{model.n}b}" for i in excitation_reach - ga_reach),
            "basis": (
                "The E-class representatives of the wide Majorana pool against "
                "the identity plus the excitation pool the mapping records use. "
                "Equality here means the pre-encoding chain reconstructs the pool "
                "the package already builds rather than a different or smaller "
                "one, which is the fact G2 turns on."
            ),
        },
    }


def build_degree_sweep(spec: dict, config: dict) -> dict:
    """Class count against the Majorana degree cap, on one declared instance.

    The sweep exists to bound the correspondence above. It is not a second
    experiment: a wider pool reaching more determinants is the pool being wider,
    not a filter finding more, and the record says so rather than leaving a
    reader to read a trend into it.
    """
    model, _, occupied, reference = _instance(spec)
    spin_ordering = spin_convention(model)
    restriction = _restriction_for(model, config["restriction_arm"]["encoding"])
    index = determinant_index(model.n, occupied)
    pools = config["candidate_pools"]

    builder_pool = determinant_excitations(
        model.n, occupied,
        max_rank=int(pools["determinant_excitations"]["max_rank"]),
        conserve_sz=True,
    )
    matched_reach = {index} | _reached_determinants(
        builder_pool, [candidate.label for candidate in builder_pool], index)

    rows = []
    for degree in config["degree_sweep"]:
        pool = majorana_monomial_pool(model.n, max_degree=int(degree))
        report = structural_preconditioner(
            pool, reference_occupied=occupied, spin_ordering=spin_ordering,
            restriction=restriction, hamiltonian=as_multivector(model.hamiltonian),
            reference_state=reference)
        stages = {stage.key: stage for stage in report.stages}
        reach = _reached_determinants(pool, report.surviving_labels, index)
        rows.append({
            "max_degree": int(degree),
            "pool_size": len(pool),
            "entered_E": int(stages["E"].entered),
            "action_equivalence_classes": int(stages["E"].survived),
            "determinants_reached": len(reach),
            "matches_the_rank_2_excitation_reach": reach == matched_reach,
        })
    return {
        "system": spec["key"],
        "sector_dimension": len(
            _sector_dimension(model, spin_ordering)
        ),
        "rows": rows,
        "basis": config["degree_sweep_basis"],
    }


def _sector_dimension(model, spin_ordering):
    from clifford_qc.backends.sector_statevector import sector_basis
    return sector_basis(
        model.n, int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]), spin_ordering=spin_ordering)


def build_character_probe(spec: dict, config: dict) -> dict:
    """Section 3.5B's parameter, exercised rather than asserted.

    The probe runs the same pool under a non-reference character twice: once
    through the declared restriction arm, and once with no restriction at all.
    The second arm is what isolates filter B, and the first is what makes the
    interaction between B and C visible instead of leaving it to be discovered
    by whoever opens section 7.4's track.
    """
    model, _, occupied, reference = _instance(spec)
    spin_ordering = spin_convention(model)
    restriction = _restriction_for(model, config["restriction_arm"]["encoding"])
    pool = majorana_monomial_pool(
        model.n, max_degree=int(config["candidate_pools"]["majorana_monomials"]["max_degree"]))
    probe = config["excited_character_probe"]
    declared = SectorCharacter(int(probe["particle_number"]), float(probe["sz"]))

    def run(character, arm):
        return structural_preconditioner(
            pool, reference_occupied=occupied, target_character=character,
            spin_ordering=spin_ordering, restriction=arm,
            hamiltonian=as_multivector(model.hamiltonian), reference_state=reference)

    ground = run(None, restriction)
    excited = run(declared, restriction)
    excited_unrestricted = run(declared, None)

    def after_b(report):
        stages = {stage.key: stage for stage in report.stages}
        return int(stages["B"].survived)

    return {
        "system": spec["key"],
        "reference_character": ground.target_character.as_dict(),
        "declared_character": declared.as_dict(),
        "survivors_after_B_reference_character": after_b(ground),
        "survivors_after_B_declared_character": after_b(excited),
        "b_stage_responds_to_the_declared_character": (
            after_b(ground) != after_b(excited)
        ),
        "final_survivors_reference_character": len(ground.surviving_labels),
        "final_survivors_declared_character_through_the_restriction_arm": len(
            excited.surviving_labels),
        "final_survivors_declared_character_without_a_restriction": len(
            excited_unrestricted.surviving_labels),
        "restriction_arm_annihilates_the_declared_character": (
            after_b(excited) > 0 and len(excited.surviving_labels) == 0
        ),
        "finding": (
            "The character is a live parameter at the filter that owns it: B "
            "admits a different population under the declared character than "
            "under the reference's. But the declared restriction arm fixes its "
            "stabilizer signs from the *reference* sector, so filter C then "
            "annihilates every candidate B admitted under a different character, "
            "and the chain returns an empty admissible pool rather than an error. "
            "The character and the restriction arm are therefore not independent "
            "declarations -- section 7.4's excited-state track needs both moved "
            "together, and a G1 that let them drift would close that track at "
            "filter C while section 3.5B's requirement at filter B still looked "
            "satisfied. Recorded here because it is cheaper to find at this gate "
            "than inside a phase built on top of it."
        ),
        "basis": probe["basis"],
    }


def build_record(config: dict | None = None) -> dict:
    config = load_config() if config is None else config
    specs = {spec["key"]: spec for spec in load_mapping_config()["systems"]}
    systems = [build_system_record(specs[key], config) for key in config["systems"]]

    sweep_system = config["systems"][0]
    sweep = build_degree_sweep(specs[sweep_system], config)
    probe = build_character_probe(specs[sweep_system], config)

    # Marginals that agree across instances are not evidence that the reference
    # was ignored: the three declared instances share one (N, S_z) sector on one
    # register, and every filter here is a function of that sector's structure.
    # What would be a defect is agreeing on the surviving *labels* across
    # different occupied sets, so the record carries the label sets per instance
    # and tests/test_g1_structural_preconditioner.py pins that they differ.
    gate_one = all(
        pool["gate_abcd_versus_existing_pauli_filters"]["agree"]
        for system in systems for pool in system["pools"]
    ) and all(
        pool["gate_abcd_versus_the_pool_builder"]["agree"]
        for system in systems for pool in system["pools"]
        if "gate_abcd_versus_the_pool_builder" in pool
    )
    # E's marginal per pool, across instances -- the split that is this record's
    # whole reading, surfaced at the top rather than left to be assembled from
    # the per-system entries.
    e_by_pool: dict[str, list[int]] = {}
    for system in systems:
        for pool in system["pools"]:
            e_by_pool.setdefault(pool["pool"], []).append(
                pool["e_marginal"]["removed"])
    e_by_pool = {name: sorted(set(values)) for name, values in e_by_pool.items()}
    e_removes_nothing_on_the_excitation_pool = all(
        pool["e_marginal"]["removed"] == 0
        for system in systems for pool in system["pools"]
        if pool["pool"] == "determinant_excitations"
    )
    e_removes_something_on_the_majorana_pool = all(
        pool["e_marginal"]["removed"] > 0
        for system in systems for pool in system["pools"]
        if pool["pool"] == "majorana_monomials"
    )
    pools_coincide = all(
        system["pool_correspondence_at_the_matched_cap"]["same_set"]
        for system in systems
    )

    record = {
        "schema": SCHEMA,
        "phase": config["phase"],
        "evidence_tier": "structural",
        "estimand": config["estimand"],
        "candidate_pools": config["candidate_pools"],
        "restriction_arm": config["restriction_arm"],
        "target_character": config["target_character"],
        "gates": config["gates"],
        "algebra_witnesses": _algebra_witnesses(systems),
        "systems": systems,
        "degree_sweep": sweep,
        "character_probe": probe,
        "why_instances_share_their_counts": (
            "The declared instances sit in one (N=4, S_z=0) sector on eight "
            "qubits, and every filter is a function of that sector's structure "
            "rather than of the Hamiltonian, so equal marginals across them are "
            "expected. The reference determinant does enter: beh2 and h4 share "
            "an occupied set and so share their surviving labels, while "
            "hubbard_2x2 fills (0,3,4,7) and its surviving label set differs. "
            "Equal counts with differing labels is the signature to look for; "
            "equal labels across different occupied sets would be the defect."
        ),
        "gate_outcomes": {
            "abcd_reproduce_the_existing_pauli_filters": gate_one,
            "e_marginal_reported_separately": True,
            "e_removes_nothing_on_the_excitation_pool": e_removes_nothing_on_the_excitation_pool,
            "e_removes_candidates_on_the_majorana_pool": e_removes_something_on_the_majorana_pool,
            "pools_coincide_at_the_matched_cap": pools_coincide,
            "e_marginal_by_pool": e_by_pool,
        },
        "qg1": _qg1_verdict(
            gate_one=gate_one,
            e_on_majorana=e_removes_something_on_the_majorana_pool,
            e_on_excitations=not e_removes_nothing_on_the_excitation_pool,
            pools_coincide=pools_coincide,
        ),
        "claim_boundary": config["claim_boundary"],
    }
    return stamp_record(record)


def _algebra_witnesses(systems: Sequence[dict]) -> dict:
    """Section 3.5's real-versus-complex argument, computed on each register."""
    return {
        "stabilizer_bilinear_squares": {
            str(system["n_qubits"]): [
                str(stabilizer_complexification_witness(system["n_qubits"], p))
                for p in range(system["n_qubits"])
            ]
            for system in systems
        },
        "basis": (
            "A product of two distinct Majorana generators squares to -1, so the "
            "occupation stabilizer n_p = (1 + i gamma_2p gamma_2p+1)/2 carries the "
            "i explicitly and the real Cl(2n,0) branch of section 3.5 is vacuous "
            "for the stabilizers this plan uses. Computed rather than asserted, so "
            "an implementation that reached for the real algebra would fail a check."
        ),
    }


def _qg1_verdict(*, gate_one: bool, e_on_majorana: bool, e_on_excitations: bool,
                 pools_coincide: bool) -> dict:
    """QG1's declared falsifier, evaluated against what the record measured.

    The falsifier is a conjunction -- A-D reproduce the existing filters AND E
    removes nothing -- so it fires only when both hold. It is evaluated here
    rather than written by hand, and ``check_g1_structural_preconditioner.py``
    re-derives it from the same fields.
    """
    fires = gate_one and not e_on_majorana and not e_on_excitations
    return {
        "falsifier_statement": (
            "Filters A-D agree with the existing post-encoding filters AND the "
            "action-equivalence quotient E removes nothing -- in which case "
            "section 3.5 is a reformulation and G2/G3 do not run."
        ),
        "falsifier_fires": fires,
        "verdict": (
            "falsified_track_G_stops" if fires
            else "not_falsified_but_content_is_pool_dependent"
        ),
        "what_was_measured": (
            "Filters A-D reproduce the existing post-encoding accept set exactly, "
            "on both pools and every declared instance, which is the agreement "
            "G1's first gate requires and not a finding. Filter E's marginal "
            "splits by pool: it removes candidates from the wide Majorana pool "
            "section 3.5 specifies, over entrants that are all distinct Pauli "
            "words, and it removes nothing from the excitation pool the mapping "
            "and cost records are built on. At the matched degree cap the "
            "surviving Majorana classes reach exactly the determinants the "
            "excitation pool reaches, so the chain reconstructs that pool rather "
            "than producing a different or smaller one."
        ),
        "what_this_does_not_license": (
            "No resource claim of any kind, and no statement that either pool is "
            "cheaper to measure. E's marginal on the Majorana pool is a property "
            "of that pool: it removes redundancy the excitation builder never "
            "creates, so it may not be reported as a reduction of the pool the "
            "mapping records use. Nothing here establishes a smaller Hilbert "
            "space, and the pre-encoding claim is only that no filter consults "
            "the encoding."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = build_record()
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for system in record["systems"]:
        print(f"{system['system']}  (n={system['n_qubits']})")
        for pool in system["pools"]:
            chain = " -> ".join(
                f"{stage['filter']}:{stage['survived']}" for stage in pool["stages"]
            )
            gate = pool["gate_abcd_versus_existing_pauli_filters"]
            print(f"  {pool['pool']:24s} {pool['pool_size']:6d} -> {chain}")
            print(f"    gate A-D vs existing Pauli filters: "
                  f"{'agree' if gate['agree'] else 'DISAGREE'} "
                  f"({gate['ga_survivors']} vs {gate['pauli_side_accepts']})")
            print(f"    E marginal: {pool['e_marginal']['removed']} removed of "
                  f"{pool['e_marginal']['entered']} entering, all distinct words: "
                  f"{pool['e_marginal']['removal_is_not_word_deduplication']}")
        correspondence = system["pool_correspondence_at_the_matched_cap"]
        print(f"  pools reach the same determinants at the matched cap: "
              f"{correspondence['same_set']} "
              f"({correspondence['determinants_reached_by_ga_survivors']})")
    print(f"QG1: {record['qg1']['verdict']} "
          f"(falsifier fires: {record['qg1']['falsifier_fires']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

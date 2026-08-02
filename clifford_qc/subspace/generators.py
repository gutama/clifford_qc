"""Generator families for the A-CASE basis hierarchy (research plan §4.2).

A generator ``A_i`` is an ``MV``; the basis state it names,
``|phi_i> = A_i|psi>``, is *never prepared*. Every projected matrix element is
an expectation on the single reference state,

    S_ij = <psi|A_i' A_j|psi>,      H_ij = <psi|A_i' H A_j|psi>,

so a generator is a label attached to a multivector, not a circuit. The levels
below are the ones the plan names; which of them a run uses is an explicit
input to :func:`~clifford_qc.subspace.solver.solve_subspace`, since Phase 1 is
fixed-basis (adaptive growth is Phase 3).

Grade is deliberately *not* a selection criterion here: it is not a good
quantum number for Jordan-Wigner-dressed Hamiltonians, so the families are
organized by operator response (Pauli orbit, commutator, Krylov), not by
Clifford grade.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

from ..ir import NamedClifford, PauliSum, PauliWord, Program
from ..measurement.bank import CommutatorBank
from ..multivector import MV


@dataclass(frozen=True)
class Generator:
    """One labelled basis direction ``A_i``, whose state ``A_i|psi>`` stays virtual."""

    label: str
    mv: MV

    @property
    def n(self) -> int:
        return self.mv.n

    def support(self) -> int:
        """Number of Pauli words in ``A_i`` -- the ``S_A`` term of the §6 accounting."""
        return self.mv.nnz()


def _to_mv(obj, n: int | None = None) -> MV:
    if isinstance(obj, MV):
        return obj
    if isinstance(obj, (PauliWord, PauliSum)):
        return obj.to_mv()
    if hasattr(obj, "word"):  # PoolOperator and friends
        return obj.word.to_mv()
    raise TypeError(f"cannot read a generator multivector from {type(obj).__name__}")


def _label_of(obj, index: int) -> str:
    for attr in ("label", "name"):
        value = getattr(obj, attr, None)
        if isinstance(value, str):
            return value
    return f"A{index}"


def as_generators(items: Iterable) -> list[Generator]:
    """Coerce a heterogeneous sequence into ``Generator``s.

    Accepts ``Generator``, ``MV``, ``PauliWord``, ``PauliSum``, ``PoolOperator``,
    and ``(label, one-of-those)`` pairs, so a caller can mix a hand-built
    multivector with a pool word without wrapping either by hand. Labels are
    what the result reports, so they must survive to the record.
    """
    out: list[Generator] = []
    for index, item in enumerate(items):
        if isinstance(item, Generator):
            out.append(item)
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            label, payload = item
            out.append(Generator(label, _to_mv(payload)))
            continue
        out.append(Generator(_label_of(item, index), _to_mv(item)))
    if not out:
        raise ValueError("no generators given")
    n = out[0].n
    if any(g.n != n for g in out):
        raise ValueError("generators live in different algebras")
    return out


def identity_generator(n: int) -> Generator:
    """Level 0: the reference state itself, ``A = 1``."""
    return Generator("I", MV.scalar(n, 1.0))


def pauli_orbit(words: Sequence) -> list[Generator]:
    """Level 1: ``P_j|psi>``, the tangent directions of the Pauli-rotor ansatz at theta=0.

    This is the *word-level benchmark mode* of §4.2 -- one generator per Pauli
    word, directly comparable with the qubit-ADAPT pools. For chemistry the
    symmetry-preserving mode (whole JW images of particle-number- and
    S_z-conserving excitations, kept as one multivector) is the physically
    meaningful default; see ``models.chemistry.excitation_multivectors``.
    """
    out = []
    for index, item in enumerate(words):
        word = item.word if hasattr(item, "word") else item
        label = getattr(item, "label", None) or getattr(word, "label", f"P{index}")
        out.append(Generator(label, _to_mv(word)))
    return out


def commutator_response(hamiltonian: PauliSum, words: Sequence,
                        *, drop_zero: bool = True) -> list[Generator]:
    """Level 2: ``G_j|psi>`` with ``G_j = -i/2 [H, P_j]``.

    The rows are exactly the ones ``CommutatorBank`` already materializes for
    ADAPT selection, reused here as basis directions rather than as scores --
    the same sparse operator, a different question asked of it. Its rows are
    ``-i sum_{k: {W_k,P}=0} h_k W_k P``, which is ``-i/2 [H, P_j]`` term by
    term; the overall scale is irrelevant to the subspace, which normalizes
    every generator before thresholding.

    ``drop_zero`` discards words that commute with ``H``: their response
    direction is the zero operator, which contributes an empty row rather than
    a direction.
    """
    bank = CommutatorBank(hamiltonian, [w.word if hasattr(w, "word") else w
                                        for w in words])
    out = []
    for label, row in zip(bank.labels, bank.coeffs):
        if drop_zero and not row:
            continue
        out.append(Generator(f"G[{label}]", MV(bank.n, dict(row))))
    return out


def occupied_spin_orbitals(model) -> tuple[int, ...]:
    """Spin orbitals the model's reference determinant fills.

    Read from the reference ``Program``'s X gates rather than assumed to be the
    first ``n_electrons`` indices. A molecular Hartree-Fock determinant does
    fill a prefix; a lattice Neel reference fills every other spin orbital, and
    building excitations against the wrong occupied set silently produces
    generators that annihilate the reference.
    """
    occupied = []
    for operation in model.reference.ops:
        if getattr(operation, "name", None) == "X":
            occupied.extend(operation.qubits)
    return tuple(sorted(occupied))


def determinant_excitations(n_qubits: int, occupied: Sequence[int], *,
                           max_rank: int = 2, conserve_sz: bool = True
                           ) -> list[Generator]:
    """Symmetry-preserving excitations of a determinant, built natively.

    Anti-Hermitian singles ``c†_a c_i - h.c.`` and doubles
    ``c†_a c†_b c_j c_i - h.c.`` between the given occupied set and its
    complement, assembled from the package's own Jordan-Wigner operators -- so
    the §4.2 symmetry-preserving mode is available for lattice models without
    the chemistry extra, and for any reference determinant rather than only a
    prefix-filled one.

    ``conserve_sz`` keeps only excitations that preserve total ``S_z`` under the
    interleaved convention (even indices up, odd down): a single needs matching
    spins, and a double needs the same number of down spins on each side. The
    weaker parity test ``(i+j) % 2 == (a+b) % 2`` admits ``Delta S_z = +-2``
    doubles and is not sufficient -- the same trap ``models.chemistry``
    documents.
    """
    from ..fermion import c_op, cdag_op

    occupied = tuple(sorted(occupied))
    virtual = tuple(p for p in range(n_qubits) if p not in set(occupied))
    if max_rank not in (1, 2):
        raise ValueError("max_rank must be 1 or 2")

    def anti_hermitian(creations: Sequence[int], annihilations: Sequence[int]) -> MV:
        term = MV.scalar(n_qubits, 1.0)
        for p in creations:
            term = term * cdag_op(n_qubits, p)
        for p in reversed(annihilations):
            term = term * c_op(n_qubits, p)
        return term - term.dagger()

    out: list[Generator] = []
    for i in occupied:
        for a in virtual:
            if conserve_sz and (a - i) % 2 != 0:
                continue
            out.append(Generator(f"E({a}<-{i})", anti_hermitian([a], [i])))
    if max_rank < 2:
        return [g for g in out if not g.mv.is_zero(1e-12)]
    for index, i in enumerate(occupied):
        for j in occupied[index + 1:]:
            for a_index, a in enumerate(virtual):
                for b in virtual[a_index + 1:]:
                    if conserve_sz and (i % 2 + j % 2) != (a % 2 + b % 2):
                        continue
                    out.append(Generator(f"E({a},{b}<-{i},{j})",
                                         anti_hermitian([a, b], [j, i])))
    return [g for g in out if not g.mv.is_zero(1e-12)]


def fermionic_excitation_generators(n_qubits: int, n_electrons: int) -> list[Generator]:
    """The chemistry default: whole JW images of conserving excitations (§4.2).

    One multivector per particle-number- and S_z-conserving fermionic
    excitation, *not* split into words. A split word generally leaves the
    physical sector, and a subspace built from split words can lower its Ritz
    value by leaking into states with the wrong electron count -- which is why
    this, rather than :func:`pauli_orbit` over the qubit-ADAPT pool, is what
    chemistry runs should grow from.

    Needs the ``chemistry`` extra (OpenFermion + PySCF), imported here rather
    than at module scope so the A-CASE layer stays importable without it.
    """
    from ..models.chemistry import excitation_multivectors
    return [Generator(label, image.to_mv())
            for label, image in excitation_multivectors(n_qubits, n_electrons)]


def krylov_response(hamiltonian, order: int) -> list[Generator]:
    """Level 3: ``H^k|psi>`` for ``k = 1..order``.

    One candidate family among several, not the organizing principle: the
    powers are kept because they are the reference construction every quantum
    Krylov method uses, so a run can price A-CASE against them under the same
    §6 accounting.
    """
    if order < 1:
        raise ValueError("Krylov order must be at least 1")
    H = hamiltonian if isinstance(hamiltonian, MV) else hamiltonian.to_mv()
    out = []
    power = MV.scalar(H.n, 1.0)
    for k in range(1, order + 1):
        power = power * H
        out.append(Generator(f"H^{k}", power.copy()))
    return out


def compound_response(left: Sequence, right: Sequence | None = None, *,
                      max_generators: int | None = None,
                      max_support: int | None = None,
                      drop_scalar: bool = True) -> list[Generator]:
    """Level 4a: the compound directions ``A_i A_j |psi>``.

    ``P_i P_j`` and ``P_i G_j`` in the plan's notation -- pass the Pauli orbit
    as ``left`` and either itself or the commutator response as ``right``. The
    product of two generators is another generator, so nothing new is needed
    downstream; what is new is that the family is *quadratic*, which is why it
    is filtered rather than enumerated.

    Three filters, all of them resource statements rather than physics:

    - ``drop_scalar`` discards products proportional to the identity. ``P P =
      I`` for any Pauli word, so the diagonal of a Pauli-orbit square is pure
      identity and contributes the level-0 direction the basis already has.
    - ``max_support`` caps ``S_A``. A compound generator is as wide as the
      product of its factors, and §6 counts that width; a cap makes the ceiling
      explicit instead of discovering it as an unaffordable bank.
    - ``max_generators`` truncates the surviving list in its deterministic
      order (``left`` major, ``right`` minor), so a run's candidate pool has a
      declared size rather than one that depends on how many products happened
      to survive.

    Products are deduplicated up to a scalar factor: ``P_i P_j`` and
    ``P_j P_i`` differ by a sign for anticommuting words, and the subspace
    normalizes every generator, so keeping both would buy a duplicate column
    and a singular overlap matrix.
    """
    left_gens = as_generators(left)
    right_gens = left_gens if right is None else as_generators(right)
    if not left_gens or not right_gens:
        return []
    n = left_gens[0].n
    for generator in (*left_gens, *right_gens):
        if generator.n != n:
            raise ValueError("compound factors act on different qubit counts")

    out: list[Generator] = []
    seen: set[tuple] = set()
    for a in left_gens:
        for b in right_gens:
            product = a.mv * b.mv
            if product.is_zero():
                continue
            if drop_scalar and product.nnz() == 1 and 0 in product.terms:
                continue
            if max_support is not None and product.nnz() > max_support:
                continue
            key = _scalar_free_key(product)
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(Generator(f"{a.label}*{b.label}", product))
            if max_generators is not None and len(out) >= max_generators:
                return out
    return out


def _scalar_free_key(mv: MV):
    """A hashable identity for ``mv`` up to an overall complex factor.

    Generators are normalized by the solver, so two multivectors differing by a
    scalar name the same basis direction and must not both enter the pool.
    """
    terms = [(code, value) for code, value in sorted(mv.terms.items())
             if abs(value) > 1e-15]
    if not terms:
        return None
    pivot = terms[0][1]
    return tuple((code, complex(round((value / pivot).real, 12),
                                round((value / pivot).imag, 12)))
                 for code, value in terms)


def determinant_program(n_qubits: int, occupied: Sequence[int]) -> Program:
    """``|occupied>`` as a Clifford preparation from ``|0...0>``."""
    for orbital in occupied:
        if not 0 <= int(orbital) < n_qubits:
            raise IndexError(f"spin orbital {orbital} out of range for {n_qubits}")
    return Program(n_qubits, [NamedClifford("X", (int(orbital),))
                              for orbital in sorted(set(int(o) for o in occupied))])


def configuration_generator(reference, target, *, label: str) -> Generator:
    """Level 4b: the generator carrying the reference onto another configuration.

    The plan asks for ``stabilizer configurations V|0...0>`` as basis
    directions for competing orders. A-CASE's basis states are all of the form
    ``A|psi>`` for one fixed reference, so a configuration enters as the
    operator that *reaches* it: with ``|psi> = R|0...0>`` and ``|phi> =
    V|0...0>``, the generator is ``A = V R^dagger``, giving ``A|psi> = |phi>``
    exactly. No second state is ever prepared -- the projected elements remain
    expectations on ``|psi>``.

    For two determinants this is a single Pauli word (the ``X``-string on the
    orbitals whose occupation differs), so a whole competing order costs
    ``S_A = 1``, which is why these are cheap to carry beside the excitation
    family rather than an alternative to it.

    ``reference`` and ``target`` are ``Program``s, or occupation sequences
    interpreted as determinants.
    """
    reference_program = (reference if isinstance(reference, Program)
                         else determinant_program(len(reference) and
                                                  max(reference) + 1, reference))
    if not isinstance(target, Program):
        target = determinant_program(reference_program.n, target)
    if target.n != reference_program.n:
        raise ValueError("configuration and reference act on different qubit counts")
    return Generator(label, target.unitary() * reference_program.unitary().dagger())


def configuration_generators(model, configurations) -> list[Generator]:
    """``configuration_generator`` over a mapping of named configurations."""
    return [configuration_generator(model.reference, configuration, label=name)
            for name, configuration in sorted(dict(configurations).items())]


def configuration_haar_packets(configurations: Sequence, *,
                               min_support: int = 2,
                               max_support: int | None = 16,
                               include_scaling: bool = False,
                               label_prefix: str = "cfgH") -> list[Generator]:
    """Support-pruned tree-Haar packets over an *ordered* configuration list.

    The list order defines configuration-space locality.  The routine does not
    pretend to infer that physics: a Hubbard caller can order determinants by
    excitation rank, doublon count, and charge/spin pattern, while a chemistry
    caller can choose a different hierarchy.  A balanced binary tree is then
    built over those ordered leaves.  At every (possibly unbalanced) split,
    the normalized scaling vectors of the two children are combined into one
    zero-mean Haar detail,

    ``sqrt(n_r/n) s_l - sqrt(n_l/n) s_r``.

    The complete coefficient transform (all details plus the root scaling
    vector) is exactly orthogonal for any finite leaf count, not just powers of
    two.  Therefore, when the supplied ``A_i|psi>`` are distinct orthonormal
    configuration states, the unpruned packets preserve ``S = I`` and Parseval
    exactly up to floating-point error.  This is a classical change of basis
    among virtual A-CASE generators, not a quantum wavelet circuit.

    ``max_support`` caps the actual Pauli-word support ``S_A`` after combining
    leaves.  Pruning makes the returned family incomplete by design: it is an
    opt-in coarse candidate tier, not a replacement for the convergence-complete
    level-4 family.  ``include_scaling`` requests the root average as well; the
    same support bounds still apply to it.
    """
    if min_support < 1:
        raise ValueError("min_support must be at least 1")
    if max_support is not None and max_support < min_support:
        raise ValueError("max_support must be at least min_support")
    if not isinstance(label_prefix, str) or not label_prefix:
        raise ValueError("label_prefix must be a non-empty string")

    leaves = as_generators(configurations)
    keys = [_scalar_free_key(generator.mv) for generator in leaves]
    if len(set(keys)) != len(keys):
        raise ValueError("configuration generators must name distinct directions")

    # A row is a sparse coefficient vector over the ordered leaves.  Returning
    # child details after the parent gives deterministic coarse-to-fine order.
    def tree(lo: int, hi: int):
        if hi - lo == 1:
            return {lo: 1.0}, []
        mid = lo + (hi - lo) // 2
        left, left_details = tree(lo, mid)
        right, right_details = tree(mid, hi)
        n_left, n_right = mid - lo, hi - mid
        total = n_left + n_right
        parent = {index: sqrt(n_left / total) * value
                  for index, value in left.items()}
        parent.update({index: sqrt(n_right / total) * value
                       for index, value in right.items()})
        detail = {index: sqrt(n_right / total) * value
                  for index, value in left.items()}
        detail.update({index: -sqrt(n_left / total) * value
                       for index, value in right.items()})
        return parent, [(lo, hi, detail)] + left_details + right_details

    scaling, details = tree(0, len(leaves))
    rows = details + ([(0, len(leaves), scaling)] if include_scaling else [])
    out: list[Generator] = []
    for lo, hi, row in rows:
        terms: dict[int, complex] = {}
        for index, coefficient in row.items():
            for code, value in leaves[index].mv.terms.items():
                terms[code] = terms.get(code, 0.0) + coefficient * value
        mv = MV(leaves[0].n, terms)
        support = mv.nnz()
        if support < min_support:
            continue
        if max_support is not None and support > max_support:
            continue
        kind = "S" if row is scaling else "H"
        out.append(Generator(f"{label_prefix}{kind}[{lo}:{hi})", mv))
    return out


def state_sector(generator, reference_state) -> dict[str, float]:
    """``(N, S_z)`` of ``A|psi>`` and how sharply they are defined.

    The operator-level test ``sector_leakage`` asks whether ``A`` commutes with
    the symmetry, which is the right question for an excitation generator and
    the *wrong* one for a configuration generator: an ``X``-string does not
    commute with ``N``, yet it maps one determinant of fixed particle number
    onto another. What matters for a configuration is the plan's own wording --
    the sector of the configuration it names -- so this reports the sector of
    the state instead.

    Both are expectations on the reference, so nothing is prepared:
    ``<N>_A = <psi|A' N A|psi> / <psi|A' A|psi>``, and the variance of ``N`` in
    the same state says whether that number is a sharp quantum number or an
    average over a superposition of sectors.
    """
    from .adaptive import _sector_operators

    A = generator.mv if isinstance(generator, Generator) else _to_mv(generator)
    rho = reference_state
    norm = (A.dagger() * A * rho).trace().real
    if norm <= 1e-15:
        raise ValueError("generator annihilates the reference state")
    out = {}
    for name, operator in zip(("particle_number", "sz"), _sector_operators(A.n)):
        first = (A.dagger() * operator * A * rho).trace().real / norm
        second = (A.dagger() * operator * operator * A * rho).trace().real / norm
        out[name] = first
        out[f"{name}_variance"] = max(0.0, second - first * first)
    return out


def response_hierarchy(hamiltonian: PauliSum, words: Sequence, *,
                       krylov_order: int = 0, include_identity: bool = True,
                       include_pauli: bool = True,
                       include_commutator: bool = True,
                       compound: bool = False,
                       max_compound: int | None = None,
                       configurations=None, model=None) -> list[Generator]:
    """Levels 0-4 stacked in order, deduplicated by nothing -- the caller's choice.

    Convenience for the fixed-basis studies: the nested prefixes of the
    returned list are the nested bases whose Ritz values must be monotone
    non-increasing. Levels 0-3 are on by default; level 4 is opt-in, because it
    is the only quadratic family here and its width is a §6 cost.
    """
    out: list[Generator] = []
    if include_identity:
        out.append(identity_generator(hamiltonian.n))
    if include_pauli:
        out.extend(pauli_orbit(words))
    if include_commutator:
        out.extend(commutator_response(hamiltonian, words))
    if krylov_order:
        out.extend(krylov_response(hamiltonian, krylov_order))
    if compound:
        orbit = pauli_orbit(words)
        out.extend(compound_response(orbit, orbit, max_generators=max_compound))
    if configurations:
        if model is None:
            raise ValueError("configurations need the model whose reference they "
                             "are measured against")
        out.extend(configuration_generators(model, configurations))
    return out

"""Determinant configurations and configuration-space Haar packets."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

from ..ir import NamedClifford, Program
from ..multivector import MV
from .generator_core import Generator, _scalar_free_key, _to_mv, as_generators


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
                               label_prefix: str = "cfg") -> list[Generator]:
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
    from .symmetry import sector_operators

    A = generator.mv if isinstance(generator, Generator) else _to_mv(generator)
    rho = reference_state
    norm = (A.dagger() * A * rho).trace().real
    if norm <= 1e-15:
        raise ValueError("generator annihilates the reference state")
    out = {}
    for name, operator in zip(("particle_number", "sz"), sector_operators(A.n)):
        first = (A.dagger() * operator * A * rho).trace().real / norm
        second = (A.dagger() * operator * operator * A * rho).trace().real / norm
        out[name] = first
        out[f"{name}_variance"] = max(0.0, second - first * first)
    return out

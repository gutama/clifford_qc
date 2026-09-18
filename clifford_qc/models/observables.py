"""Material observables as ``PauliSum``s, for the projected-matrix route (§8).

A-CASE never stores a Ritz state, so every observable is answered by projecting
the operator into the subspace -- ``Q_sub[i,j] = <psi|A_i' Q A_j|psi>`` -- and
contracting with the Ritz coefficients. These builders are the ``Q`` side of
that: densities, double occupancy, spin correlations, and the structure factors
that say whether a cluster is ordered.

They read the site/orbital metadata the lattice models carry, so a caller asks
for "the double occupancy of site 2" rather than juggling spin-orbital indices,
and the same call works for a chain, a 2x2 cluster, or a multi-orbital model.
Spin lattices (one qubit per site) and fermionic lattices (two spin orbitals
per site per orbital) need different operators for the same physical quantity,
which is what ``model.metadata['kind']`` selects.
"""

from __future__ import annotations

from typing import Sequence

from ..fermion import c_op, cdag_op
from ..ir import PauliSum
from ..multivector import MV
from ..pauli import X, Y, Z
from .lattice import SPIN_DOWN, SPIN_UP, spin_orbital
from .metadata import ANDERSON_IMPURITY, FERMIONIC_LATTICE, SPIN_LATTICE

#: The two operator families these builders distinguish. They are the kind
#: constants themselves rather than copies of the strings, so the vocabulary
#: has one definition -- adding a kind is an edit to ``models.metadata``, not a
#: grep across the modules that consume it.
FERMIONIC = FERMIONIC_LATTICE
SPIN = SPIN_LATTICE

#: Kinds that carry site/orbital metadata, mapped to the family whose operators
#: apply to them. A kind absent from this table is a fermionic or spin model
#: that is simply not on a lattice -- ``molecular``, ``fermionic_orbital_basis``
#: -- and has no site index for a caller to ask about.
_LATTICE_KINDS = {
    FERMIONIC_LATTICE: FERMIONIC,
    ANDERSON_IMPURITY: FERMIONIC,
    SPIN_LATTICE: SPIN,
}


def _kind(model) -> str:
    kind = model.metadata.get("kind", "")
    try:
        return _LATTICE_KINDS[kind]
    except KeyError:
        raise ValueError(
            f"model {model.name!r} carries no lattice metadata; observables "
            f"need 'kind' in {sorted(_LATTICE_KINDS)}, 'sites', and (for "
            f"fermionic models) 'n_orbitals', but kind is {kind!r}") from None


def _as_sum(mv: MV) -> PauliSum:
    return PauliSum(mv.n, {code: complex(coeff.real)
                           for code, coeff in mv.terms.items()
                           if abs(coeff.real) > 1e-14})


def _orbital_index(model, site: int, spin: int, orbital: int) -> int:
    return spin_orbital(site, spin, orbital=orbital,
                        n_orbitals=model.metadata.get("n_orbitals", 1))


def _number(model, site: int, spin: int, orbital: int = 0) -> MV:
    index = _orbital_index(model, site, spin, orbital)
    return cdag_op(model.n, index) * c_op(model.n, index)


def occupation(model, site: int, *, spin: int | None = None,
               orbital: int = 0) -> PauliSum:
    """``n_i`` (both spins) or ``n_is`` for one spin, on a fermionic lattice.

    On a spin lattice there are no electrons to count; the analogous local
    observable is :func:`magnetization`.
    """
    if _kind(model) != FERMIONIC:
        raise ValueError("occupation is a fermionic-lattice observable; "
                         "use magnetization on a spin lattice")
    if spin is None:
        return _as_sum(_number(model, site, SPIN_UP, orbital)
                       + _number(model, site, SPIN_DOWN, orbital))
    return _as_sum(_number(model, site, spin, orbital))


def double_occupancy(model, site: int | None = None, *, orbital: int = 0) -> PauliSum:
    """``n_iu n_id`` at one site, or its average over sites.

    The order parameter of the Mott transition, and the observable that says
    whether ``U`` is doing anything: it drops from 1/4 at ``U = 0`` toward 0 in
    the strongly correlated limit.
    """
    if _kind(model) != FERMIONIC:
        raise ValueError("double occupancy is a fermionic-lattice observable")
    sites = [site] if site is not None else range(model.metadata["sites"])
    total = MV(model.n)
    for s in sites:
        total = total + (_number(model, s, SPIN_UP, orbital)
                         * _number(model, s, SPIN_DOWN, orbital))
    scale = 1.0 / len(list(sites)) if site is None else 1.0
    return _as_sum(scale * total)


def spin_operators(model, site: int, *, orbital: int = 0) -> tuple[MV, MV, MV]:
    """``(S_x, S_y, S_z)`` at a site, in whichever representation the model uses.

    Fermionic: the bilinears ``S_z = (n_u - n_d)/2``,
    ``S_x = (c†_u c_d + c†_d c_u)/2``, ``S_y = -i(c†_u c_d - c†_d c_u)/2``.
    Spin lattice: the Pauli operators halved.
    """
    if _kind(model) == SPIN:
        return (0.5 * X(model.n, site), 0.5 * Y(model.n, site),
                0.5 * Z(model.n, site))
    up = _orbital_index(model, site, SPIN_UP, orbital)
    down = _orbital_index(model, site, SPIN_DOWN, orbital)
    flip = cdag_op(model.n, up) * c_op(model.n, down)
    return (0.5 * (flip + flip.dagger()),
            (-0.5j) * (flip - flip.dagger()),
            0.5 * (_number(model, site, SPIN_UP, orbital)
                   - _number(model, site, SPIN_DOWN, orbital)))


def magnetization(model, site: int, *, axis: str = "z", orbital: int = 0) -> PauliSum:
    """``S_a`` at one site."""
    components = dict(zip("xyz", spin_operators(model, site, orbital=orbital)))
    if axis not in components:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return _as_sum(components[axis])


def spin_correlation(model, i: int, j: int, *, axis: str | None = None,
                     orbital: int = 0) -> PauliSum:
    """``S_i . S_j`` (all three components) or one component ``S_i^a S_j^a``.

    The observable the Kitaev and Hubbard clusters are actually interesting
    for: it distinguishes a Neel-ordered cluster from a spin liquid, where
    only the nearest-neighbour correlation on a given link type survives.
    """
    left = spin_operators(model, i, orbital=orbital)
    right = spin_operators(model, j, orbital=orbital)
    if axis is not None:
        index = "xyz".index(axis)
        return _as_sum(left[index] * right[index])
    total = MV(model.n)
    for a, b in zip(left, right):
        total = total + a * b
    return _as_sum(total)


def structure_factor(model, momentum: Sequence[float] | None = None, *,
                     axis: str = "z", orbital: int = 0) -> PauliSum:
    """``S(q) = (1/N) sum_ij exp(i q.(r_i - r_j)) S_i^a S_j^a`` on the site grid.

    ``momentum`` defaults to the antiferromagnetic point ``(pi, pi)``, whose
    structure factor is the Neel order parameter. Sites are placed on the
    ``rows x cols`` grid the model records; the phases are real for the
    high-symmetry momenta, and the operator is symmetrized so it stays
    Hermitian at any ``q``.
    """
    import math

    metadata = model.metadata
    rows = metadata.get("rows", 1)
    cols = metadata.get("cols", metadata["sites"])
    sites = metadata["sites"]
    q = (math.pi, math.pi) if momentum is None else tuple(float(x) for x in momentum)

    def position(site: int) -> tuple[int, int]:
        return divmod(site, cols) if rows > 1 else (0, site)

    index = "xyz".index(axis)
    total = MV(model.n)
    for i in range(sites):
        ri = position(i)
        for j in range(sites):
            rj = position(j)
            phase = math.cos(q[0] * (ri[0] - rj[0]) + q[1] * (ri[1] - rj[1]))
            if phase == 0.0:
                continue
            si = spin_operators(model, i, orbital=orbital)[index]
            sj = spin_operators(model, j, orbital=orbital)[index]
            total = total + (phase / sites) * (si * sj)
    return _as_sum(total)


def link_correlations(model) -> dict[str, PauliSum]:
    """Per-link-type bond operators of a Kitaev cluster, averaged over links.

    ``{'x': <X_iX_j>_x-links, ...}``: in the Kitaev spin liquid these are the
    only nonvanishing spin correlations, so the three numbers are the compact
    signature the adaptive basis has to reproduce.
    """
    if _kind(model) != SPIN:
        raise ValueError("link correlations are defined for the spin lattice models")
    # A chain spin model is a spin lattice and clears the guard above, but it
    # has no typed links -- only a Kitaev cluster labels its bonds x/y/z. Before
    # the metadata contract, tfim/xxz/random_ising carried no ``kind`` at all
    # and were turned away by ``_kind``; now that they declare one, this is the
    # check that has to say so, rather than letting a bare KeyError out.
    if "links" not in model.metadata:
        raise ValueError(
            f"model {model.name!r} is a {model.metadata.get('lattice')!r} "
            "spin lattice with no typed links; link correlations need the "
            "per-link x/y/z labelling that kitaev_honeycomb records")
    letters = {"x": X, "y": Y, "z": Z}
    grouped: dict[str, MV] = {}
    counts: dict[str, int] = {}
    for kind, i, j in model.metadata["links"]:
        operator = letters[kind](model.n, i) * letters[kind](model.n, j)
        grouped[kind] = grouped.get(kind, MV(model.n)) + operator
        counts[kind] = counts.get(kind, 0) + 1
    return {kind: _as_sum((1.0 / counts[kind]) * total)
            for kind, total in grouped.items()}


def total_spin_squared(model, *, orbital: int = 0) -> PauliSum:
    """``S^2 = (sum_i S_i)^2`` -- the spin quantum number of a Ritz state.

    Useful as a projected observable precisely because A-CASE's subspace is not
    guaranteed to be a spin eigenspace: ``<S^2>`` says whether a low Ritz value
    was bought by mixing multiplicities.
    """
    sites = model.metadata["sites"]
    components = [spin_operators(model, site, orbital=orbital) for site in range(sites)]
    total = MV(model.n)
    for axis in range(3):
        summed = MV(model.n)
        for site in range(sites):
            summed = summed + components[site][axis]
        total = total + summed * summed
    return _as_sum(total)

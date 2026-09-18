"""Materials models: the correlated subproblem downfolding hands over.

The §1 scope of ``PLAN.md`` in code. These are not materials
simulations; they are the small-but-hard clusters an embedding or downfolding
step produces -- Hubbard and extended Hubbard, multi-orbital Kanamori, a small
Anderson impurity, and a Kitaev honeycomb patch -- expressed in the same
``Model`` the spin benchmarks use.

Two families with genuinely different structure:

*Fermionic lattices* are built from the package's own Jordan-Wigner
creation/annihilation operators, so no external chemistry stack is involved.
Spin-orbitals use the interleaved convention the chemistry module and
``fermion.total_sz_op`` already assume: site ``i``, orbital ``m``, spin ``s``
maps to ``2*(i*n_orbitals + m) + s`` with ``s = 0`` up and ``s = 1`` down. Every
Hamiltonian is assembled as ``h + h†`` where the hopping part is written once,
so hermiticity is structural rather than checked afterwards.

*Spin lattices* (Kitaev) need no transformation at all: the Hamiltonian is
already a sum of Pauli words, one qubit per site.

Every model carries site/orbital/sector metadata, because a projected
observable (§8) has to know which qubits are which -- ``models/observables.py``
reads exactly these keys.
"""

from __future__ import annotations

from typing import Sequence

from ..fermion import c_op, cdag_op
from ..ir import PauliSum, Program
from ..multivector import MV
from .metadata import (ANDERSON_IMPURITY, FERMIONIC_LATTICE, SPIN_LATTICE,
                       model_metadata)
from .spin import Model

SPIN_UP, SPIN_DOWN = 0, 1


def spin_orbital(site: int, spin: int, *, orbital: int = 0,
                 n_orbitals: int = 1) -> int:
    """Interleaved index of one spin orbital: ``2*(site*n_orbitals + orbital) + spin``."""
    if spin not in (SPIN_UP, SPIN_DOWN):
        raise ValueError("spin must be 0 (up) or 1 (down)")
    if not (0 <= orbital < n_orbitals):
        raise ValueError(f"orbital must be in [0, {n_orbitals})")
    return 2 * (site * n_orbitals + orbital) + spin


def _hop(n: int, p: int, q: int) -> MV:
    """``c†_p c_q`` as a multivector."""
    return cdag_op(n, p) * c_op(n, q)


def _number(n: int, p: int) -> MV:
    return cdag_op(n, p) * c_op(n, p)


def _hermitize(n: int, one_way: MV, diagonal: MV) -> PauliSum:
    """``one_way + one_way† + diagonal`` as a real-coefficient ``PauliSum``.

    The hopping half is written once and added to its own adjoint, so the result
    is Hermitian by construction; the diagonal (number-operator) part is already
    self-adjoint. Coefficients are then real to round-off, and the check below
    is a guard against an index slip, not a numerical repair.
    """
    total = one_way + one_way.dagger() + diagonal
    if not total.is_hermitian(1e-12):  # pragma: no cover - structural
        raise ValueError("lattice Hamiltonian came out non-Hermitian")
    terms = {code: complex(coeff.real) for code, coeff in total.terms.items()
             if abs(coeff.real) > 1e-14}
    return PauliSum(n, terms)


def _rectangle_bonds(rows: int, cols: int, *, periodic: bool) -> list[tuple[int, int]]:
    """Nearest-neighbour bonds of a rows x cols grid, row-major site indices."""
    bonds = []
    for r in range(rows):
        for c in range(cols):
            site = r * cols + c
            if c + 1 < cols:
                bonds.append((site, r * cols + c + 1))
            elif periodic and cols > 2:
                bonds.append((site, r * cols))
            if r + 1 < rows:
                bonds.append((site, (r + 1) * cols + c))
            elif periodic and rows > 2:
                bonds.append((site, c))
    return bonds


def _shape(shape) -> tuple[int, int]:
    if isinstance(shape, int):
        return 1, shape
    rows, cols = shape
    return int(rows), int(cols)


def _half_filled_reference(n_qubits: int, sites: int, *,
                           n_orbitals: int = 1) -> Program:
    """Neel-like product state: up on even sites, down on odd ones.

    One electron per site, alternating spin -- the strong-coupling ground state
    of the half-filled Hubbard model, and a determinant in the ``N = sites``,
    ``S_z = 0`` sector for an even site count. A reference in the right sector
    matters more here than a good energy: the response generators are built on
    top of it.
    """
    program = Program(n_qubits)
    for site in range(sites):
        spin = SPIN_UP if site % 2 == 0 else SPIN_DOWN
        for orbital in range(n_orbitals):
            if orbital == 0:
                program.clifford("X", spin_orbital(site, spin, orbital=orbital,
                                                  n_orbitals=n_orbitals))
    return program


def reference_sector(program: Program) -> tuple[int, float]:
    """``(N, S_z)`` of the determinant an X-gate reference prepares.

    Read from the gates rather than assumed. An odd-site cluster at half filling
    has ``S_z = +-1/2``, not 0, and hardcoding zero would advertise an empty
    sector -- which is exactly what the Phase-6 backend refuses to build, and how
    this was caught.
    """
    occupied = []
    for operation in program.ops:
        if getattr(operation, "name", None) != "X":
            raise ValueError("reference is not an X-gate determinant")
        occupied.extend(operation.qubits)
    spin = sum(0.5 if q % 2 == 0 else -0.5 for q in occupied)
    return len(occupied), float(spin)


def _fermionic_metadata(rows: int, cols: int, bonds, n_orbitals: int,
                        couplings: dict, reference: Program) -> dict:
    """Contract-satisfying metadata for a fermionic lattice builder.

    ``n_orbitals`` is the count *per site*, so the total spatial-orbital count
    the contract asks for is ``sites * n_orbitals``.  Both are recorded: the
    per-site number is what ``spin_orbital`` needs to compute an index, and the
    total is what a consumer comparing against a molecular model needs.
    """
    sites = rows * cols
    electrons, sz = reference_sector(reference)
    return model_metadata(
        FERMIONIC_LATTICE,
        spin_orbitals=2 * sites * n_orbitals,
        n_spatial_orbitals=sites * n_orbitals,
        n_electrons=electrons,
        sz=sz,
        spin_convention="interleaved",  # 2*(site*n_orbitals+orbital)+spin
        sites=sites,
        rows=rows,
        cols=cols,
        n_orbitals=n_orbitals,
        bonds=[tuple(bond) for bond in bonds],
        **couplings,
    )


def hubbard(shape=4, t: float = 1.0, U: float = 4.0, *, periodic: bool = False,
            mu: float | None = None) -> Model:
    """Single-band Hubbard cluster ``-t sum <ij>s (c†_is c_js + h.c.) + U sum_i n_iu n_id``.

    ``shape`` is a chain length or a ``(rows, cols)`` grid, so ``hubbard(4)`` is
    the 4-site chain and ``hubbard((2, 2))`` the 2x2 cluster of the Phase-7
    ladder.

    ``mu`` subtracts ``mu * n_i`` per site and defaults to ``U/2``, the
    particle-hole symmetric point. The default matters: written
    grand-canonically with ``mu = 0``, the cluster's *global* ground state is
    not at half filling -- for the 4-site chain at ``U = 4`` it sits in the
    two-electron sector, because shedding electrons avoids the interaction. At
    ``mu = U/2`` half filling is the global minimum, so the model means what its
    name implies and a comparison against ``sparse_ground`` is informative.
    Pass ``mu=0.0`` explicitly for the bare Hamiltonian, and see
    ``sparse.sparse_ground_in_sector`` when a specific filling is wanted.
    """
    rows, cols = _shape(shape)
    sites = rows * cols
    if sites < 2:
        raise ValueError("need at least two sites")
    mu = 0.5 * float(U) if mu is None else float(mu)
    n = 2 * sites
    bonds = _rectangle_bonds(rows, cols, periodic=periodic)
    hop = MV(n)
    for i, j in bonds:
        for spin in (SPIN_UP, SPIN_DOWN):
            hop = hop + (-float(t)) * _hop(n, spin_orbital(i, spin),
                                           spin_orbital(j, spin))
    diagonal = MV(n)
    for site in range(sites):
        up = _number(n, spin_orbital(site, SPIN_UP))
        down = _number(n, spin_orbital(site, SPIN_DOWN))
        diagonal = diagonal + float(U) * (up * down)
        if mu:
            diagonal = diagonal - float(mu) * (up + down)
    reference = _half_filled_reference(n, sites)
    return Model(
        name=f"hubbard({rows}x{cols},t={t},U={U},{'pbc' if periodic else 'obc'})",
        n=n, hamiltonian=_hermitize(n, hop, diagonal),
        reference=reference, hva_layers=(),
        metadata=_fermionic_metadata(rows, cols, bonds, 1,
                                     {"t": float(t), "U": float(U),
                                      "mu": float(mu), "periodic": bool(periodic)},
                                     reference))


def extended_hubbard(shape=4, t: float = 1.0, U: float = 4.0, V: float = 1.0, *,
                     periodic: bool = False) -> Model:
    """Hubbard plus nearest-neighbour repulsion ``V sum_<ij> (n_i - 1)(n_j - 1)``.

    The term that makes charge order compete with the Mott physics, and a
    candidate for the competing-order references of §4.2 level 4. Written in the
    particle-hole symmetric form (densities measured from half filling) so that
    it composes with the ``mu = U/2`` convention of :func:`hubbard` instead of
    pushing the global ground state out of the half-filled sector.
    """
    rows, cols = _shape(shape)
    sites = rows * cols
    base = hubbard(shape, t, U, periodic=periodic)
    n = base.n
    bonds = base.metadata["bonds"]
    diagonal = MV(n)
    one = MV.scalar(n, 1.0)
    for i, j in bonds:
        n_i = (_number(n, spin_orbital(i, SPIN_UP))
               + _number(n, spin_orbital(i, SPIN_DOWN)) - one)
        n_j = (_number(n, spin_orbital(j, SPIN_UP))
               + _number(n, spin_orbital(j, SPIN_DOWN)) - one)
        diagonal = diagonal + float(V) * (n_i * n_j)
    hamiltonian = PauliSum(n, dict(base.hamiltonian.terms)) + _hermitize(
        n, MV(n), diagonal)
    metadata = dict(base.metadata)
    metadata["V"] = float(V)
    return Model(name=f"extended_{base.name[:-1]},V={V})", n=n,
                 hamiltonian=hamiltonian, reference=base.reference,
                 hva_layers=(), metadata=metadata)


def kanamori(sites: int = 2, n_orbitals: int = 2, t: float = 1.0, U: float = 4.0,
             J: float = 0.5, *, periodic: bool = False,
             u_prime: float | None = None, mu: float | None = None) -> Model:
    """Multi-orbital Hubbard-Kanamori cluster.

    Interaction ``U sum n_mu n_md + (U' - J/2) sum_{m<m'} n_m n_m'``
    ``- 2J sum_{m<m'} S_m.S_m'``
    in the density-density plus Hund form, with the standard rotationally
    invariant relation ``U' = U - 2J`` unless ``u_prime`` overrides it. Spin-flip
    and pair-hopping terms are included, since dropping them is what turns a
    Kanamori model into a density-density caricature -- and they are exactly the
    terms that make the multi-orbital problem hard.

    Hopping is intra-orbital between neighbouring sites.

    ``mu`` defaults to the particle-hole symmetric value
    ``U/2 + (M-1)(U' - J/2)`` for ``M`` orbitals, which is the linear residue
    the rotationally invariant interaction picks up under particle-hole
    conjugation.
    Without it the global ground state sits below half filling, as it does for
    the single-band model.
    """
    if n_orbitals < 1:
        raise ValueError("need at least one orbital")
    if sites < 1:
        raise ValueError("need at least one site")
    n = 2 * sites * n_orbitals
    up_prime = float(U) - 2.0 * float(J) if u_prime is None else float(u_prime)
    interorbital_density = up_prime - 0.5 * float(J)
    chemical = (0.5 * float(U) + (n_orbitals - 1) * interorbital_density if mu is None
                else float(mu))
    bonds = _rectangle_bonds(1, sites, periodic=periodic) if sites > 1 else []

    def index(site: int, orbital: int, spin: int) -> int:
        return spin_orbital(site, spin, orbital=orbital, n_orbitals=n_orbitals)

    hop = MV(n)
    for i, j in bonds:
        for orbital in range(n_orbitals):
            for spin in (SPIN_UP, SPIN_DOWN):
                hop = hop + (-float(t)) * _hop(n, index(i, orbital, spin),
                                               index(j, orbital, spin))
    diagonal = MV(n)
    for site in range(sites):
        for m in range(n_orbitals):
            up = _number(n, index(site, m, SPIN_UP))
            down = _number(n, index(site, m, SPIN_DOWN))
            diagonal = diagonal + float(U) * (up * down)
        for m in range(n_orbitals):
            for mp in range(m + 1, n_orbitals):
                n_m = (_number(n, index(site, m, SPIN_UP))
                       + _number(n, index(site, m, SPIN_DOWN)))
                n_mp = (_number(n, index(site, mp, SPIN_UP))
                        + _number(n, index(site, mp, SPIN_DOWN)))
                diagonal = diagonal + interorbital_density * (n_m * n_mp)
                # -2 J S_m . S_m' in the same-site orbital pair, written out as
                # the Ising part (diagonal) plus spin flip (off-diagonal below)
                sz_m = 0.5 * (_number(n, index(site, m, SPIN_UP))
                              - _number(n, index(site, m, SPIN_DOWN)))
                sz_mp = 0.5 * (_number(n, index(site, mp, SPIN_UP))
                               - _number(n, index(site, mp, SPIN_DOWN)))
                diagonal = diagonal + (-2.0 * float(J)) * (sz_m * sz_mp)
        for m in range(n_orbitals):
            for spin in (SPIN_UP, SPIN_DOWN):
                diagonal = diagonal - chemical * _number(n, index(site, m, spin))

    off = MV(n)
    for site in range(sites):
        for m in range(n_orbitals):
            for mp in range(n_orbitals):
                if m == mp:
                    continue
                # spin flip: -J c†_mu c_md c†_m'd c_m'u
                off = off + (-float(J)) * (
                    cdag_op(n, index(site, m, SPIN_UP))
                    * c_op(n, index(site, m, SPIN_DOWN))
                    * cdag_op(n, index(site, mp, SPIN_DOWN))
                    * c_op(n, index(site, mp, SPIN_UP)))
                # pair hopping: J c†_mu c†_md c_m'd c_m'u
                off = off + float(J) * (
                    cdag_op(n, index(site, m, SPIN_UP))
                    * cdag_op(n, index(site, m, SPIN_DOWN))
                    * c_op(n, index(site, mp, SPIN_DOWN))
                    * c_op(n, index(site, mp, SPIN_UP)))
    # The spin-flip and pair-hopping sums already run over both orderings of
    # (m, m'), so they are self-adjoint as written; only the hopping needs its
    # adjoint added.
    hamiltonian = _hermitize(n, hop, diagonal + off)
    reference = Program(n)
    for site in range(sites):  # one electron per orbital, alternating spin
        for m in range(n_orbitals):
            spin = SPIN_UP if (site + m) % 2 == 0 else SPIN_DOWN
            reference.clifford("X", index(site, m, spin))
    metadata = _fermionic_metadata(1, sites, bonds, n_orbitals,
                                   {"t": float(t), "U": float(U), "J": float(J),
                                    "u_prime": up_prime, "mu": chemical,
                                    "periodic": bool(periodic)}, reference)
    return Model(name=f"kanamori(sites={sites},orbitals={n_orbitals},U={U},J={J})",
                 n=n, hamiltonian=hamiltonian, reference=reference,
                 hva_layers=(), metadata=metadata)


def anderson_impurity(n_bath: int = 2, U: float = 4.0, V: float = 1.0,
                      *, bath_energies: Sequence[float] | None = None,
                      impurity_energy: float | None = None) -> Model:
    """Single-impurity Anderson model with a discretized bath.

    ``U n_iu n_id + eps_i n_i + sum_k eps_k n_k + V sum_ks (c†_is c_ks + h.c.)``
    -- site 0 is the impurity, sites ``1..n_bath`` the bath levels.
    ``impurity_energy`` defaults to ``-U/2`` (the particle-hole symmetric
    point), and the bath levels default to a symmetric flat grid in
    ``[-1, 1]``. This is the substrate a DMFT-style outer loop solves
    repeatedly, which is why the repeated-solve cost of the bank matters here.
    """
    if n_bath < 1:
        raise ValueError("need at least one bath level")
    sites = n_bath + 1
    n = 2 * sites
    energies = ([-1.0 + 2.0 * k / (n_bath - 1) for k in range(n_bath)]
                if bath_energies is None and n_bath > 1
                else [0.0] * n_bath if bath_energies is None
                else [float(e) for e in bath_energies])
    if len(energies) != n_bath:
        raise ValueError("bath_energies must have one entry per bath level")
    eps_impurity = -0.5 * float(U) if impurity_energy is None else float(impurity_energy)

    hop = MV(n)
    for k in range(n_bath):
        for spin in (SPIN_UP, SPIN_DOWN):
            hop = hop + float(V) * _hop(n, spin_orbital(0, spin),
                                        spin_orbital(k + 1, spin))
    diagonal = float(U) * (_number(n, spin_orbital(0, SPIN_UP))
                           * _number(n, spin_orbital(0, SPIN_DOWN)))
    for spin in (SPIN_UP, SPIN_DOWN):
        diagonal = diagonal + eps_impurity * _number(n, spin_orbital(0, spin))
        for k, energy in enumerate(energies):
            diagonal = diagonal + energy * _number(n, spin_orbital(k + 1, spin))

    reference = Program(n)
    reference.clifford("X", spin_orbital(0, SPIN_UP))
    for k in range(n_bath):  # fill the bath levels below zero, spins paired
        if energies[k] < 0:
            reference.clifford("X", spin_orbital(k + 1, SPIN_UP))
            reference.clifford("X", spin_orbital(k + 1, SPIN_DOWN))
    metadata = _fermionic_metadata(1, sites, [(0, k + 1) for k in range(n_bath)], 1,
                                   {"U": float(U), "V": float(V),
                                    "bath_energies": energies,
                                    "impurity_energy": eps_impurity}, reference)
    metadata.update({"kind": ANDERSON_IMPURITY, "impurity_site": 0,
                     "bath_sites": list(range(1, sites))})
    return Model(name=f"anderson(bath={n_bath},U={U},V={V})", n=n,
                 hamiltonian=_hermitize(n, hop, diagonal), reference=reference,
                 hva_layers=(), metadata=metadata)


def honeycomb_links(rows: int, cols: int, *, periodic: bool = False):
    """Brick-wall honeycomb links as ``(kind, i, j)`` with kind in ``xyz``.

    Site ``2*(r*cols + c) + s`` is sublattice ``s`` of cell ``(r, c)``. The
    z-link joins the two sublattices inside a cell; x- and y-links leave the B
    site to the A site of the next cell along the two lattice directions. This
    is the standard brick-wall drawing of the honeycomb lattice -- every site
    has exactly one link of each kind in the bulk, which is what makes the
    Kitaev model's three-fold structure show up.
    """
    def site(r: int, c: int, s: int) -> int:
        return 2 * ((r % rows) * cols + (c % cols)) + s

    links = []
    for r in range(rows):
        for c in range(cols):
            links.append(("z", site(r, c, 0), site(r, c, 1)))
            if c + 1 < cols or periodic:
                links.append(("x", site(r, c, 1), site(r, c + 1, 0)))
            if r + 1 < rows or periodic:
                links.append(("y", site(r, c, 1), site(r + 1, c, 0)))
    return links


def kitaev_honeycomb(rows: int = 2, cols: int = 2, kx: float = 1.0,
                     ky: float = 1.0, kz: float = 1.0, *,
                     periodic: bool = False) -> Model:
    """Kitaev honeycomb cluster ``-Kx sum_x X_iX_j - Ky sum_y Y_iY_j - Kz sum_z Z_iZ_j``.

    A spin model: one qubit per site and no Jordan-Wigner transformation at all,
    so the Hamiltonian is written straight into Pauli words. The reference is
    ``|0...0>``; the plaquette structure and the competing-order states of §4.2
    level 4 are what the adaptive basis has to discover.
    """
    if rows < 1 or cols < 1:
        raise ValueError("need at least one unit cell")
    n = 2 * rows * cols
    couplings = {"x": float(kx), "y": float(ky), "z": float(kz)}
    letters = {"x": "X", "y": "Y", "z": "Z"}
    links = honeycomb_links(rows, cols, periodic=periodic)
    terms: dict[int, complex] = {}
    grouped: dict[str, list] = {"x": [], "y": [], "z": []}
    for kind, i, j in links:
        letter = letters[kind]
        label = ["I"] * n
        label[i] = letter
        label[j] = letter
        word = PauliSum.from_labels({"".join(label): 1.0}, n).items()[0][0]
        terms[word.code] = terms.get(word.code, 0.0) - couplings[kind]
        grouped[kind].append(word)
    reference = Program(n)
    return Model(
        name=f"kitaev({rows}x{cols},K={kx},{ky},{kz},{'pbc' if periodic else 'obc'})",
        n=n, hamiltonian=PauliSum(n, terms), reference=reference,
        hva_layers=tuple((kind, tuple(words)) for kind, words in grouped.items()
                         if words),
        metadata=model_metadata(SPIN_LATTICE, lattice="honeycomb", sites=n,
                                rows=rows, cols=cols, periodic=bool(periodic),
                                links=[(kind, i, j) for kind, i, j in links],
                                couplings=couplings))


def bipartition(metadata) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Two-colour the bond graph, so "the other sublattice" is well defined.

    Neel order is a statement about sublattices, and on a 2x2 or 2x3 cluster the
    site numbering is *not* the sublattice: sites 0 and 3 of the 2x2 grid are
    both corners of the same colour. Colouring the bonds is the only way to get
    this right for every shape, and it also reports honestly when it cannot --
    a frustrated (odd-cycle) lattice has no bipartition, and there is no Neel
    state to name.
    """
    sites = int(metadata["sites"])
    bonds = [tuple(bond) for bond in metadata.get("bonds", ())]
    neighbours: dict[int, list[int]] = {s: [] for s in range(sites)}
    for i, j in bonds:
        neighbours[i].append(j)
        neighbours[j].append(i)
    colour: dict[int, int] = {}
    for start in range(sites):
        if start in colour:
            continue
        colour[start] = 0
        stack = [start]
        while stack:
            site = stack.pop()
            for other in neighbours[site]:
                if other not in colour:
                    colour[other] = 1 - colour[site]
                    stack.append(other)
                elif colour[other] == colour[site]:
                    raise ValueError("lattice is not bipartite: no Neel sublattice")
    return (tuple(s for s in range(sites) if colour[s] == 0),
            tuple(s for s in range(sites) if colour[s] == 1))


def competing_orders(model) -> dict[str, tuple[int, ...]]:
    """Occupation patterns of the orders a Hubbard cluster chooses between.

    The §4.2 level-4 "stabilizer configurations for competing orders", as
    determinants: antiferromagnetic (one electron per site, spins alternating by
    sublattice), its spin-flipped partner, charge density wave (doublons on one
    sublattice, holes on the other), and the striped state the site numbering
    happens to produce. Each is returned as the tuple of filled spin orbitals,
    so ``subspace.configuration_generator`` can turn it into the single
    ``X``-string that carries the reference onto it.

    Every pattern is emitted at the reference's own ``(N, S_z)`` -- a
    configuration in a different sector is not a competing order for this
    problem, it is a different problem, and the subspace would be carrying a
    direction its Hamiltonian block cannot connect to. Patterns that cannot be
    built at that filling are omitted rather than returned at the wrong sector.
    """
    metadata = model.metadata
    if metadata.get("kind") != FERMIONIC_LATTICE:
        raise ValueError("competing orders are defined for fermionic lattices")
    if int(metadata.get("n_orbitals", 1)) != 1:
        raise ValueError("competing orders are implemented for single-orbital models")
    sites = int(metadata["sites"])
    electrons, sz = int(metadata["n_electrons"]), float(metadata["sz"])
    try:
        even, odd = bipartition(metadata)
    except ValueError:
        even, odd = tuple(range(0, sites, 2)), tuple(range(1, sites, 2))

    def pattern(up_sites, down_sites, doubled=()):
        filled = ([spin_orbital(s, SPIN_UP) for s in up_sites]
                  + [spin_orbital(s, SPIN_DOWN) for s in down_sites]
                  + [o for s in doubled
                     for o in (spin_orbital(s, SPIN_UP), spin_orbital(s, SPIN_DOWN))])
        return tuple(sorted(filled))

    candidates = {
        "afm": pattern(even, odd),
        "afm_flipped": pattern(odd, even),
        "cdw": pattern((), (), doubled=even),
        "cdw_odd": pattern((), (), doubled=odd),
        "stripe": pattern(range(0, sites, 2), range(1, sites, 2)),
    }
    out = {}
    for name, filled in candidates.items():
        n_electrons = len(filled)
        spin = 0.5 * sum(1 if orbital % 2 == 0 else -1 for orbital in filled)
        if n_electrons == electrons and abs(spin - sz) < 1e-12:
            out[name] = filled
    return out

r"""The metadata contract every :class:`~clifford_qc.models.spin.Model` must satisfy.

``Model.metadata`` is where a model states the conventions under which its
Pauli sum means anything: which spin ordering the Jordan-Wigner indices use,
which ``(N, S_z)`` sector its reference determinant occupies, how many spatial
orbitals it was built from.  Every consumer that transforms a model has to know
those things, and until this module existed none of them were required.

That was not a hypothetical gap.  Three symptoms, all of them present in the
tree before this contract:

1. **A silent default on a value whose absence changes the answer.**  The
   literal ``metadata.get("spin_convention", "interleaved")`` appeared 8 times
   across 4 benchmark scripts -- every read of that key in the repository
   carried a default.  ``spin_ordering="blocked"`` is meanwhile a first-class
   option in ``fermion_mapping``, ``backends.sector_statevector``,
   ``subspace.adaptive`` and ``subspace.restriction``, and
   ``_spin_parity_rows`` computes a *different* spin-up parity row for it.  So
   a two-qubit reduction run under the wrong ordering is not approximately
   wrong, it is a different tapering: on the 2-site Hubbard dimer,
   ``parity+2q`` returns ``-4.000000000000`` instead of ``-4.828427124746``
   with no exception raised and a plausible-looking ``W = 1``.  A default is
   the wrong shape for a value like that.

2. **A consumer carrying the model's identity in a parallel variable.**
   ``benchmarks/run_acase_ladder.py`` builds ``(model, kind)`` tuples and
   threads ``kind`` through a dozen call sites, because ``spin.tfim`` did not
   state its own ``kind``.  For ``hubbard`` and ``kitaev_honeycomb``, which do,
   the parallel value duplicated metadata that already existed and could
   silently disagree with it.

3. **A whole builder family unusable with a sibling module.**
   ``models.observables`` refuses any model without ``kind``, so every
   ``spin.tfim``/``xxz``/``random_ising`` model raised there on arrival.

The remedy is the pattern ``models.effective`` already uses at the
effective-Hamiltonian boundary: a versioned schema string and a validator that
rejects what does not conform.  Here it runs in ``Model.__post_init__``, so a
model that cannot state its own conventions cannot be constructed at all.

**What is deliberately *not* required.**  ``n`` and the Pauli-word count are
already carried by the ``Model`` itself, and duplicating them into the metadata
dict would create two sources of truth that can drift apart.  So this module
requires neither, and instead *checks* ``spin_orbitals`` and ``pauli_words``
against the model when they are present.  The contract asks a builder to state
what cannot be derived, and to keep what can be derived consistent.

The key is ``metadata_schema`` rather than ``schema`` because
``models.effective`` already uses ``schema`` in model metadata for a different
thing -- the schema of the *input payload* it was downfolded from.  Overloading
one key with two meanings is how a contract stops being checkable.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

MODEL_METADATA_SCHEMA = "clifford_qc.model_metadata.v1"

# Kind vocabulary. ``models.observables`` already exported FERMIONIC and SPIN;
# these are the same strings, collected so the set is enumerable rather than
# discovered by grep.
FERMIONIC_LATTICE = "fermionic_lattice"
FERMIONIC_ORBITAL_BASIS = "fermionic_orbital_basis"
ANDERSON_IMPURITY = "anderson_impurity"
MOLECULAR = "molecular"
SPIN_LATTICE = "spin_lattice"

#: Kinds whose Pauli sum is a fermionic operator under a named encoding, and
#: which therefore have to declare an ordering and a sector.
FERMIONIC_KINDS = frozenset({
    FERMIONIC_LATTICE, FERMIONIC_ORBITAL_BASIS, ANDERSON_IMPURITY, MOLECULAR,
})

#: Kinds whose Pauli sum is a spin Hamiltonian: no fermionic ordering applies,
#: and particle number is not a symmetry to declare.
SPIN_KINDS = frozenset({SPIN_LATTICE})

MODEL_KINDS = FERMIONIC_KINDS | SPIN_KINDS

SPIN_ORDERINGS = ("interleaved", "blocked")

_UNIVERSAL_KEYS = ("metadata_schema", "kind")
_FERMIONIC_KEYS = ("spin_convention", "spin_orbitals", "n_spatial_orbitals",
                   "n_electrons", "sz")


def model_metadata(kind: str, *, spin_orbitals: int | None = None,
                   n_spatial_orbitals: int | None = None,
                   n_electrons: int | None = None, sz: float | None = None,
                   spin_convention: str = "interleaved",
                   **extra: Any) -> dict:
    """Assemble a contract-satisfying metadata dict for a model builder.

    Builders call this instead of writing the schema string and the sector keys
    out by hand, so a new builder cannot forget half of them.  ``extra`` carries
    the model-specific parameters (couplings, source paths, basis names), which
    the contract does not constrain.

    For a spin kind, the fermionic arguments must all be omitted -- a spin
    Hamiltonian has no spin-orbital ordering, and accepting one would let a
    caller declare a convention that nothing honours.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"kind must be one of {sorted(MODEL_KINDS)}, got {kind!r}")
    metadata: dict[str, Any] = {
        "metadata_schema": MODEL_METADATA_SCHEMA,
        "kind": kind,
    }
    fermionic_arguments = {
        "spin_orbitals": spin_orbitals,
        "n_spatial_orbitals": n_spatial_orbitals,
        "n_electrons": n_electrons,
        "sz": sz,
    }
    if kind in SPIN_KINDS:
        named = sorted(k for k, v in fermionic_arguments.items() if v is not None)
        if named:
            raise ValueError(
                f"spin model {kind!r} cannot declare fermionic metadata {named}")
    else:
        missing = sorted(k for k, v in fermionic_arguments.items() if v is None)
        if missing:
            raise ValueError(f"fermionic kind {kind!r} needs {missing}")
        metadata.update({
            "spin_convention": spin_convention,
            "spin_orbitals": int(spin_orbitals),
            "n_spatial_orbitals": int(n_spatial_orbitals),
            "n_electrons": int(n_electrons),
            "sz": float(sz),
        })
    metadata.update(extra)
    return metadata


def _require_int(metadata: Mapping[str, Any], key: str, *, model: str) -> int:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"model {model!r}: metadata[{key!r}] must be an int, "
                        f"got {type(value).__name__}")
    return value


def validate_model_metadata(metadata: Mapping[str, Any], *, n_qubits: int,
                            model_name: str = "<model>",
                            pauli_words: int | None = None) -> None:
    """Check a metadata mapping against :data:`MODEL_METADATA_SCHEMA`.

    Raises ``TypeError`` for a wrong type and ``ValueError`` for a missing key,
    an unknown vocabulary value, or an internal inconsistency.  Called from
    ``Model.__post_init__``; also useful directly on a payload about to become a
    model.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError(f"model {model_name!r}: metadata must be a mapping")
    missing = [key for key in _UNIVERSAL_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            f"model {model_name!r}: metadata is missing {missing}. Every model "
            f"must declare {list(_UNIVERSAL_KEYS)}; build the dict with "
            "models.metadata.model_metadata().")
    schema = metadata["metadata_schema"]
    if schema != MODEL_METADATA_SCHEMA:
        raise ValueError(f"model {model_name!r}: metadata_schema must be "
                         f"{MODEL_METADATA_SCHEMA!r}, got {schema!r}")
    kind = metadata["kind"]
    if kind not in MODEL_KINDS:
        raise ValueError(f"model {model_name!r}: kind must be one of "
                         f"{sorted(MODEL_KINDS)}, got {kind!r}")

    # Derivable quantities are not required, but a stated one that disagrees
    # with the model is worse than an absent one -- it reads as authoritative.
    if "pauli_words" in metadata and pauli_words is not None:
        declared = _require_int(metadata, "pauli_words", model=model_name)
        if declared != pauli_words:
            raise ValueError(f"model {model_name!r}: metadata declares "
                             f"pauli_words={declared} but the Hamiltonian has "
                             f"{pauli_words}")

    if kind in SPIN_KINDS:
        stated = sorted(key for key in _FERMIONIC_KEYS if key in metadata)
        if stated:
            raise ValueError(f"model {model_name!r}: spin kind {kind!r} must not "
                             f"declare fermionic metadata {stated}")
        return

    missing = [key for key in _FERMIONIC_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            f"model {model_name!r}: fermionic kind {kind!r} is missing "
            f"{missing}. These are the conventions a consumer cannot infer: "
            "the Jordan-Wigner ordering and the sector the reference occupies.")

    ordering = metadata["spin_convention"]
    if ordering not in SPIN_ORDERINGS:
        raise ValueError(f"model {model_name!r}: spin_convention must be one of "
                         f"{list(SPIN_ORDERINGS)}, got {ordering!r}")

    spin_orbitals = _require_int(metadata, "spin_orbitals", model=model_name)
    if spin_orbitals != n_qubits:
        raise ValueError(f"model {model_name!r}: spin_orbitals={spin_orbitals} "
                         f"but the model has n={n_qubits} qubits")
    spatial = _require_int(metadata, "n_spatial_orbitals", model=model_name)
    if 2 * spatial != spin_orbitals:
        raise ValueError(f"model {model_name!r}: n_spatial_orbitals={spatial} is "
                         f"inconsistent with spin_orbitals={spin_orbitals}")

    electrons = _require_int(metadata, "n_electrons", model=model_name)
    if not 0 <= electrons <= spin_orbitals:
        raise ValueError(f"model {model_name!r}: n_electrons={electrons} is "
                         f"outside [0, {spin_orbitals}]")
    sz = metadata["sz"]
    if isinstance(sz, bool) or not isinstance(sz, (int, float)):
        raise TypeError(f"model {model_name!r}: metadata['sz'] must be a number")
    sz = float(sz)
    if not math.isfinite(sz):
        raise ValueError(f"model {model_name!r}: sz must be finite")
    if abs(2.0 * sz) > electrons:
        raise ValueError(f"model {model_name!r}: |2*sz|={abs(2.0*sz)} exceeds "
                         f"n_electrons={electrons}")
    # 2*S_z and N have the same parity for any determinant: flipping one spin
    # moves 2*S_z by 2. A mismatch means the sector is empty, which is the
    # error that previously reached a backend as an unbuildable subspace.
    if abs((2.0 * sz) % 2.0 - electrons % 2) > 1e-9:
        raise ValueError(f"model {model_name!r}: sz={sz} and n_electrons="
                         f"{electrons} have mismatched parity, so the sector is "
                         "empty")


def spin_convention(model) -> str:
    """The model's Jordan-Wigner spin ordering, or a raise.

    Use this in place of ``model.metadata.get("spin_convention", "interleaved")``.
    Reading this key wrong silently changes the Hamiltonian a reduction acts on,
    so there is no defensible default to fall back to; a spin model has no
    ordering at all and asking for one is a bug in the caller.
    """
    kind = model.metadata.get("kind")
    if kind in SPIN_KINDS:
        raise ValueError(f"model {model.name!r} is a spin model ({kind!r}); it "
                         "has no fermionic spin ordering")
    try:
        return model.metadata["spin_convention"]
    except KeyError:
        raise ValueError(
            f"model {model.name!r} does not declare spin_convention") from None


__all__ = [
    "ANDERSON_IMPURITY",
    "FERMIONIC_KINDS",
    "FERMIONIC_LATTICE",
    "FERMIONIC_ORBITAL_BASIS",
    "MODEL_KINDS",
    "MODEL_METADATA_SCHEMA",
    "MOLECULAR",
    "SPIN_KINDS",
    "SPIN_LATTICE",
    "SPIN_ORDERINGS",
    "model_metadata",
    "spin_convention",
    "validate_model_metadata",
]

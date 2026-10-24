"""Immutable fixed-(N,Sz) references and costed matrix actions.

Occupation masks use q0 as the leftmost bit, Pauli codes its lowest lane.
The supported convention is interleaved Jordan-Wigner. No density or H² is built.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..reference_states import ReferenceMetadata, operator_identity, snapshot_operator
from .sector_statevector import SectorStatevectorBackend


def immutable_array(value, dtype):
    value = np.asarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class SectorSpec:
    n: int
    n_electrons: int
    sz: float
    spin_ordering: str = "interleaved"

    def __post_init__(self):
        for name in ("n", "n_electrons"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 < self.n < 63 or self.n % 2 or not 0 <= self.n_electrons <= self.n:
            raise ValueError("require even 0 < n < 63 and 0 <= n_electrons <= n")
        if (isinstance(self.sz, bool) or not isinstance(self.sz, (float, int))
                or not math.isfinite(self.sz) or abs(2*self.sz-round(2*self.sz)) > 1e-12):
            raise ValueError("sz must be a finite integer or half-integer")
        if self.spin_ordering != "interleaved":
            raise ValueError("sector-action workflow requires interleaved spin ordering")
        if (self.n_electrons + round(2*self.sz)) % 2:
            raise ValueError("particle number and Sz have inconsistent parity")
        up = (self.n_electrons + round(2*self.sz)) // 2
        down = self.n_electrons-up
        if not 0 <= up <= self.n//2 or not 0 <= down <= self.n//2:
            raise ValueError("the requested sector is empty")

    @property
    def dimension(self):
        up = (self.n_electrons+round(2*self.sz))//2
        return math.comb(self.n//2, up)*math.comb(self.n//2, self.n_electrons-up)

    def spin_dimension(self, spin):
        up, down = round(self.n_electrons/2+spin), round(self.n_electrons/2-spin)
        def choose(k): return math.comb(self.n//2,k) if 0 <= k <= self.n//2 else 0
        return choose(up)*choose(down)-choose(up+1)*choose(down-1)

    @property
    def identity(self):
        text = f"sector.v1:{self.n}:{self.n_electrons}:{float(self.sz).hex()}:{self.spin_ordering}:q0-leftmost"
        return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True, init=False)
class SectorStateReference:
    spec: SectorSpec
    metadata: ReferenceMetadata
    _amplitudes: np.ndarray
    _backend: SectorStatevectorBackend

    def __init__(self, spec, amplitudes, *, max_owned_bytes=256*2**20):
        from ..subspace.bounded_actions import ActionMemoryLimit, positive_integer
        if not isinstance(spec, SectorSpec):
            raise TypeError("a SectorSpec is required")
        positive_integer(max_owned_bytes,"max_owned_bytes")
        if 64*spec.dimension > max_owned_bytes:
            raise ActionMemoryLimit("sector reference exceeds owned-array admission before basis construction")
        values = np.asarray(amplitudes,complex)
        if (values.shape != (spec.dimension,) or not np.isfinite(values).all()
                or abs(np.vdot(values,values)-1) > 1e-10):
            raise ValueError("finite normalized amplitudes matching the declared sector are required")
        backend = SectorStatevectorBackend(spec.n,spec.n_electrons,spec.sz,spin_ordering=spec.spin_ordering)
        backend.basis = immutable_array(backend.basis,np.int64)
        values = immutable_array(values,complex)
        identity = hashlib.sha256(spec.identity.encode()+values.tobytes()).hexdigest()
        object.__setattr__(self,"spec",spec)
        object.__setattr__(self,"_backend",backend)
        object.__setattr__(self,"_amplitudes",values)
        object.__setattr__(self,"metadata",ReferenceMetadata(
            spec.n,identity,frozenset({"sector_actions","amplitudes"}),spin_ordering=spec.spin_ordering))

    @property
    def n(self): return self.spec.n

    @property
    def sector(self): return self.spec.n_electrons,self.spec.sz

    @property
    def amplitudes(self): return self._amplitudes

    @property
    def occupation_words(self): return self._backend.basis

    def compile_operator(self, operator, *, precompute=False, validate_sector=True):
        return copy(self._backend).operator(operator,precompute=precompute,validate_sector=validate_sector)

    def apply_operator(self, operator):
        return self.compile_operator(operator,validate_sector=False).matvec(self.amplitudes)

    def storage(self):
        return {"amplitude_payload_bytes":self.amplitudes.nbytes,
                "basis_payload_bytes":self.occupation_words.nbytes,"density_coefficients":0,
                "scope":"owned sector arrays; excludes Python and operator compilation"}


class SectorActionProvider:
    """Exact block actions, immutable inputs and a finite-polynomial spin projection.

    H/S² columns include start projection, recovery and diagnostics. No dense spin
    frame is built. Compilation is separately admitted and priced by the workflow.
    """
    def __init__(self, reference, hamiltonian, *, target_spin=0.,
                 max_owned_bytes=256*2**20, precompute=False):
        from ..subspace.bounded_actions import ActionMemoryLimit, positive_integer
        from ..subspace.contracts import as_multivector
        from ..subspace.spin_packets import spin_squared_operator
        if not isinstance(reference,SectorStateReference):
            raise TypeError("a SectorStateReference is required")
        positive_integer(max_owned_bytes,"max_owned_bytes")
        spec = reference.spec
        max_spin = min(spec.n_electrons,spec.n-spec.n_electrons)/2
        if (isinstance(target_spin,bool) or not isinstance(target_spin,(float,int))
                or not np.isfinite(target_spin) or not abs(spec.sz) <= target_spin <= max_spin
                or abs(target_spin-abs(spec.sz)-round(target_spin-abs(spec.sz))) > 1e-12):
            raise ValueError("target_spin must be an allowed total spin in the declared sector")
        h = as_multivector(hamiltonian)
        if h.n != spec.n or not h.is_hermitian():
            raise ValueError("Hamiltonian must be Hermitian on the reference register")
        retained = 24*spec.dimension+24*h.nnz()+256*spec.n**2
        groups_bound = min(h.nnz(),1 << spec.n)
        if precompute:
            retained += 32*spec.dimension*(groups_bound+1+spec.n**2)
        if retained+256*spec.dimension > max_owned_bytes:
            raise ActionMemoryLimit("sector provider exceeds owned-array admission before compilation")
        self._reference = reference
        self._hamiltonian_input = snapshot_operator(h)
        self._target_spin = float(target_spin)
        spin = spin_squared_operator(spec.n).to_mv()
        commutator = self.hamiltonian*spin-spin*self.hamiltonian
        scale = max(1.,sum(abs(x) for x in h.terms.values())*sum(abs(x) for x in spin.terms.values()))
        if max((abs(x) for x in commutator.terms.values()),default=0) > 1e-10*scale:
            raise ValueError("Hamiltonian must conserve the requested total-spin sector")
        self._hamiltonian = reference.compile_operator(self.hamiltonian,precompute=precompute)
        self._spin = reference.compile_operator(spin,precompute=precompute)
        self._hamiltonian_identity = operator_identity(self.hamiltonian)
        self._reference_identity = reference.metadata.identity
        self._sector_identity = hashlib.sha256((spec.identity+f":S={float(target_spin).hex()}").encode()).hexdigest()
        self.h_action_columns = self.spin_action_columns = self.spin_projections = 0
        self.diagonal_word_scans = 0
        self._precompute = bool(precompute)
        self._input_payload_bytes = (24*spec.dimension+24*(h.nnz()+spin.nnz())
                                    +self._hamiltonian.memory_estimate()["compiled_bytes"]
                                    +self._spin.memory_estimate()["compiled_bytes"])

    @property
    def precompute(self): return self._precompute

    @property
    def input_payload_bytes(self): return self._input_payload_bytes

    @property
    def reference(self): return self._reference

    @property
    def hamiltonian(self): return self._hamiltonian_input

    @property
    def target_spin(self): return self._target_spin

    @property
    def hamiltonian_identity(self): return self._hamiltonian_identity

    @property
    def reference_identity(self): return self._reference_identity

    @property
    def sector_identity(self): return self._sector_identity

    @property
    def dimension(self): return self.reference.spec.dimension

    @property
    def target_dimension(self): return self.reference.spec.spin_dimension(self.target_spin)

    def apply_hamiltonian(self, vectors):
        return self._apply(vectors,self._hamiltonian,"h_action_columns")

    def apply_spin_squared(self, vectors):
        return self._apply(vectors,self._spin,"spin_action_columns")

    def _apply(self, vectors, operator, counter):
        vectors = np.asarray(vectors,complex)
        if vectors.ndim not in (1,2) or vectors.shape[0] != self.dimension or not np.isfinite(vectors).all():
            raise ValueError("finite matching vector or block required")
        out = np.empty_like(vectors)
        if vectors.ndim == 1:
            setattr(self,counter,getattr(self,counter)+1)
            out[:] = operator.matvec(vectors)
        else:
            for j in range(vectors.shape[1]):
                setattr(self,counter,getattr(self,counter)+1)
                out[:,j] = operator.matvec(vectors[:,j])
        return out

    def project_spin(self, vector, *, check=None):
        value = np.array(vector,complex,copy=True)
        if value.shape != (self.dimension,) or not np.isfinite(value).all():
            raise ValueError("finite matching spin-projection vector required")
        self.spin_projections += 1
        wanted = self.target_spin*(self.target_spin+1)
        largest = min(self.reference.spec.n_electrons,
                      self.reference.spec.n-self.reference.spec.n_electrons)/2
        for spin in np.arange(abs(self.reference.spec.sz),largest+.5,1.):
            if abs(spin-self.target_spin) > 1e-12:
                other = spin*(spin+1)
                if check is not None: check()
                acted = self.apply_spin_squared(value)
                if check is not None: check()
                value = (acted-other*value)/(wanted-other)
        return value

    def diagonal(self):
        from ..pauli_action import parity, word_masks
        values = np.zeros(self.dimension,float)
        for code,coefficient in self.hamiltonian.terms.items():
            self.diagonal_word_scans += 1
            x,z,_ = word_masks(self.reference.n,code)
            if not x:
                values += coefficient.real*(1.-2.*parity(self.reference.occupation_words,z))
        return values

    def resources(self):
        spec = self.reference.spec
        return {"provider":"sector_actions","reference_identity":self.reference_identity,
                "sector":{"n_qubits":spec.n,"n_electrons":spec.n_electrons,"spin_z":spec.sz,
                          "spin_ordering":spec.spin_ordering,"bit_order":"q0-leftmost"},
                "hamiltonian_identity":self.hamiltonian_identity,"sector_identity":self.sector_identity,
                "dimension":self.dimension,"target_spin":self.target_spin,"target_dimension":self.target_dimension,
                "diagonal_word_scans":self.diagonal_word_scans,"h_action_columns":self.h_action_columns,
                "spin_action_columns":self.spin_action_columns,"spin_projections":self.spin_projections,
                "precompute":self.precompute,"input_payload_bytes":self.input_payload_bytes,
                "input_scope":"reference arrays, 24-byte Pauli coefficient model and retained gather passes; excludes Python/compilation transients"}

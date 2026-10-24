"""NumPy/GF(2) commuting readouts, with signed native Clifford maps.

Qubit zero occupies the lowest Pauli-code lane and leftmost occupation bit.
The graph is a finite simple CZ interaction graph. A commuting CZ stage is
scheduled into disjoint matchings; it is not a single physical depth layer.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..ir import PauliWord, Program
from ..pauli_kernel import pauli_lane_mask, validate_word_code, word_mul
from ..reference_states import immutable_array
from .block_commuting import block_ranges
from .block_synthesis import gate_stats, independent_codes, local_code
from .compiled import CompiledSetting
from .cost import SettingResources


# Conjugation of the 16 two-qubit Paulis by CZ, derived using the native
# packed product. Independent MV/unitary tests validate phases separately.
def _cz_table():
    images = ((1 | (3 << 2), 3), (3 | (1 << 2), 3 << 2))
    out = []
    for code in range(16):
        phase, image = (1j ** sum(((code >> (2*q)) & 3) == 2 for q in range(2))), 0
        for q in range(2):
            letter = (code >> (2*q)) & 3
            for present, generator in ((letter in (1,2), images[q][0]),
                                       (letter in (2,3), images[q][1])):
                if present:
                    factor, image = word_mul(2,image,generator)
                    phase *= factor
        if phase not in (1,-1):
            raise AssertionError('CZ sent a Hermitian Pauli to a non-real phase')
        out.append((int(phase.real),image))
    return tuple(out)


_CZ = _cz_table()
_LOCAL = {'H': ((1,0),(1,3),(-1,2),(1,1)),
          'S': ((1,0),(1,2),(-1,1),(1,3)),
          'SDG': ((1,0),(-1,2),(1,1),(1,3))}
_CZ_ARRAYS = (immutable_array([v[0] for v in _CZ],np.int8),
              immutable_array([v[1] for v in _CZ],np.uint64))
_LOCAL_ARRAYS = MappingProxyType({name:(immutable_array([v[0] for v in table],np.int8),
                                       immutable_array([v[1] for v in table],np.uint64))
                                 for name,table in _LOCAL.items()})


@dataclass(frozen=True)
class GraphForm:
    pivots: tuple[int,...]
    pre_hadamards: tuple[int,...]
    active_x: tuple[int,...]
    phase_qubits: tuple[int,...]
    edges: tuple[tuple[int,int],...]
    matching_layers: tuple[tuple[tuple[int,int],...],...]
    kind: str = "symplectic_R"


def matching_layers(size, edges):
    """Restrict a round-robin factorization of K_even to the requested graph.

    At most k-1 layers for even k; at most k for odd k (one idle dummy vertex).
    This complete-graph construction, rather than Vizing alone, gives the
    dyadic bound. It is a schedulable upper bound, not minimum edge coloring.
    """
    edges = {tuple(sorted(edge)) for edge in edges}
    if any(a == b or not 0 <= a < b < size for a,b in edges):
        raise ValueError('edges must be distinct-vertex pairs within the block')
    order = list(range(size + size%2))
    layers = []
    for _ in range(max(0,len(order)-1)):
        layer = tuple(sorted(tuple(sorted((a,b))) for a,b in zip(order[:len(order)//2], reversed(order[len(order)//2:]))
                             if tuple(sorted((a,b))) in edges))
        if layer:
            layers.append(layer)
        order = [order[0],order[-1],*order[1:-1]]
    return tuple(layers)


def _cancel_local(gates, n):
    # Commute only across disjoint qubits. Never pass a local operation through
    # an incident CZ. This removes, e.g., the two Hs for an initially Z-only row.
    pending = [[] for _ in range(n)]
    out = list(gates)
    for index,(name,qubits) in enumerate(gates):
        if name == 'CZ':
            for q in qubits: pending[q].clear()
        else:
            q = qubits[0]
            previous = out[pending[q][-1]][0] if pending[q] else None
            if previous is not None and ((name == previous == 'H')
                               or (name,previous) in (('S','SDG'),('SDG','S'))):
                out[pending[q].pop()] = None
                out[index] = None
            else:
                pending[q].append(index)
    # Keep surviving gates in their original stage order. Delaying them to
    # the next incident CZ can split a disjoint matching into priced layers.
    return tuple(gate for gate in out if gate is not None)


@dataclass(frozen=True)
class NativeClifford:
    """Immutable gate list with exact signed P -> U P U†, without Stim."""
    n: int
    gates: tuple
    graph: GraphForm | None = None

    def __post_init__(self):
        if isinstance(self.n,bool) or not isinstance(self.n,int) or self.n < 1:
            raise ValueError('a positive register size is required')
        gates = tuple((name,tuple(qubits)) for name,qubits in self.gates)
        for name,qubits in gates:
            if name not in (*_LOCAL,'CZ') or len(qubits) != (2 if name == 'CZ' else 1):
                raise ValueError('native Clifford supports H, S, SDG and CZ')
            if len(set(qubits)) != len(qubits) or any(isinstance(q,bool) or not isinstance(q,int) or not 0 <= q < self.n for q in qubits):
                raise ValueError('invalid Clifford qubits')
        object.__setattr__(self,'gates',gates)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps([self.n,self.gates],separators=(',',':')).encode()).hexdigest()

    def conjugate(self, word):
        if not isinstance(word,PauliWord) or word.n != self.n:
            raise ValueError('word and native Clifford disagree on the register')
        sign,code = 1,word.code
        for name,qubits in self.gates:
            if name == 'CZ':
                a,b = qubits
                local = ((code >> (2*a)) & 3) | (((code >> (2*b)) & 3) << 2)
                factor,image = _CZ[local]
                code = (code & ~((3 << (2*a)) | (3 << (2*b)))) | ((image & 3) << (2*a)) | ((image >> 2) << (2*b))
            else:
                q = qubits[0]
                factor,image = _LOCAL[name][(code >> (2*q)) & 3]
                code = (code & ~(3 << (2*q))) | (image << (2*q))
            sign *= factor
        return complex(sign),PauliWord(self.n,code)

    def transform_codes(self, codes):
        """Vectorized signed conjugation for local blocks of at most 32 qubits."""
        if self.n > 32:
            raise ValueError('vectorized packed conjugation is limited to 32-qubit blocks')
        raw = list(codes)
        for code in raw:
            if isinstance(code,bool) or not isinstance(code,Integral):
                raise ValueError('integer Pauli codes required')
            validate_word_code(self.n,int(code))
        out = np.asarray(raw,dtype=np.uint64)
        return self._transform_codes_unchecked(out)

    def _transform_codes_unchecked(self, codes):
        """Trusted uint64 block codes, validated once by the compiler boundary.

        Does not mutate the cached input array. Public callers use transform_codes.
        """
        out = codes
        signs = np.ones(len(out),dtype=np.int8)
        for name,qubits in self.gates:
            if name == 'CZ':
                a,b = qubits
                local = ((out >> np.uint64(2*a)) & np.uint64(3)) | (((out >> np.uint64(2*b)) & np.uint64(3)) << np.uint64(2))
                phase = _CZ_ARRAYS[0][local]
                image = _CZ_ARRAYS[1][local]
                mask = np.uint64((3 << (2*a)) | (3 << (2*b)))
                out = (out & ~mask) | ((image & np.uint64(3)) << np.uint64(2*a)) | ((image >> np.uint64(2)) << np.uint64(2*b))
            else:
                q = qubits[0]
                letter = (out >> np.uint64(2*q)) & np.uint64(3)
                phase = _LOCAL_ARRAYS[name][0][letter]
                image = _LOCAL_ARRAYS[name][1][letter]
                out = (out & ~np.uint64(3 << (2*q))) | (image << np.uint64(2*q))
            signs *= phase
        return signs,out

    def program(self):
        program = Program(self.n)
        for name,qubits in self.gates: program.clifford(name,*qubits)
        return program

    def inverse(self):
        return NativeClifford(self.n,tuple(({'S':'SDG','SDG':'S'}.get(name,name),qubits)
                                         for name,qubits in reversed(self.gates)))


def _rref_x(x,z):
    pivots = []
    for q in range(x.shape[1]):
        live = np.flatnonzero(x[len(pivots):,q])
        if not live.size: continue
        row = len(pivots); other = row+int(live[0])
        x[[row,other]],z[[row,other]] = x[[other,row]],z[[other,row]]
        for i in range(x.shape[0]):
            if i != row and x[i,q]:
                x[i] ^= x[row]; z[i] ^= z[row]
        pivots.append(q)
    return tuple(pivots)


def graph_diagonalizer(codes, size):
    """An isotropic subgroup -> H/S/CZ/H normal form over GF(2).

    Z-only spectator columns are left in the readout parity. Clearing them by
    CZ is unnecessary when they receive no final H; this avoids entangling a
    Jordan-Wigner Z tail merely to remove it from a decoded observable.
    """
    if isinstance(size,bool) or not isinstance(size,int) or not 1 <= size <= 32:
        raise ValueError('graph blocks require 1 <= size <= 32')
    codes = list(codes)
    for code in codes:
        if isinstance(code,bool) or not isinstance(code,Integral): raise ValueError('integer codes required')
        validate_word_code(size,int(code))
    codes = [int(code) for code in codes]
    basis = independent_codes(codes,size)
    axes = [set((c >> (2*q)) & 3 for c in codes)-{0} for q in range(size)]
    if all(len(axis) <= 1 for axis in axes):
        # A graph normal form is unnecessary for qubit-wise commuting words:
        # e.g. a single YY could otherwise acquire a gratuitous CZ.
        gates = []
        for q,axis in enumerate(axes):
            if axis == {2}: gates.append(('S',(q,)))
            if axis in ({1},{2}): gates.append(('H',(q,)))
        support = np.asarray([[(c >> (2*q)) & 3 != 0 for q in range(size)] for c in basis],np.uint8).reshape(-1,size)
        pivots = _rref_x(support,np.zeros_like(support))
        active = tuple(q for q,axis in enumerate(axes) if axis in ({1},{2}))
        form = GraphForm(pivots,(),active,tuple(q for q,axis in enumerate(axes) if axis == {2}),(),(),"qwc")
        return NativeClifford(size,tuple(gates),form)
    if len(basis) > size: raise ValueError('local subgroup is not isotropic')
    x = np.asarray([[(c >> (2*q)) & 3 in (1,2) for q in range(size)] for c in basis],np.uint8).reshape(-1,size)
    z = np.asarray([[(c >> (2*q)) & 3 in (2,3) for q in range(size)] for c in basis],np.uint8).reshape(-1,size)
    if np.any((x @ z.T ^ z @ x.T) & 1):
        raise ValueError('local Pauli subgroup must commute')
    pre = []
    while True:
        pivots = _rref_x(x,z)
        if len(pivots) == len(basis): break
        choices = [q for q in range(size) if q not in pivots and z[len(pivots):,q].any()]
        if not choices: raise AssertionError('isotropic X-rank completion failed')
        q = choices[0]; x[:,q],z[:,q] = z[:,q].copy(),x[:,q].copy(); pre.append(q)
    rest = tuple(q for q in range(size) if q not in pivots)
    a,b,c = x[:,rest],z[:,pivots],z[:,rest]
    symmetric = b ^ ((a @ c.T) & 1)
    if not np.array_equal(symmetric,symmetric.T): raise AssertionError('isotropic normal form lost symmetry')
    r = np.zeros((size,size),np.uint8)
    r[np.ix_(pivots,pivots)] = symmetric
    r[np.ix_(pivots,rest)] = c; r[np.ix_(rest,pivots)] = c.T
    active = tuple(q for q in range(size) if x[:,q].any())
    phases = tuple(q for q in active if r[q,q])
    edges = tuple((u,v) for u in active for v in active if u < v and r[u,v])
    layers = matching_layers(size,edges)
    gates = ([('H',(q,)) for q in pre] + [('S',(q,)) for q in phases]
             + [('CZ',edge) for layer in layers for edge in layer] + [('H',(q,)) for q in active])
    form = GraphForm(pivots,tuple(pre),active,phases,edges,layers)
    result = NativeClifford(size,_cancel_local(gates,size),form)
    if codes:
        _,images = result.transform_codes(codes)
        mask = np.uint64(pauli_lane_mask(size))
        if np.any((images & mask) ^ ((images >> np.uint64(1)) & mask)):
            raise AssertionError('a member did not become Z-only')
    return result


def native_stats(clifford):
    """Exact gate counts, 2q-only critical depth, and one shared-clock schedule."""
    return gate_stats(clifford.gates)


@dataclass(frozen=True)
class GraphSynthesis:
    block_size: int
    settings: tuple
    gate_counts: object
    two_qubit_per_setting: tuple[int,...]
    two_qubit_depth_per_setting: tuple[int,...]
    compatibility: np.ndarray
    assignment: tuple[int,...]
    z_only_restrictions_checked: int
    coverage: tuple[float,...]
    method: str = 'graph_state'

    @property
    def n_settings(self): return len(self.settings)

    @property
    def logical_two_qubit_per_sweep(self): return sum(self.two_qubit_per_setting)


def compile_graph_settings(n,codes,groups,block_size):
    """Compile signed readouts and price the identical immutable circuits."""
    codes = tuple(codes)
    for code in codes:
        if isinstance(code,bool) or not isinstance(code,Integral):
            raise ValueError('integer Pauli codes required')
        validate_word_code(n,int(code))
    codes = tuple(map(int,codes)); ranges = tuple(block_ranges(n,block_size))
    locals_by_block = {(start,size):np.asarray([local_code(c,start,size) for c in codes],np.uint64)
                       for start,size in ranges}
    compiled,resources,two,depths,coverage = [],[],[],[],[]
    compatibility = np.zeros((len(groups),len(codes)),bool)
    assignment = [-1]*len(codes); total = Counter(); checked = 0
    for setting_index,members in enumerate(groups):
        for member in members:
            if assignment[member] != -1: raise AssertionError('duplicate assignment')
            assignment[member] = setting_index
        gates,images_by_block,block_stats = [],[],[]
        readable = np.ones(len(codes),bool); signs = np.ones(len(codes),np.int8)
        for start,size in ranges:
            local = locals_by_block[start,size]
            diagonalizer = graph_diagonalizer([local[i] for i in members],size)
            phases,images = diagonalizer._transform_codes_unchecked(local)
            mask = np.uint64(pauli_lane_mask(size))
            readable &= (((images & mask) ^ ((images >> np.uint64(1)) & mask)) == 0)
            signs *= phases; images_by_block.append((start,size,images))
            gates.extend((name,tuple(start+q for q in qubits)) for name,qubits in diagonalizer.gates)
            block_stats.append(native_stats(diagonalizer))
            checked += len(set(local[i] for i in members))
        native = NativeClifford(n,tuple(gates))
        readouts = {}
        for i in np.flatnonzero(readable):
            positions = tuple(start+q for start,size,images in images_by_block
                              for q in range(size) if (int(images[i]) >> (2*q)) & 3 == 3)
            expected = int(signs[i]),positions
            phase,image = native.conjugate(PauliWord(n,codes[i]))
            if phase != expected[0] or tuple(image.support()) != positions or any(image.letter(q) not in ('I','Z') for q in image.support()):
                raise AssertionError('block signed readout disagrees with global native conjugation')
            readouts[codes[i]] = expected
        if not readable[list(members)].all(): raise AssertionError('assigned word is unreadable')
        compatibility[setting_index] = readable
        key = ('block-commuting','graph_state.v1',n,block_size,tuple(codes[i] for i in members),native.identity)
        compiled.append(CompiledSetting(key,native,tuple(codes[i] for i in members),readouts))
        counts = Counter()
        for block_counts,_,_,_ in block_stats: counts.update(block_counts)
        critical = max((row[1] for row in block_stats),default=0)
        d1 = max((row[2] for row in block_stats),default=0)
        d2 = max((row[3] for row in block_stats),default=0)
        total.update(counts)
        two.append(counts['CZ']);depths.append(critical)
        resources.append(SettingResources(n_1q=sum(counts[v] for v in _LOCAL),n_2q=counts['CZ'],d_1q=d1,d_2q=d2))
        coverage.append(float(readable.mean()) if len(codes) else 0.)
    if any(v < 0 for v in assignment): raise AssertionError('missing word assignment')
    compatibility = immutable_array(compatibility)
    ledger = GraphSynthesis(block_size,tuple(resources),MappingProxyType(total),tuple(two),tuple(depths),
                            compatibility,tuple(assignment),checked,tuple(coverage))
    return tuple(compiled),ledger

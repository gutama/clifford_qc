# Symmetry-adapted excitation pools

This implements the next pool-reduction step in the
[pencil improvement sequence](PENCIL_ESTIMATORS.md#validation-and-remaining-sequence).
It adds opt-in generators and a bounded structural audit. It does not change
the existing pools or establish a finite-shot saving.

## Conventions and construction

Use `n_spatial` spatial orbitals and `n = 2*n_spatial` spin orbitals, with
hbar = 1. Spatial orbital `p` owns qubits `(2p, 2p+1)` under the default
`spin_ordering="interleaved"` and `(p, n_spatial + p)` under `"blocked"`, alpha
first in both. Operators carry no record of their ordering, so every API here
takes it explicitly; it must match how the Hamiltonian and any filtered pool
were built. Operators use the package's native Jordan--Wigner image in complex
`Cl(2n, C)`; qubit zero is the leftmost occupation bit. Each returned generator
is one whole MV.

[`spin_adapted_excitations`](../../clifford_qc/subspace/symmetry_pools.py)
forms spin-free transfers
`E_ai = c†_(a,alpha)c_(i,alpha) + c†_(a,beta)c_(i,beta)`.
The singles are `E_ai - E_ai†`, and doubles are
`E_ai E_bj - (E_ai E_bj)†`. Unordered transfer pairs include repeated indices
and both matchings of four distinct spatial indices. A repeated transfer
automatically supplies the factor two of a paired double excitation.

These operators commute with particle number, Sz and both spin ladders, hence
with S². Applied to a closed-shell determinant, the unfiltered pool spans its
complete singlet singles/doubles space. `occupied_spatial` is an explicit set
of doubly occupied spatial orbitals; it can be non-prefix. The remaining
orbitals are virtual. `closed_shell_occupied_spatial(occupied, n_spatial,
spin_ordering=...)` reads that set from a determinant's occupied spin orbitals
and rejects any singly occupied spatial orbital. The directions are not generally orthogonal. On a
spin-mixed reference the operators preserve its spin sectors; they do not
project it into a singlet. Open-shell-specific couplings and generalized
occupied/virtual rotations remain separate work.

## Point-group characters

`orbital_irreps` is an explicit nonnegative integer character for each spatial
orbital in a real Abelian `(Z2)^k` convention: tensor products are XOR and
zero is the identity. This does not support complex or multidimensional irreps.
FCIDUMP `ORBSYM` and other programs' arbitrary irrep indices are **not**
automatically converted to this convention.

A single is retained when `irrep(a) XOR irrep(i) = 0`. A double uses the XOR
of all four labels. Two forbidden single transfers can make an allowed double;
filtering the singles before enumerating doubles would incorrectly lose it.

[`point_group_filter`](../../clifford_qc/subspace/symmetry_pools.py) also
filters an existing generator list. It checks global operator commutation with
each parity `U_b = product_p Z_(p,alpha) Z_(p,beta)`, over orbitals whose
character contains bit b, with the qubits of each pair given by
`spin_ordering`. Reading a blocked pool as interleaved would keep forbidden
operators and drop allowed ones, which is why the ordering is a parameter. It preserves order and existing Generator objects. A sum with
any forbidden-character component is removed whole, never split or projected.
This filter alone does not add total-spin conservation.

Supply `hamiltonian=` to either API to verify the declared point-group
invariance of the stored Hamiltonian before removing directions. Its words with
nonzero character must have `|coefficient| <= hamiltonian_atol`; the default 0
admits none above the `1e-12` storage threshold, and a rejection reports the
largest violation. Raise the tolerance only to accept integral noise from an
SCF run that did not enforce symmetry; it never relaxes the whole-operator
generator filter. Without a Hamiltonian, that invariance is a caller
assumption. The check concerns the supplied
characters, not an inferred molecular point group.

```python
from clifford_qc.models import fcidump_model
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.subspace import MatrixElementBank, identity_generator
from clifford_qc.subspace.symmetry_pools import spin_adapted_excitations

model = fcidump_model("benchmarks/data/h4_sto3g_r0.9.FCIDUMP")
reference = DeterminantReference.from_program(model.reference)
pool = spin_adapted_excitations(
    4, (0, 1), orbital_irreps=(0, 1, 0, 1), hamiltonian=model.hamiltonian,
)
bank = MatrixElementBank(reference, model.hamiltonian,
                         [identity_generator(model.n)] + pool)
result = bank.solve()
```

The supplied H4 character assignment is checked against this Hamiltonian.
It is not read from its `ORBSYM` header, which has only identity labels.

## Structural audit

[`benchmark_symmetry_pools.py`](../../benchmarks/benchmark_symmetry_pools.py)
compares the existing number/Sz-conserving pool with the spin-free pool.
Explicit point-group characters add separately filtered arms for each pool,
so spin summation and point-group restriction can be assessed independently.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m benchmarks.benchmark_symmetry_pools \
  --orbital-irreps 0 1 0 1 --output work/symmetry-pools-h4.json
```

The audit is bounded to four spatial orbitals and a closed-shell reference.
Both arms excite the model's actual reference determinant: the spin-free arm
reads its doubly occupied spatial orbitals from that reference instead of
assuming an aufbau prefix, and both take the spin ordering from the model's
metadata. It counts the initial identity-versus-each-candidate 2x2 screening cache and
the complete candidate bank, including rejected-sector directions in the
unfiltered comparator. It reports pair builds, coefficient occurrences, word
universes including identity, QWC setting counts, retained ranks and energies.
Each arm starts with a cold kernel cache and records `initial_screen_seconds`
(screen pairs and their ledger), `completion_seconds` (the remaining pairs, the
solve and the complete ledger) and their sum `total_seconds`; every ledger
includes its QWC grouping. The timings are single runs, descriptive rather than
a throughput comparison. `--hamiltonian-atol` sets the Hamiltonian check
tolerance and is recorded with the spin ordering and occupied orbitals. Named
source hashes, the input digest, solver controls, environment and
Git/dirty-tree provenance are recorded. The command line hashes the named
sources before importing the package, so the record binds the loaded code, and
says so in `named_sources_hashed_before_import`; an in-process `build_report`
call hashes them on entry instead and records `false`. Source changes after
import, and input, named-source and Git-state changes during an audit, cause an
explicit failure; source archives leave Git provenance unverified. `work/` is
gitignored, so an earlier output there does not mark a later run's tree dirty.
The named sources are not a complete import closure. Existing outputs cannot be
overwritten.

An exploratory local audit of the committed H4 input gave the following
structural counts. All four complete-bank energies agree within `2e-12` Ha
at `-2.179550752063` Ha (the CISD value, not full CI):

| Pool | Generators, excluding I | Initial screen words | Complete bank words | Complete pairs |
| --- | ---: | ---: | ---: | ---: |
| Number/Sz conserving | 26 | 7,987 | 15,847 | 378 |
| Spin-free | 14 | 9,727 | 15,847 | 120 |
| Number/Sz plus supplied point group | 14 | 4,387 | 7,927 | 120 |
| Spin-free plus supplied point group | 8 | 6,031 | 7,927 | 45 |

Point-group restriction reduces this input's complete word bank. Spin summation
reduces the pair count but increases the initial screening word union; it does
not reduce the complete word union. The LiH CAS(4e,4o) input shows the same
limitation for spin summation alone: 26 to 14 generators, unchanged complete
bank at 15,847 words, and initial screening words increasing from 7,747 to
9,651. Its two CISD energies agree within `2e-12` Ha at `-7.863222354968` Ha.
These are exact, small-system, exploratory structural comparisons. They do
not supersede a frozen experiment or establish an adaptive measurement saving.

The reference and all matrix elements are evaluated exactly on a classical
computer; physical shots are zero. The initial screen is one specified stage,
not the cumulative cost of an adaptive selection trajectory. Neither fewer
generators nor fewer words alone establishes a shot or time saving at matched
error. A sampled campaign with covariance, failures, bias and interval coverage
is still required for that claim.

## Validation

Independent occupation-space ladder signs and S² matrices verify the complete
singlet CISD image span, including point-group restriction, non-prefix
occupations and blocked spin ordering. Other checks cover global
number/Sz/spin-ladder commutation in both orderings, anti-Hermiticity, paired
doubles from forbidden singles, whole-operator filtering against dense parity
commutators in both orderings, mismatched Hamiltonians, the Hamiltonian
tolerance, closed-shell occupation reading, invalid conventions, deterministic
ordering and exact bank energies/counts. The audit's tiny FCIDUMP control checks
separate spin and point-group arms, charges both screening and complete-bank
rows, partitions stage timings and rejects a stale import-time source snapshot.
A non-prefix reference control checks that the spin-free arm reaches the exact
singlet ground energy of the determinant it was given, and the command line
writes strict JSON once.

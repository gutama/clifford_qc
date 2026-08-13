# Molecular Simulation Plan: H₂O, LiH, HF, and BeH₂ via A-CASE and `clifford_qc`

This plan establishes an end-to-end electronic structure and operator-subspace simulation pipeline for four benchmark molecules (**LiH, BeH₂, HF, H₂O**) using `clifford_qc`'s complexified Clifford algebra engine $Cl(2n, \mathbb{C}) \cong M(2^n, \mathbb{C})$.

---

## 1. Pipeline Architecture

```text
+-------------------------------------------------------------------------------+
|                       Upstream Electronic Structure                           |
|  - PySCF (RHF / STO-3G) -> 1-body & 2-body Integrals                          |
|  - Restricted FCIDUMP Ingestion (IUHF=0, chemist (pq|rs) notation)           |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                        clifford_qc Ingestion & Mapping                        |
|  - Load FCIDUMP via strict NumPy adapter (clifford_qc.models.fcidump)        |
|  - Map integrals to JW multivectors via fermion CAR operators (c_j, c_j†)    |
|  - Identify reference Hartree-Fock state |psi_0> & excitation generator pool  |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                         A-CASE Subspace Eigensolver                           |
|  - Rayleigh-Ritz in operator-response subspace basis {A_i |psi_0>}           |
|  - Evaluate matrix elements H_ij and overlap S_ij via single reference        |
|  - Solve generalized eigenvalue problem H c = E S c                          |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       Observables & Response Analysis                         |
|  - Exact sector ground state reference (SectorStatevectorBackend)            |
|  - Ground state energy E0, double occupancy <d>, spin correlations <S_0.S_1> |
|  - Lehmann response spectrum & static susceptibility chi(0)                   |
+-------------------------------------------------------------------------------+
```

---

## 2. Target Molecules & System Configurations

| Molecule | Geometry / Bond Length ($d$) | Basis Set | Qubits ($n$) | Electrons ($N$) | Sector Size $\binom{n/2}{N/2}^2$ | Exact Reference |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **LiH** | $d(\text{Li-H}) = 1.595\text{ \AA}$ | STO-3G | 12 | 4 | 225 | SectorStatevectorBackend |
| **BeH₂** | $d(\text{Be-H}) = 1.326\text{ \AA}$ (linear) | STO-3G | 14 | 6 | 1,225 | SectorStatevectorBackend |
| **HF** | $d(\text{H-F}) = 0.917\text{ \AA}$ | STO-3G | 12 | 10 | 36 | SectorStatevectorBackend |
| **H₂O** | $d(\text{O-H}) = 0.9575\text{ \AA}$, $\theta = 104.51^\circ$ | STO-3G | 14 | 10 | 441 | SectorStatevectorBackend |

---

## 3. Workflow Steps

1. **Environment Setup & Verification:**
   - Install `pyscf`, `openfermion`, `openfermionpyscf`.
   - Run `check_env.py` and `verify.py`.

2. **FCIDUMP Generation:**
   - Compute Hartree-Fock STO-3G integrals using PySCF for LiH, BeH₂, HF, and H₂O.
   - Dump restricted FCIDUMP files into `molecular_results/`.

3. **`clifford_qc` A-CASE Execution:**
   - Ingest each FCIDUMP with `fcidump_model`.
   - Perform exact sector diagonalization to establish benchmark ground energy $E_0^{\text{exact}}$.
   - Execute adaptive A-CASE (`run_acase`) using determinant excitation generators (SD pool).
   - Evaluate matrix element bank (`MatrixElementBank`), projected observables ($\langle d \rangle$, $\langle S_0 \cdot S_1 \rangle$, $\langle S^2 \rangle$), and Lehmann response spectrum.

4. **Results Aggregation & Reporting:**
   - Save all JSON records and summary files in `molecular_results/`.
   - Generate a full Markdown report summarizing energies, errors, basis sizes, correlation functions, and resource metrics.

---

## 4. Multiresolution status: what is and is not wired in

Steps 1–4 above describe the pipeline **as committed**. No wavelet or
multiresolution stage is part of it. `run_molecular_pipeline.py` contains no
reference to Haar packets, configuration packets, or orbital bases, and the
committed records in `molecular_results/` carry no packet fields. The two
multiresolution capabilities the repository does own sit at different distances
from this pipeline, and they must not be described as one capability.

### 4.1 Configuration-space Haar — reachable, but the driver is Hubbard-gated

The transform itself is model-agnostic:
`subspace.configuration.configuration_haar_packets` builds an orthogonal finite
tree-Haar basis over any caller-ordered list of configuration generators, and
`configuration_generator(reference, target)` accepts any reference determinant.
Nothing in either function is specific to a lattice.

The *driver* is what blocks molecules. In `benchmarks/run_configuration_packets.py`:

- `ordered_sector_configurations` raises `ValueError` unless
  `model.metadata["kind"] == "fermionic_lattice"`. FCIDUMP models are
  `kind: "molecular"` (`clifford_qc/models/fcidump.py:350`), so the call fails
  before any transform runs.
- `physics_order_key` defines configuration-space locality as excitation rank,
  then doublon count, then charge and spin patterns **over lattice sites**.
  Molecules have no site index and no doublon count; the ordering heuristic has
  to be replaced, not merely re-parameterised.
- `_level4_pool` builds its configuration tier from `competing_orders(model)`
  (`clifford_qc/models/lattice.py:503`), also gated to `fermionic_lattice`.
  The molecular pipeline has no analogue of SDW/CDW competing orders; its
  configuration content is the SD excitation pool.

Sector size is *not* a blocker. The packet tier caps at 4096 configurations and
the four molecules span 36–1225:

| Record | Qubits | Sector | Adaptive `M` | SD candidates | Adaptive error (mHa) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `hf` | 12 | 36 | 5 | 35 | 1.072 |
| `lih` | 12 | 225 | 6 | 92 | 1.126 |
| `lih_stretched` | 12 | 225 | 5 | 92 | 1.521 |
| `h2o` | 14 | 441 | 21 | 140 | 1.448 |
| `h2o_stretched` | 14 | 441 | 21 | 140 | 55.421 |
| `beh2` | 14 | 1225 | 11 | 204 | 1.341 |
| `beh2_stretched` | 14 | 1225 | 31 | 204 | 28.514 |

So this is a driver and ordering problem, not a mathematics or feasibility
problem.

### 4.2 Orbital wavelets — excluded, on two independent grounds

**Falsified.** `benchmarks/reference_results/orbital_basis.json` records that no
single-particle basis wins uniformly, and that every tested Daubechies basis is
Pareto-dominated: on the clean ring `site` needs 41 Pauli words and 3658
determinants against `db1`'s 363/4310, and `db2` stays dominated by `site` at
every disorder strength tested. Orbital basis is a recorded model parameter.

**Structurally unavailable here anyway.** `clifford_qc/models/orbital.py`
rotates a single-band Hubbard cluster: `rotate_model(one_body, onsite_u, basis)`
with `rotate_onsite_interaction` handling an **onsite** interaction. A molecular
FCIDUMP carries a full four-index `(pq|rs)` tensor, and no ERI rotation exists in
the package. Adding one is an O(N⁵) integral transform plus a rewritten JW
builder — real work, undertaken to re-test a hypothesis the repository already
falsified.

**Decision: orbital wavelets stay out of the molecular study.** Reopening them
requires a specific falsifiable reason from a new system, per
`PLAN.md` §2.5.

### 4.3 Relationship to the submitted manuscript

arXiv:2608.00560 (*Adaptive operator-generated subspaces for effective many-body
Hamiltonians*) makes no wavelet or multiresolution claim; its chemistry evidence
is linear H₄ in STO-3G with CAS(4e,4o) plus the benchmark ladder. Any packet
result produced here is therefore **new work extending the submitted paper**,
not a correction to it, and must not be back-described as having been part of
it.

---

## 5. Revised workflow: adding the configuration-Haar tier

Steps 1–4 of §3 are unchanged and remain the baseline. The following steps are
additive, and each one is separately abandonable.

5. **Generalise the configuration driver beyond lattices.**
   Lift `ordered_sector_configurations` out of `benchmarks/` and admit
   `kind: "molecular"`, taking `n_spatial_orbitals`, electron count, and `S_z`
   from FCIDUMP metadata instead of `sites`. The 4096-configuration cap stays:
   this is a small-sector tier by construction.

6. **Define a molecular ordering heuristic — and declare it as a heuristic.**
   The Haar construction does not infer a metric on determinants; the caller
   supplies one. The lattice key (excitation rank → doublons → charge → spin)
   has no molecular meaning. Proposed replacement, in order: excitation rank
   from the RHF reference, then seniority (unpaired-electron count), then
   summed RHF orbital-energy of the occupied set, then the occupation bitmask
   as a deterministic tiebreak.

7. **Run the ordering ablations before reporting any packet number.**
   Required by `PLAN.md` Phase 11C. Compare at least: the §6 physics
   order, excitation rank alone, orbital-energy order alone, and randomised
   order controls with fixed seeds. A packet result that survives only one
   favourable ordering is an artefact, and is reported as one.

8. **Stage packets, then hand off to the convergence-complete pool.**
   Support-pruned Haar details form an opt-in coarse tier during early growth;
   the retained coarse directions are then handed to the ordinary SD pool
   already used in §3. Packets never replace the convergence-complete family.

9. **Report at matched resources, not matched accuracy.**
   Every packet row carries the fields the baseline already emits —
   `element_word_universe`, `max_generator_support`,
   `max_hamiltonian_element_support`, `condition_number`, `subspace_size_m`,
   peak RSS, wall seconds — so the Hubbard trade-off can be tested for
   recurrence: on the 2×2 cluster the packet arm reached the exact sector energy
   with 33.1 % fewer projected Pauli words and one fewer basis direction, while
   raising maximum element support (1152 → 1224) and `κ(S)` (1.0 → 12.47).
   A molecular repeat of *that* trade is the result; a bare energy is not.

### 5.1 The classical control is already present

`PLAN.md` Phase 9 makes an excitation-closure control mandatory before
any hybrid novelty claim. The molecular pipeline already ships it: the
complete-SD arm (`complete_sd`, 35–204 candidates) is exactly the determinant
excitation closure of the same SD pool, and it already reproduces PySCF CISD.
Any packet arm is therefore judged against a control that exists, at no
additional implementation cost.

### 5.2 Where a coarse tier could plausibly help — and where it cannot

Four of the seven committed records already reach chemical accuracy with
`M` = 5–21, so there is little headroom to win. The honest targets are the two
stretched rows where adaptive growth fails outright — `h2o_stretched` at 55.4
mHa and `beh2_stretched` at 28.5 mHa, both hitting the subspace cap rather than
an accuracy stop. If coarse-to-fine staging cannot move those two, it does not
help molecules, and that is a publishable negative result of the same kind as
§4.2.

Cost is the binding practical constraint, not correctness: the committed
`beh2_stretched` run took 2864 s at 11.5 GB peak RSS, and `h2o` 4320 s at
8.4 GB. Ordering ablations multiply that by the number of orderings tested, so
sequence the work `hf` → `lih` → `lih_stretched` first (76–514 s each) and only
promote an ordering to the 14-qubit rows once it has survived the cheap ones.

---

## 6. Go/no-go

The configuration-Haar tier enters the molecular study only if **all** of these
hold:

1. the generalised driver reproduces the committed `hubbard_2x2` packet record
   bit-for-bit, proving the lattice path was refactored and not rewritten;
2. unpruned packets preserve `S = I` and the exact sector energy on `hf`,
   confirming orthogonality survives the molecular ordering;
3. at least one molecular row shows a resource gain at equal or better energy;
4. that gain survives the §7 ordering ablations, including random controls.

If (3) or (4) fails, the recorded outcome is that configuration-space Haar
staging is a Hubbard-specific result that does not transfer to molecular
determinant spaces — reported, not withheld.

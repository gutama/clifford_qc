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

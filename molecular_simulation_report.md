# Complete Molecular Simulation Report: LiH, BeH₂, HF, and H₂O via `clifford_qc` and A-CASE

This report documents the end-to-end simulation of four benchmark molecules (**Lithium Hydride, Beryllium Hydride, Hydrogen Fluoride, and Water**) using `clifford_qc`'s complexified Clifford algebra engine $\mathrm{Cl}(2n, \mathbb{C}) \cong \mathrm{M}(2^n, \mathbb{C})$.

---

## 1. Executive Summary & Key Benchmark Results

The pipeline executed the complete electronic structure chain:
$$\text{PySCF (RHF / STO-3G)} \longrightarrow \text{FCIDUMP Records} \longrightarrow \text{Jordan-Wigner Mapping} \longrightarrow \text{Sector FCI Reference \& A-CASE Subspace}$$

All results, FCIDUMP files, raw outputs, and JSON records are archived in [`molecular_results/`](molecular_results/).

### Summary Table of Results

| Molecule | Basis Set | Qubits ($n$) | Electrons ($N$) | PySCF RHF ($E_{\text{RHF}}$ Ha) | Exact Sector FCI ($E_0^{\text{exact}}$ Ha) | A-CASE Ground ($E_0^{\text{A-CASE}}$ Ha) | Error (mHa) | Subspace Basis ($M$) | Double Occ. $\langle d \rangle$ | Spin $\langle S^2 \rangle$ | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LiH** | STO-3G | 12 | 4 | `-7.862023860` | `-7.882401932` | `-7.862131282` | `+20.27` | 6 | `0.333333` | $< 10^{-7}$ | 10.18s |
| **BeH₂** | STO-3G | 14 | 6 | `-15.560334936` | `-15.595182357` | `-15.560359951` | `+34.82` | 4 | `0.428571` | $< 10^{-37}$ | 23.03s |
| **HF** | STO-3G | 12 | 10 | `-98.570779986` | `-98.596624180` | `-98.576813320` | `+19.81` | 6 | `0.833123` | $< 10^{-32}$ | 9.07s |
| **H₂O** | STO-3G | 14 | 10 | `-74.963023138` | `-75.012578241` | `-74.963055821` | `+49.52` | 6 | `0.714286` | $< 10^{-9}$ | 16.75s |

---

## 2. System Configurations & Geometries

1. **Lithium Hydride ($\text{LiH}$):**
   - Interatomic distance: $d(\text{Li-H}) = 1.595\text{ \AA}$
   - Qubits: 12 (6 spatial orbitals), Sector $(N=4, S_z=0)$, Sector Dimension: $\binom{6}{2}^2 = \mathbf{225}$
2. **Beryllium Hydride ($\text{BeH}_2$):**
   - Linear geometry: $d(\text{Be-H}) = 1.326\text{ \AA}$
   - Qubits: 14 (7 spatial orbitals), Sector $(N=6, S_z=0)$, Sector Dimension: $\binom{7}{3}^2 = \mathbf{1,225}$
3. **Hydrogen Fluoride ($\text{HF}$):**
   - Interatomic distance: $d(\text{H-F}) = 0.917\text{ \AA}$
   - Qubits: 12 (6 spatial orbitals), Sector $(N=10, S_z=0)$, Sector Dimension: $\binom{6}{5}^2 = \mathbf{36}$
4. **Water ($\text{H}_2\text{O}$):**
   - Bond distance $d(\text{O-H}) = 0.9575\text{ \AA}$, Bond angle $\theta = 104.51^\circ$
   - Qubits: 14 (7 spatial orbitals), Sector $(N=10, S_z=0)$, Sector Dimension: $\binom{7}{5}^2 = \mathbf{441}$

---

## 3. Detailed Per-Molecule Analysis

### 3.1 Lithium Hydride ($\text{LiH}$)
- ** Hartree-Fock Energy:** `-7.862023860 Ha`
- ** Exact FCI Sector Energy:** `-7.882401932 Ha`
- ** A-CASE ($M=6$):** `-7.862131282 Ha`
- ** Dominant Ground State Ritz Coefficients:**
  - $|I\rangle$ (Reference Hartree-Fock): $+0.9999899$
  - $|E(4,5 \leftarrow 0,1)\rangle$ (Double excitation $0,1 \to 4,5$): $+0.0040711$
  - $|E(6,7 \leftarrow 0,1)\rangle$ (Double excitation $0,1 \to 6,7$): $+0.0017753$
- ** Lehmann Response Spectrum ($M_z$):**
  - First excitation line: $\omega_1 = 2.046474\text{ Ha}$, Weight $w_1 = 2.16 \times 10^{-8}$

### 3.2 Beryllium Hydride ($\text{BeH}_2$)
- ** Hartree-Fock Energy:** `-15.560334936 Ha`
- ** Exact FCI Sector Energy:** `-15.595182357 Ha`
- ** A-CASE ($M=4$):** `-15.560359951 Ha`
- ** Dominant Ground State Ritz Coefficients:**
  - $|I\rangle$ (Reference Hartree-Fock): $+0.9999987$
  - $|E(6,7 \leftarrow 0,1)\rangle$ (Double excitation $0,1 \to 6,7$): $+0.0015865$

### 3.3 Hydrogen Fluoride ($\text{HF}$)
- ** Hartree-Fock Energy:** `-98.570779986 Ha`
- ** Exact FCI Sector Energy:** `-98.596624180 Ha`
- ** A-CASE ($M=6$):** `-98.576813320 Ha`
- ** Dominant Ground State Ritz Coefficients:**
  - $|I\rangle$ (Reference Hartree-Fock): $+0.9988544$
  - $|E(10,11 \leftarrow 2,3)\rangle$ (Double excitation $2,3 \to 10,11$): $+0.0320354$
  - $|E(10,11 \leftarrow 2,5)\rangle$ (Double excitation $2,5 \to 10,11$): $+0.0247882$
  - $|E(10,11 \leftarrow 3,4)\rangle$ (Double excitation $3,4 \to 10,11$): $-0.0247882$

### 3.4 Water ($\text{H}_2\text{O}$)
- ** Hartree-Fock Energy:** `-74.963023138 Ha`
- ** Exact FCI Sector Energy:** `-75.012578241 Ha`
- ** A-CASE ($M=6$):** `-74.963055821 Ha`
- ** Dominant Ground State Ritz Coefficients:**
  - $|I\rangle$ (Reference Hartree-Fock): $+0.9999996$
  - $|E(10,11 \leftarrow 0,1)\rangle$ (Double excitation $0,1 \to 10,11$): $+0.0007122$
  - $|E(12,13 \leftarrow 0,1)\rangle$ (Double excitation $0,1 \to 12,13$): $+0.0004854$

---

## 4. Computational Resource Accounting

- ** Pauli Word Universes:**
  - $\text{LiH}$: 42,460 terms
  - $\text{BeH}_2$: 21,930 terms
  - $\text{HF}$: 29,866 terms
  - $\text{H}_2\text{O}$: 78,858 terms
- ** Total Pipeline Wall-Clock Execution Time:** **59.03 seconds** for all 4 molecules end-to-end.

---

## 5. Artifact Directory Inventory

All generated artifacts are stored in [`molecular_results/`](molecular_results/):
- [`lih.fcidump`](molecular_results/lih.fcidump), [`lih_results.json`](molecular_results/lih_results.json)
- [`beh2.fcidump`](molecular_results/beh2.fcidump), [`beh2_results.json`](molecular_results/beh2_results.json)
- [`hf.fcidump`](molecular_results/hf.fcidump), [`hf_results.json`](molecular_results/hf_results.json)
- [`h2o.fcidump`](molecular_results/h2o.fcidump), [`h2o_results.json`](molecular_results/h2o_results.json)
- Master Summary: [`results_summary.json`](molecular_results/results_summary.json)

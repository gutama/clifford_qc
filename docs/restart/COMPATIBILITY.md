# Checked first-PR compatibility scope

This implements Steps 1–2 of the [restart blueprint](blueprint_2026-10-24.md)
under the [v2 contract](CONTRACT.md). Historical campaign records, digests,
ancestry, and negative conclusions are unchanged. The v1 bindings are recorded
at their reviewed ancestor in [v1_evidence_snapshot.json](v1_evidence_snapshot.json).

## Numerical and ownership checks

| Path | Checked scope | Acceptance |
| --- | --- | --- |
| Analytic determinant | Every occupation pattern at 0–4 qubits; all Pauli words; complex coefficients | Independent letter-by-letter sign/off-diagonal calculation |
| Trace-compatible determinant and MV adapter | Both smaller-operand branches; forward/reversed emissions; non-prefix occupations; cancellation, signed zero, tiny/large finite coefficients; generic MV state with a Y component | Complex scalar bytes equal the legacy trace pairing |
| Exact projected provider | Three-qubit complex controls; determinant/MV adapters; object and packed interleaved/SoA; retain-all and streaming | S/H and Ritz-energy bytes equal the existing density-MV bank; subset order retained |
| Packed scalar evaluation | Rows constructed before scalar reads, including reordered emissions | No packed-row materialization or legacy density-pairing call |
| Prepared molecular workflow | H4 STO-3G r=0.9 and LiH STO-3G r=1.5949 CAS(4e,4o); 14-candidate cap, two additions; object/packed × retain/stream | S/H and energy-history bytes, labels, predicted-lowering/conditioning decisions, and stopping match explicit `legacy_mv` |
| Analytic molecular workflow | Same two bounded fixtures and candidate/addition budgets | S/H and energy history agree at `rtol=atol=1e-12`; same labels and stopping |
| Large-register reference | 4,096 qubits and four supplied Pauli terms, analytic arithmetic | Independent expectation; density/backend construction, dense conversion and NumPy allocation entry points forbidden; zero density coefficients |
| Input/cache ownership | Caller mutates original density, H and generators; attempts mutation/deletion through public cached inputs | Owned snapshots remain unchanged; identities distinguish coefficient order, arithmetic, conventions and storage policy |
| Capability boundary | Sampled-only reference; changed custom reference identity; unmigrated reporting options; incompatible prepared spin/encoding conventions | Visible rejection before unsupported work; no implicit density fallback |

The byte comparisons are observations on these declared fixtures and the named
`legacy_trace` contract. They are not a universal bitwise equivalence promise
for analytic arithmetic, other algorithms, platforms, or input scales. A
subnormal regression explicitly demonstrates why the two arithmetic modes are
separate. Trace compatibility rejects 40 or more qubits, where the historical
MV pruning threshold removes the computational-density coefficients.

The 4,096-qubit test evaluates a bounded row only. It does not run a molecular
Hamiltonian, build a candidate pool, or solve a large projected problem.
Operator products and coefficient storage keep their existing scaling.

## Validation and accounting

The three new test modules contain 96 cases. Local checks use Python 3.12.14,
NumPy 2.5.2, SciPy 1.18.0, and one BLAS/OpenMP thread. Reproduce the acceptance
checks with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_reference_states.py \
  tests/test_exact_reference_provider.py tests/test_restart_pipeline.py \
  --hypothesis-profile=ci
```

Run the existing subspace, contraction, molecular, convergence, sector and
architecture tests as regression checks. Automatic CI remains responsible for
the full suite and record gates; collection totals in `REPRODUCING.md` are
expected green-suite accounting, not a claim of a full local execution.

New provider records separate reference occupation/snapshot payload, row terms
scanned, diagonal hits (not applicable for generic MV references), evaluation
calls, compatibility sorting and peak retained index footprint. The input H
and generator coefficient payload is also identified separately. Index figures
include referenced boxes that may overlap row storage; they exclude transient
sorting workspaces, Python allocator effects and process RSS. Whole-solve wall
time remains separate. No matched-accuracy timing, RSS, or quantum-resource
advantage is claimed by this PR.

Reference-conditioned sector certification and H-squared convergence reporting
still require explicit `reference_mode="legacy_mv"`. Step 3 must add the action
consumer, owned-memory admission, root recovery, spin diagnostics and full-action
residuals. Step 5 must add an actual sampled consumer and independent pilot and
evaluation observations; an exact oracle is not a sampled implementation.

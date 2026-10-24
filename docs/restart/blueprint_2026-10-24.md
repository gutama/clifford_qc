# Restarting clifford_qc: a selective rebuild

Review date: 24 October 2026, Asia/Jakarta. Reviewed main at [89d3cff](https://github.com/gutama/clifford_qc/commit/89d3cffd25aa86b2d155a74291486800f17b31ce), including merged PRs #149–153.

This is a source, committed-evidence, and architecture review. Historical campaigns and the repository test suite were not rerun. Small independent calculations checked coordinate invariance and memory formulas. Recommendations below are proposed work, not demonstrated improvements. The 20 October improvement review was read and compared with subsequent results.

Implementation refinement: 24 October 2026. The agreed first PR covers the determinant-reference and exact-projection seams in Steps 1–2. Root recovery follows in Step 3; the measurement experiment follows in Step 5. The evidence snapshot above is unchanged. The contracts and acceptance checks below specify proposed implementation work.

## Recommendation

Restart the scientific specification and public workflow while reusing verified components. Build one narrowly scoped tool:

> Given a declared Hamiltonian, a usable reference state, and a correction family, estimate an energy or observable to a stated tolerance while reporting its full classical and measurement cost.

The central question should be whether operator-response measurements add useful information at a lower total cost for a defined workload. Clifford algebra remains a precise operator language. It is not, by itself, an accuracy or complexity improvement.

An analogy from compiler design is helpful: the Clifford/Pauli representation is an intermediate language; state representations and measurement devices are execution backends; eigensolvers and estimators are consumers. The compiler should preserve meaning and expose costs. A better intermediate language cannot alone make an insufficient approximation space accurate.

Much of this separation is already acknowledged in PLAN and ARCHITECTURE. The restart should enforce it in the default entry points and interfaces rather than introduce another collection of algorithm names.

## Evidence that determines the restart

| Current evidence | What it supports | Restart consequence |
|---|---|---|
| The molecular CLI constructs rank-1/rank-2 determinant excitations around a determinant reference. The molecular report gives stretched BeH2/H2O A-CASE errors of 28.514/55.421 mHa, close to the corresponding CISD errors of 28.511/54.570 mHa. | This candidate family is constrained by its excitation span. | Change the reference/correction space when accuracy fails. Cache optimization cannot remove this bias. |
| In the older matched-M=7 exact suite, Davidson meets the declared accuracy threshold on 9/12 inputs; A-CASE and its matched selected-CI arm meet it on 0/12. | Davidson is a strong classical accuracy control on that suite. | Keep A-CASE as a comparator and measurement construction; establish a mature classical solve first. |
| The latest primary 2x4 Hubbard campaign ends at 512 columns with packet error 0.0762604 t, 215,421 H actions, and median setup+solver time 1,326.24 s. Internal selected CI gives 0.247694 t, 1,764 actions, and 134.28 s. | Packets improve the final fixed-budget error while spending much more work. Neither reaches 0.0016 t. | These are not matched-accuracy speed comparisons. Reopening packets needs a new measurable-support cost hypothesis. |
| On that primary, the Davidson reference stops with residual 9.21e-9 at an excited singlet 0.994928 t above the ground state. The initial span does have nonzero ground-state weight. | Small residual does not identify the ground state; weak-start excited-root convergence is a real failure. | Make root evidence and recovery explicit in every solve. Do not attribute this result to a span that excludes the ground state. |
| The packet line is explicitly closed in the latest PLAN. Frame-coordinate projection reduces primary fixed-budget time by 50.22% exhaustive and 36.83% early, but its frozen joint criterion fails on shortlist ordering. | Useful execution improvements exist; they do not overturn the scientific verdict. | Reuse validated mechanics when useful, preserve failed historical criteria, and stop treating cache work as a new solver hypothesis. |
| The frozen packet-Davidson complete-bank finite-shot study reports primary median errors at 1M shots of 6.35 mHa on stretched H4, 0.02800 t on Hubbard 2x2, and 196.80 mHa on stretched H2O. | A compact exact space does not establish a useful measured estimator. | Price measurement and inference before expanding exact solver experiments. |
| Phase 19 compares fixed Hartree–Fock energy functionals. Clique shot outcomes differ by molecule, and illustrative hardware costs can reverse a shot advantage. | The measurement machinery produces informative, bounded comparisons. | Preserve this machinery; test reuse and end-to-end cost separately before widening claims. |

Sources: [molecular report](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/molecular/report.md), [current plan](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/PLAN.md), [column-budget campaign](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/benchmarks/results/packet_budget_convergence.md), [frame-coordinate report](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/benchmarks/results/frame_coordinate_projection.md).

The revised orthogonal-active-space pilot also contains a useful control: same-span coordinate changes preserve exact energies to approximately 5e-13 Ha, and support-containing determinant spaces are at least as accurate as the packet spaces. Smaller column counts need support, selection, storage, and measurement pricing before they become a resource benefit. [Pilot report](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/benchmarks/results/orthogonal_active_space_v2.md).

## What to keep, redesign, and retire

| Component | Action | Reason |
|---|---|---|
| Pauli/Majorana conventions, sparse products, mappings, independent sign checks | Keep | Foundational correctness work with broad reuse. |
| Model metadata, PreparedProblem identity, FCIDUMP preparation, reusable reference validation | Keep | Good input contracts and provenance already exist. |
| Sector actions, sparse restrictions, spin diagnostics, exact Ritz and residual helpers | Keep | Useful classical building blocks; give them clear execution boundaries. |
| Shared word caches, joint histograms, covariance, compiled readouts, allocations, sample-split coupling certification | Keep | Substantial measurement infrastructure already implemented. |
| Packed rows and streaming | Keep as explicit policies | Measured tradeoffs exist; neither is a universal speed improvement. |
| MatrixElementBank and the MV-typed backend contracts | Redesign interfaces | Exact scalar evaluation, coefficient compilation, state storage, and row lifetime are too tightly coupled for a representation-neutral engine. |
| ActionSpace and the single-vector Davidson comparison arm | Extend or wrap mature numerical software | Add bounded search spaces, block/multistart recovery, and explicit root status; avoid rewriting an eigensolver ecosystem. |
| Public API and workflows | Narrow | One supported path per task is easier to validate than many experimental entry points. |
| Historical research campaigns | Archive and preserve | They remain scientific evidence and should not determine active source layout. |
| Packet optimization as the default new research direction | Retire on the present scope | The current declared comparison has answered it negatively. |
| Broader bridges, new mapping/clique/packet variants | Defer unless needed by the first experiment | Each should serve a concrete workload or measured hypothesis. |

The generated census records 117 package modules, 30,355 package code lines, 167 test modules, and 73 benchmark checker entry points, with 54 checkers configured for every PR. These are census definitions, not physical-file totals. PLAN has 6,201 lines. This is meaningful research infrastructure, but a restart should expose a small supported surface. Counts were inspected, not regenerated. [Census](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/paper_architecture/data/source_census.json).

## Change 1: make execution independent of density-MV storage

The present molecular solve calls ExactMVBackend and constructs an MV density. MatrixElementBank takes that MV reference. Backend and SamplingBackend also expose MV in their state contracts; a sector backend exists but is not interchangeable with this route.

For a computational determinant,
`rho = product_j (I + s_j Z_j)/2`.
Its explicit Pauli expansion has 2^n nonzero terms, although the determinant can be specified by n occupation bits. A generic operator can require 4^n Pauli coefficients. This is a representation cost, not a property that the quantum state must be stored that way.

### Reference and provider contracts

Introduce a small ReferenceState metadata/capability contract carrying the register size, bit/spin ordering, immutable reference identity, and supported operations. Exact expectation evaluation is a separate capability: a sampled reference must not be required to implement it.

| Adapter | Representation and capabilities | Implementation stage |
|---|---|---|
| DeterminantReference | Occupations and analytic Pauli-row expectations | First PR: working adapter |
| MVReference | Existing density-MV semantics and legacy arithmetic | First PR: compatibility adapter |
| SectorStateReference | Amplitudes, declared sector/ordering, and exact actions or expectations | Implement with the classical action consumer in Step 3 |
| SampledReference | Preparation identity, sampling capabilities, batches, shot provenance, and uncertainty | Implement with the measured consumer in Step 5 |

Define the sector and sampled capability boundaries in the first PR; add their concrete adapters with real consumers. A sampled reference exposes neither exact amplitudes nor exact expectations unless an explicitly separate diagnostic oracle is provided. Unsupported operations fail visibly. Preserve Hamiltonian/reference fingerprints in every cache identity, and prevent mutation from silently invalidating cached values.

Separately expose ProjectedProblemProvider capabilities:

1. ActionProvider supplies numerical S/H blocks or H actions without compiling measurement rows; its action-based residual uses retained vectors and actions.
2. MeasurementProvider compiles coefficient maps and reconstructs the required quantities from actual observations, carrying uncertainty and sampling cost.

Both feed the existing stable small-matrix Ritz solver. The cyclic contraction option, ActionSpace, and shared measurement classes are starting points. The first PR integrates the determinant exact-evaluation path and its MV compatibility adapter; later consumers supply the full action and measured workflows.

### Determinant expectation seam

For a Pauli word W represented by declared masks x and z and a computational determinant b,

`<b|W|b> = 0` if `x != 0`, and
`<b|W|b> = (-1)**popcount(z & b)` otherwise.

Pauli-word phases remain in the row coefficients. Translate masks and b through the repository's existing qubit/occupation convention rather than assuming integer-endian agreement.

A determinant reference stores O(n) occupation bits and does not construct rho. An arbitrary row can be scanned in O(nnz(row)) time. An indexed diagonal view permits O(k) expectation work for k terms with x=0; building or storing that view and any compatibility-order bookkeeping have their own costs. Do not claim O(k) for an unindexed scan of all terms.

### Historical arithmetic and acceptance

Operator emission order alone does not guarantee bitwise agreement with MV.trace_pairing: that implementation can iterate the smaller dictionary, including the density's generation order. PackedRowStore also preserves a generation permutation. Multiplication, power-of-two scaling, complex accumulation, and signed zeros can affect exact comparisons.

Trace and test both operand-size branches before changing evaluation. Where required and feasible, reproduce reference-generation order implicitly without allocating its 2^n terms. Keep historical scaling and summation operations in a clearly identified compatibility path; price any sorting or bookkeeping.

The acceptance report must distinguish bitwise-identical legacy fixtures from numerically equivalent cases under declared tolerances. Do not assert universal bitwise identity from the analytic Pauli formula. If a required bitwise fixture fails, retain the legacy path for that scope until the difference is explained and any new numerical contract is declared.

Tests cover arbitrary determinant bit patterns, complex coefficients, cancellation, signed zeros, reordered emissions, packed generation order, both smaller-operand branches, reference identity, and supported extreme scales. A large-qubit analytic test must forbid density construction and use a bounded-size row; compare to analytic expectations there, reserving expanded-MV comparisons for small systems.

Source: [pipeline](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/clifford_qc/pipeline.py), [projection](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/clifford_qc/subspace/projection.py), [backend protocols](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/clifford_qc/backends/protocol.py), [pipeline guide](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/molecular/PIPELINE.md).

## Change 2: establish a reliable classical reference path

Before making new quantum claims, make the primary Hubbard failure a regression case for a strengthened classical workflow.

Use a matrix-action interface with block application and optional diagonal/preconditioning. Evaluate a mature solver adapter such as PRIMME, alongside existing sector exact controls. PRIMME supports Hermitian standard/generalized eigenproblems, multiple target positions, and preconditioning; it does not guarantee a ground state for every supplied start or solve the representation problem.

The supported workflow should include:

- Explicit sector and root target.
- More than one initial direction when appropriate, with deterministic seeded starts and documented classical reference quality. Candidate starts include low-diagonal configurations and Neel/selected-CI directions, projected into the declared spin sector.
- A bounded basis with block, restart, and recovery policies; count all starts, projections, deflation, restarts, and recovery actions. These reduce trapping risk without guaranteeing discovery of the ground root.
- Numerical convergence separate from ground-state identification.
- A clear unresolved/resource-limited outcome when the requested accuracy has not been established.

Existing exact convergence reports already distinguish residual evidence and ground-state evidence. Make that distinction universal in the main results API. A residual certifies proximity to some eigenvalue under the usual Hermitian assumptions. Independent spectral information is needed to identify the ground root.

Decouple the common ConvergenceReport construction from MatrixElementBank and SecondMomentBank. An action consumer can compute `psi = V c` and `r = HV c - E V c` from its existing buffers, reporting `||r||_2 / ||V c||_2` with explicit normalization and numerical-resolution checks. The exact action route must not build H-squared solely to populate the report. Retain the second-moment route for consumers that actually supply that information, and record the residual source.

Separate multiplicity checks from root tracking. Check particle number, S_z, and the target total-spin sector on retained Ritz vectors. For general target spin S, an S-squared expectation alone can conceal mixtures: use projection or the action defect `||(S^2 - S(S+1)) psi||` where available. A pure excited singlet can pass every spin check. Ray/subspace overlaps support root tracking, including degenerate blocks; they do not certify that a tracked state is the lowest root.

Step 3 must include the observed 2x4 Hubbard excited-singlet stop and an independent spin-contamination regression. Use known small-system spectra only for assessment, never to choose production corrections. Report numerical convergence, quantum-number consistency, root tracking, and ground-state evidence separately. If recovery does not establish the requested root, retain an unresolved outcome and charge every attempt.

For chemistry, compare with PySCF FCI/CASCI where feasible and an established selected-CI implementation such as SHCI/Dice. Preserve internal selected CI as a matched scientific control. Its timing does not represent an optimized external package. Keep variational and perturbatively corrected energies separate.

Sources: [ActionSpace](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/clifford_qc/subspace/action_space.py), [comparison solver](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/clifford_qc/subspace/sparse_baselines.py), [PRIMME](https://github.com/primme/primme), [PySCF CI/FCI](https://pyscf.org/user/ci.html), [Dice SHCI](https://github.com/sanshar/Dice/blob/master/README_shci.md), [SciPy LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html).

## Change 3: design for the complete memory boundary

Matrix-free H removes a D-by-D matrix requirement. It does not remove the D-by-M basis or Hamiltonian-action storage, and it does not make full-sector methods polynomial.

Calculated complex128 payloads for both V and HV at M=512:

| Half-filled Hubbard sites | Fixed N-up/N-down sector D | V plus HV |
|---|---:|---:|
| 8 | 4,900 | 0.0748 GiB |
| 10 | 63,504 | 0.9690 GiB |
| 12 | 853,776 | 13.0276 GiB |

These are array sizes excluding workspaces, candidates, H storage, spin frames, Python objects, and RSS overhead. At ten sites a dense complex128 sector H alone would take 60.09 GiB.

Bound the retained search space, candidate buffers, coefficient stores, functional snapshots, and logs. ActionSpace currently column-stacks its arrays and reforms the projected matrix.

For the bounded action solver, admit fixed-capacity `(D, M_max)` V and HV buffers, or explicitly budgeted blocks, before allocation. Expose `max_owned_bytes` for accounted arrays and a separate monitored process-RSS ceiling such as `max_rss_bytes`. The admission estimate must include projected matrices, candidate/reorthogonalization workspaces, and planned restart storage in addition to the `2 D M_max * itemsize` basis/action payload. Reject or reduce capacity before allocating when the owned budget cannot admit the plan.

Use live-prefix views and fill new projection borders, for example `H_sub[i,new] = v_i† (H v_new)` and `S_sub[i,new] = v_i† v_new`, instead of copying and reprojecting the complete history at each extension. Maintain Hermiticity and Gram diagnostics; S=I is available only under a verified orthonormal-basis contract. Ensure old snapshots cannot alias mutable buffers. Define tolerances for changed accumulation order rather than assuming bitwise trajectory preservation.

An owned-array admission check cannot guarantee a process-RSS ceiling: BLAS workspaces, allocator retention, transient arrays, logging, and monitor latency remain relevant. Record measured RSS separately, provide declared headroom and monitored termination, and retain limit breaches as failures. Enforcement strength, including any OS-enforced limit, belongs in the run metadata.

The completed Phase 2M matrix is valuable evidence: stretched-H2O M=31 uses 9.873/3.666/3.201/2.626 GiB peak RSS for object-retain/packed-retain/object-stream/packed-stream, with parent times 14.39/25.31/20.89/26.15 minutes. Lower memory costs more time on that workload. A fresh design should offer a declared memory budget and explicit policy selection, with admission estimates clearly distinguished from an actual process-memory guarantee.

Disk-backed storage remains conditional; the current results do not justify building it first. [Phase 2M](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/PLAN.md).

## Change 4: make the quantum experiment about the final estimand

For a fixed Ritz coefficient vector c,
`E(c) = c† H_sub c / (c† S c)`.
Near a simple, well-conditioned eigenpair, its first-order sensitivity is
`delta E = c†(delta H_sub - E delta S)c / (c† S c)`.
This motivates measurement of the information the final output needs, while preserving the cost of learning c.

The project already has WordFunctional, SharedMeasurement, ritz_functional, coupling_functional, variance_optimal_group_plan, compiled samplers, and sample-split certified selection. Reuse them. A fixed-energy clique setting is a different capability from reusable per-word estimates for many H/S entries.

### Compile final functionals without reconstructing packed rows

When wiring ritz_functional and coupling_functional, consume `bank.iter_operator_terms(...)` rather than requesting `overlap_operator(...).terms` or `element_operator(...).terms`. The target is to avoid reconstructing every packed row as an MV dictionary. The aggregate functional may still allocate its own coefficient table, whose support and bytes must be priced.

A packed-bank regression should forbid row-materializing accessors while building the functional, then compare its coefficients and evaluations with the object-bank path under the declared arithmetic contract. Verify these seams on the implementation head before editing; this revision does not claim that the change has shipped.

### Stage 1: learn and freeze from the pilot

Spend a declared N_pilot to select the span and learn c_hat, E_pilot, rank/regularization choices, and pilot group-variance estimates. Charge all pilot and selection shots. Freeze the evaluation directions, settings, allocation, and failure rules before obtaining held-out outcomes. Exact coefficients or exact variances belong only to separately labeled oracle controls.

### Stage 2: evaluate the correlated ratio on independent data

For fixed c_hat, define `f_H = c_hat† H_sub c_hat` and `f_S = c_hat† S c_hat`. The held-out plug-in estimate is `E_hat = f_H_hat / f_S_hat`. Preserve their covariance by estimating both from the same held-out joint group histograms.

Only an algebraically known overlap and normalization permits treating f_S as deterministic. Distinct normalized determinant directions on a determinant reference can supply such a case; dressed or warm-start directions require their own proof or measured overlap. Pilot normalization does not make the held-out denominator deterministic.

Use the frozen linearized ratio functional `f_H - E_pilot f_S` to guide variance allocation. For equal per-shot group costs, Neyman allocation is `n_g proportional to sqrt(V_g)`, using pilot estimates; differing costs require a declared cost-weighted rule. Account consistently for denominator normalization, minimum group coverage, integer rounding, and all shots used to estimate variances.

Evaluate both f_H_hat and f_S_hat on the evaluation batch. Require the denominator's valid lower confidence bound to exceed a declared positive floor; otherwise report an unresolved/failed ratio and include it in unconditional success denominators. The linearized variance is a diagnostic or asymptotic approximation unless a finite-sample argument is supplied. Any ratio confidence interval must use an explicitly justified joint-coverage construction and preserve numerator/denominator dependence.

A two-stage independent pilot/evaluation split is the first protocol. Cross-fitting is an optional separately specified extension: each fold's c_hat and allocation must be learned without that fold's evaluation outcomes, and fold aggregation and coverage need their own contract. These are distinct designs.

Compare the output-focused estimator with complete-bank reconstruction on the same problem/reference and total shot budget. Charge preparation/readout, pilot/selection, compilation, final functional storage, failed solves, denominator failures, and inference time. Report total ground-energy error separately from estimation error around the retained-space energy.

Centering, sample splitting, cross-fitting, and regularization do not automatically remove finite-sample ratio bias, produce an energy certificate, or restore variationality. A Gaussian matrix-noise illustration is not a Pauli-shot model.

Avoid starting with measured H-squared certification: the current Phase 15 studies found unfavorable scope-specific costs. A new estimator requires its own feasibility screen and declaration.

Noisy generalized eigenproblems and thresholding already have a substantial numerical-analysis literature: [Epperly, Lin and Nakatsukasa](https://arxiv.org/abs/2110.07492). [Existing session implementation](https://github.com/gutama/clifford_qc/blob/89d3cffd25aa86b2d155a74291486800f17b31ce/clifford_qc/measurement/session.py).

## Change 5: separate a research archive from the active product

Keep one repository initially; separate responsibilities before creating multiple packages or services.

A compact structure would have operator/model contracts, exact projection providers, measurement providers, inference, a supported workflow layer, and a research archive. Keep optional bridges outside the default import surface. Preserve legacy imports through a versioned compatibility layer during extraction.

Replace the active 6,201-line narrative with a short current roadmap and links to immutable studies. Retain PHASE_STATUS as historical implementation status. Add a separate current capability table answering: implemented? validated on what? measurable without an oracle? demonstrated advantage on what scope?

The current corpus already contains independent checkers, archival sources, digests, and content-addressed payloads. Consolidate their common mechanics into a shared experiment runner and artifact contract. Do not weaken historical declarations, ancestry requirements, or negative outcomes.

Complete the already-listed migration away from live-source hash guards and bespoke AST exceptions. Frozen evidence binds its execution snapshots. Current implementations should be validated by their own numerical and behavioral tests, with an explicit migration whenever a historical contract changes.

Use fast invariant checks for ordinary core changes, relevant campaign checks when affected, and a periodic complete archival replay. Changes to evidence, declarations, or archived sources must still trigger their full required checks.

For future coordinate optimizations, declare whether equivalence means physical rays/projectors, spectra/residuals, selected keys, shortlist membership, or exact internal ordering. Preserve the frame-coordinate study's failed historical criterion; do not rewrite it after observation.

## First rebuild sequence and stop rules

| Order | Deliverable | Exit condition |
|---|---|---|
| 1 | Freeze a v1 evidence snapshot and write a short v2 problem/claim contract | One workload, one target output, one baseline family, complete cost boundary, and a falsifiable benefit are specified. |
| 2 | First PR: add reference-evaluation and exact projected-provider seams | Working DeterminantReference/MVReference adapters and determinant molecular integration pass explicit bitwise/tolerance contracts; a large-qubit test forbids density expansion; sector/sampled capabilities are defined for later consumers. |
| 3 | Add the sector consumer, bounded classical solving, and root recovery | Cached-action convergence reporting, spin checks, and block/multistart recovery cover the primary Hubbard excited-root failure; unresolved roots remain explicit; owned-memory admission and monitored RSS are distinguished; all attempts are priced. |
| 4 | Reproduce a small old suite through the new supported path | Same Hamiltonians, sectors, and variational spans give compatible results; storage/admission behavior is recorded. |
| 5 | Add the sampled consumer, then declare and run one measurement experiment | Compare the two-stage ratio estimator with complete-bank reconstruction; pilot and evaluation are independent, packed rows are not reconstructed for functionals, and pilot costs, covariance, denominator failures, and unconditional target attainment are reported on development and held-out inputs. |
| 6 | Promote only supported benefits | A frozen improvement target is met at common accuracy and resource limits, or the hypothesis closes without another family of variants. |

A possible new promotion rule is at least 20% lower end-to-end cost at the same target accuracy and declared success probability, with no breach of the memory cap, on the primary and predeclared held-out inputs. This is a suggested decision rule, not an observed result or a universal statistical standard. Repeated timings on one deterministic problem measure timing variation; they do not create additional independent physical test cases.

For a pure classical-solver objective, prioritize steps 1–4 and external FCI/SHCI/DMRG comparisons where appropriate. For quantum-method research, steps 2–5 are essential. For a software-toolkit objective, success can be useful interfaces and reproducibility without a new algorithmic advantage. Keep these publication claims distinct.

## First PR: determinant reference and exact projection

Implement Steps 1–2 in this order, keeping one supported determinant molecular workflow:

1. Write the short v2 problem/claim contract and link the unchanged historical evidence snapshot.
2. Add ReferenceState metadata/capability contracts, with working DeterminantReference and MVReference adapters.
3. Add the exact ProjectedProblemProvider seam, separating numerical evaluation from measurement-row compilation.
4. Integrate analytic determinant evaluation into the existing molecular workflow while retaining a clearly identified legacy arithmetic path.
5. Validate the implementation and publish a compatibility matrix before moving other workflows.

### First-PR acceptance matrix

| Check | Required result |
|---|---|
| Analytic determinant expectations | Match independent bit-pattern calculations for the declared ordering and Pauli phases. |
| Legacy S/H comparisons | Explicit bitwise tests exercise both trace-pairing iteration branches, operator emission orders, complex cancellation, scaling, and signed zeros. Any non-bitwise case has an explained and declared tolerance contract; required legacy fixtures retain their compatible path. |
| Packed/object consistency | Generation permutations, row iteration, coefficients, and final scalar evaluations satisfy the stated compatibility contract. |
| No density expansion | A bounded-row, large-qubit test forbids ket_density/ExactMVBackend construction and dense Hilbert-space allocation on the determinant path. |
| Identity and capability safety | Reference/Hamiltonian identities prevent stale caches; exact calls on sampled-only capabilities are rejected. |
| Molecular integration | Existing small determinant fixtures reproduce S/H, variational energies, and solver decisions to the declared scope; changed arithmetic is disclosed. |
| Resource accounting | Occupation-state bytes, row/index bookkeeping, allocations, and evaluation work are distinguished. No unmeasured RSS or speedup claim is made. |

Define the sector and sampled boundaries now; implement their full adapters with their Step 3 and Step 5 consumers. Step 3 adds root recovery, action-based residual reporting, and bounded buffers. Step 5 adds the independent pilot/evaluation runner and sampled integration. They are subsequent PRs.

Preserve the existing algebra and evidence, select one supported workflow, and let the new costed experiment decide whether operator-response corrections earn a central role.

## Review checks

- Inspected pinned README, PLAN, ARCHITECTURE, phase ledger, source census, recent PR records, central backend/projection/measurement/solver modules, and current molecular/packet reports.
- Read the 20 October research review; several of its sparse/cache suggestions have since shipped or been scientifically closed.
- Independently checked a random 12-dimensional Hermitian example with four basis vectors and an invertible coordinate transformation. Generalized Ritz spectra agree to 2.22e-15; this illustrates the exact same-span invariance, not a benchmark improvement.
- Computed sector-memory sizes from D = binomial(L,L/2)^2 and 2 D M times 16 bytes.
- Consulted official PySCF, SciPy, PRIMME and Dice material and the primary QSD theory paper.
- No new performance campaign, hardware execution, or full repository test run was performed for this review.
- The implementation refinement incorporates the follow-up discussion: summation-order qualification, separate owned/RSS memory contracts, spin versus root evidence, packed-row iteration, the correlated Rayleigh quotient, and the narrowed first-PR scope. It updates the plan; code-seam changes and their acceptance tests remain to be implemented.


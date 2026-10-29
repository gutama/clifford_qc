# Cross-domain sampled ratio validation

This is a separate, prospectively declared extension of restart Step 5. The
[spin-only experiment](SAMPLED_RATIO.md), its 2,304 observations and its failed
20% cost criterion remain unchanged. The [declaration](cross_domain_ratio_declaration.json)
is committed before any molecular measurement outcomes. This extension tests
cross-domain integration; it does not turn an estimation result into an
eigensolver, chemistry or hardware advantage.

## Fixed inputs and physical targets

| Input | Role | Register and reference | Estimation target |
| --- | --- | --- | --- |
| TFIM, J=1, h=1.2 | Development control | Four qubits; fixed product reference from the original protocol | 0.05 t |
| H4 STO-3G, r=0.9 Angstrom | Held-out molecular validation | Eight qubits; N=4, Sz=0, S=0 | 0.0016 Ha (1.6 mHa) |
| LiH CAS(4e,4o), r=1.5949 Angstrom | Held-out molecular validation | Eight qubits; N=4, Sz=0, S=0 | 0.0016 Ha (1.6 mHa) |

The molecular FCIDUMP bytes and input identities are bound in the declaration.
These are known historical Hamiltonians held out from tuning this protocol;
they are not blind external data. Rank floors, correction operators, budgets,
seeds, failure rules and targets are fixed before drawing outcomes. Molecular
chemical units and TFIM hopping units are separate.

The molecular preparation is the unoptimized paired singlet
`cos(0.3)|11110000> + exp(0.2i) sin(0.3)|11001100>`, with q0 leftmost.
Use the explicit circuit and feasible preparation schedule in the declaration:
six one-qubit gates and three CNOTs, with three one-qubit and three two-qubit
layers. Preparation is priced as well as the measurement basis changes. This
is a known small simulator state, not a scalable preparation or hardware claim.

The fixed molecular generators are `I`, `(I+G12)/sqrt(2)` and
`(I+G13)/sqrt(2)`. Spatial indices are zero-based:
`T_s_to_t = a†_(2t) a†_(2t+1) a_(2s+1) a_(2s)` and `G=T-T†` is
anti-Hermitian. Pair transfers preserve N, Sz and singlet multiplicity.
The retained directions have a measured, generally nonidentity Gram matrix.
The TFIM generators remain `I, H/n, Z0`. Generator choice follows structural
requirements; no variant is selected after molecular outcomes.

## Pilot, evaluation and failure accounting

Each of the three shot budgets, 16,384 / 65,536 / 262,144, is executed for
128 replicas per arm on all three inputs: 2,304 scheduled attempts. A common
quarter-budget pilot learns the overlap rank, retained frame, coefficients and
variance allocations. Both arms evaluate the same frozen pilot direction on
fresh independent joint Pauli-shot batches. No exact coefficients, energy or
variance enters selection, allocation or stopping.

The output-focused arm allocates from the pilot variance of
`f_H - E_pilot f_S`; the baseline allocates to complete S/H reconstruction.
Each reconstructs H and S from the same evaluation histograms, retaining their
covariance. The finite-sample joint Hoeffding rectangle is transformed into a
ratio interval only if its denominator lower endpoint exceeds the fixed
positive floor. Otherwise the ratio is unresolved. Centering and sample
splitting do not remove finite-sample ratio bias or establish variationality.

All pilot, denominator, nonfinite, numerical and resource failures stay in the
128-attempt denominators. Every shot and all work actually spent is charged,
including failed attempts. Missing checkpoint observations are disclosed as
failures rather than selectively rerun. Uniform/variance allocation details,
minimum group counts and integer remainder rules are fixed in the declaration.

## Independent assessment and resources

The molecular checker builds occupation-sign FCIDUMP H, S-squared and pair
operators independently of the native Pauli product. It assesses the known
reference, frozen coefficient vector, retained-frame minimum and ground energy
in the N=4, Sz=0, singlet sector. The full-Fock minimum is not the molecular
ground-state comparison. Exact assessment occurs after inference, is never
provided to the consumer, and is timed separately. Estimation error around a
frozen direction, retained-space error and physical-sector ground-energy error
are reported separately.

Each input runs in a fresh process with a 1,800-second timeout. A timed-out
worker preserves its completed checkpoints; unobserved scheduled attempts remain
failures and are never selectively rerun. The fixed owned-array cap is 512 MiB; the
conservative eight-qubit array/workspace admission model is 378,535,936 bytes
(361 MiB). A separate process high-water RSS cap is 1 GiB. These are new,
prospectively declared limits, not changes to the earlier experiment's 64 MiB
and 512 MiB limits. Python/allocator overhead is outside the array model and
inside process RSS. Record validation checks the observed resource bounds
without applying the campaign's execution-time limit to a shared pytest
process.

End-to-end cost combines an illustrative serial device scenario (initialization,
explicit reference preparation, readout circuits and reset/readout) with measured
model/provider setup, pilot inference/allocation and evaluation inference.
Ideal simulator draw time and independent assessment time are separate. No
hardware calibration, noise mitigation or quantum advantage is implied.

## Frozen decision and chronology

A cell qualifies only if its simultaneous one-sided exact-binomial success
lower bound is at least 0.9, at the input's declared absolute estimation target,
with no resource failure. Confidence accounts for all 18 input/budget/arm cells.
For each arm/input choose the cheapest qualifying declared budget. Promotion
requires at least 20% lower median end-to-end cost on every input. If a baseline
never qualifies, a matched-success saving is unavailable, not imputed. Failure
closes this particular cross-domain hypothesis without changing budgets,
cutoffs, input definitions or the earlier spin-only conclusion.

The execution binds committed declaration, all scientific sources, FCIDUMP
inputs, historical evidence and the Python/NumPy/SciPy stack. An exclusive
common-Git-directory claim prevents a second execution of the same campaign.
Case checkpoints and the final result are written atomically; results are
published as deterministic gzip-compressed JSON (mtime=0) in a separate commit
after the source/declaration commit. Full seed
replay is explicit, requires the recorded stack and never overwrites evidence.
The independent short checker validates chronology, accounting, units,
physical-sector assessment, summaries and the verdict without drawing outcomes.

## Results and decision

All 2,304 scheduled outcomes are retained. The frozen decision is `closed_on_declared_scope`.
Success means a resolved denominator and a point-estimation error within the input-specific target; the probability lower bounds are simultaneous across all 18 cells.

| Input | Shots | Ratio successes / 128 (lower bound) | Bank successes / 128 (lower bound) | Ratio / bank unresolved denominators |
| --- | ---: | ---: | ---: | ---: |
| TFIM control | 16,384 | 125 (0.911) | 122 (0.876) | 0 / 0 |
| TFIM control | 65,536 | 128 (0.955) | 128 (0.955) | 0 / 0 |
| TFIM control | 262,144 | 128 (0.955) | 128 (0.955) | 0 / 0 |
| H4 held out | 16,384 | 3 (0.002) | 5 (0.007) | 0 / 0 |
| H4 held out | 65,536 | 14 (0.047) | 12 (0.037) | 0 / 0 |
| H4 held out | 262,144 | 31 (0.146) | 29 (0.134) | 0 / 0 |
| LiH held out | 16,384 | 9 (0.023) | 8 (0.019) | 0 / 0 |
| LiH held out | 65,536 | 14 (0.047) | 7 (0.015) | 0 / 0 |
| LiH held out | 262,144 | 40 (0.205) | 30 (0.140) | 0 / 0 |

| Input | Cheapest qualifying ratio budget | Cheapest qualifying bank budget | Cost reduction |
| --- | ---: | ---: | ---: |
| TFIM control | 16384 | 65536 | 73.0% |
| H4 held out | None | None | Unavailable |
| LiH held out | None | None | Unavailable |

The all-input 20% criterion failed. TFIM qualifies at 16,384 ratio shots versus
65,536 bank shots, giving a 73.0% saving in the illustrative scenario. Neither
molecular arm qualifies at any declared budget. At 262,144 shots the ratio arm
meets 1.6 mHa in only 31/128 H4 and 40/128 LiH attempts. All attempts resolve
the denominator; this failure concerns the estimation-accuracy target. An
unavailable matched-success comparison supplies no evidence of a saving. No
budget, target, cutoff or case has been changed after sampling.

Errors below are conditional medians among resolved attempts at the largest declared budget. They describe three different targets and do not certify the physical ground root. Molecular values are in Ha; the TFIM control is in t.

| Input | Arm | Frozen-direction estimation error | Retained-frame error | Physical-sector ground-energy error |
| --- | --- | ---: | ---: | ---: |
| TFIM control | ratio | 0.00377272 | 0.00374202 | 0.234856 |
| TFIM control | bank | 0.00339973 | 0.00339534 | 0.235769 |
| H4 held out | ratio | 0.00375554 | 0.00427553 | 0.0354816 |
| H4 held out | bank | 0.00368131 | 0.00408308 | 0.034835 |
| LiH held out | ratio | 0.00259597 | 0.0105038 | 0.0106782 |
| LiH held out | bank | 0.00426231 | 0.0126359 | 0.0128103 |

| Input | Observed / scheduled outcomes | Peak worker RSS (MiB) | Resource / worker failures |
| --- | ---: | ---: | ---: |
| TFIM control | 768 / 768 | 96.1 | 0 / 0 |
| H4 held out | 768 / 768 | 225.9 | 0 / 0 |
| LiH held out | 768 / 768 | 210.4 | 0 / 0 |

Declaration and execution source: `84ba20719028cf9b11e29984dc526c959d04566a`. The result is published in a later commit; the checker verifies that ancestry and all source/input/history digests.

Validation uses an independent occupation/sign molecular Hamiltonian and singlet projection, joint-moment covariance and ratio-interval checks, and unconditional failure denominators. Full seed replay compares scientific observations while leaving wall-time and RSS observations environment dependent.

This completes the declared spin-plus-molecular integration check. It supports no hardware, eigensolver or general quantum advantage claim. The original spin-only negative result and frozen blueprint remain unchanged.

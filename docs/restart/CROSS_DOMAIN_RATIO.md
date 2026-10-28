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

## Status

Declaration and implementation are being validated. No cross-domain outcome
has been drawn or benefit promoted at this stage. The final result, including
all failures and a separately derived decision, will be recorded here after
the single authorized execution.

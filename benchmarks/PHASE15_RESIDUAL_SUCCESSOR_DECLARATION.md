# Q18-S1: post-hoc successor to the measured-residual cost preflight

## Status and disclosure

This is a separate declaration after Q18's `INVALID` record, its ratios, its
Neyman diagnostics and the Hubbard six-step sweep were inspected. It is
**post-hoc on all five banks**. Declaring the config before its own execution
protects identity and reproducibility; it does not make the inference
preregistered. Q18's original declaration and record remain byte-for-byte intact.

The successor asks the original uniform-cost question. It does not choose a new
threshold or allocation to make an old result pass. Its config is
`configs/phase15_residual_successor.json`; its gate is
`check_phase15_residual_successor_declaration.py`.

## Question and unchanged controls

Keep the five licensed banks, their selected bases, estimator, two derivative
directions, Gaussian seed, QWC partitions, single assignment, uniform allocation,
matched-standard-error ratio and tenfold threshold from Q18. The estimator includes
the Ritz-vector response. Any failed deterministic check makes the bank INVALID;
any INVALID bank makes the aggregate INVALID. Otherwise the original FULL,
RESTRICTED and NONE combination rule applies, **under uniform QWC only**.

The declaration binds the predecessor's config and INVALID record, all source
inputs and every implementation used by its producer and checker. Input hashes
remain binding after the run. Implementation bindings lapse once the successor
record exists, which names its own clean source commit. The declaration's last
commit must strictly precede the record's first commit.

## Numerical validator

Evaluate central differences of the complete nonlinear pipeline at all four
steps: `0.002`, `0.001`, `0.0005`, `0.00025`. For each pair, compute

`R(h) = (4 D(h/2) - D(h)) / 3`.

Both final Richardson estimates must agree with the analytic derivative. Their
tolerance retains the original `1e-6 * ||g||` relative term and propagates each
central difference's roundoff floor:

`floor_R = RESOLUTION * cancellation_scale * (4/(h/2) + 1/h) / 3`.

They must also agree with each other within the relative term plus both
propagated floors. The earliest Richardson estimate is diagnostic; there is no
step selection based on a discrepancy. Every endpoint must retain the full block
and a ground Ritz gap greater than the original relative gap floor. This is
numerical validation, not a proof of differentiability or a confidence interval.

Tests precede execution. They compare with a separately assembled dense density
perturbation and SciPy generalized eigensolve on the undeclared Hubbard dimer,
reject an omitted Ritz-vector response, and exercise known cubic truncation,
unstable estimates, invalid domains and forged record fields.

## Separate allocation screen

Per-setting exact variances are retained so the oracle Neyman diagnostic is
reproducible. A second diagnostic converts them to integer production counts:

`n_g = max(1, ceil(sqrt(v_g) * sum sqrt(v) / s^2))`.

Both energy and residual norm use `s = sigma/100`; the residual setting variances
are divided by `4 sigma^2`. Charge **64 additional pilot shots per setting** on
each side. Pilot outcomes are excluded from the production estimate. Zero-variance
settings still receive the positive floor and pilot charge.

**No pilot is performed.** Weights use exact variances, so charging that overhead
does not turn this oracle screen into a realizable pilot protocol. Report every
count, achieved first-order variance and total cost ratio; no decision clause reads
them and no allocation estimator is licensed. A practical study requires a separate
declaration of pilot variance estimation, seeds, budgets, replication and validation
before its sampled results. This screen supplies no bootstrap coverage, nonlinear
estimator bias, device time or quantum-advantage statement.

## Execution and consequence

Land and test the declaration, numerical helpers, producer and independent checker
first. Run once from a clean tree on the pinned record environment. Commit the
record alone, then update the ledger and reproduction notes.

FULL or RESTRICTED can license a subsequent separately validated **uniform-QWC**
implementation only on the named affordable banks. NONE closes that uniform
implementation negative on these five banks. INVALID leaves the uniform question
unanswered. Every outcome is post-hoc. Exact residuals and convergence reports
remain available; the allocation question stays separate under every outcome.

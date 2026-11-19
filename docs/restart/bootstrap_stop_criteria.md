# Fresh-seed check of bootstrap intervals at the stop (fixed before any fresh run)

Written 2026-11-19, after the development screen on the declared seeds and
before any run on the seeds below.

Systems, schedule, targets, delta, solver policy: exactly the declared
target-acquisition study (4 banks, 2 allocations, targets 0.02 and 0.01,
19 looks, delta 0.05, default thresholding). Target value: exact subspace
ground energy. Seed root 2026111901 (the declared root was 20261117).
200 replicas per bank. 500 bootstrap draws per bootstrapped look, seeded from
(root, 2_000_000 + system, replica, 10 * allocation + target, look).

Robust spread r = (q84 - q16) / 2 of the bootstrap draws at a look.
z = 3.008 (two-sided 0.05 over 19 looks, as acquire_ritz reports).

Arm A (primary): the declared delta-method stop, unchanged. Reported interval
E +- z r at that look.

Arm B (secondary, exploratory): stop at the first look where sigma <= target
and r <= target (bootstrap only at looks where sigma <= target). Reported
interval E +- z r. Exhausted runs report at the last look.

Pass for arm A: in every one of the 16 cells, coverage is not resolved below
0.95 (Wilson 95% upper bound >= 0.95), and in every tfim6_mixed cell the
observed coverage is >= 0.95. Widths (median, fraction with half-width > 2 z
target) and, for arm B, mean shots relative to the delta stop and the fraction
exhausting the schedule are reported without a threshold.

The recorded delta interval (E +- z sigma) at the arm-A stop is reported beside
both, as the comparison.

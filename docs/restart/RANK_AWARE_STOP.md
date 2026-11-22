# Rank-aware precision stopping (opt-in)

The [stretched rank stress test](STRETCHED_MOLECULAR_STOP.md) showed that a
precise Ritz value in a truncated space can be grossly wrong for the declared
full space. The existing estimated-SE-only stop remains the default.
The opt-in `RankStopPolicy` guards against that failure mode, and the separate
bootstrap-width guard can be used alone or together with it.

```python
from clifford_qc.subspace.acquisition import RankStopPolicy, acquire_ritz

result = acquire_ritz(shared, backend,
    pilot_shots_per_group=250, round_budgets=round_budgets,
    target_std_error=0.0016, solver_kwargs=solver_kwargs,
    control_variates=family, rank_stop=RankStopPolicy(consecutive_looks=2),
    bootstrap_replicates=64, max_bootstrap_half_width=0.0016)
if result.stopped == "target_met":
    print(result.energy, result.bootstrap_interval)
else:
    # An unresolved guard or exhausted budget is not precision attainment.
    print(result.stopped, result.looks[-1])
```

The full declared bank must be independent: repeated agreement on a smaller
rank does not qualify. The controller requires its full rank and positive
lower cutoff margins at two consecutive looks. For measured overlap S,
diagonal normalization D and normalized extremal eigenvectors u, use
the delta derivative of `u† D^-1/2 S D^-1/2 u`, **including** the derivative
of D. The three margins compare the smallest normalized eigenvalue with
absolute tau, relative tau times the largest eigenvalue and the largest
eigenvalue divided by the condition cap. Relative margins differentiate the
largest mode too, retaining covariance. Use covariance-aware Gaussian radii
with a finite-look Bonferroni family of three comparisons per look. The
weakest lower margin determines targeted overlap acquisition: half of the
round (rounded up) goes toward the margin while SE is above target, all of
the round if SE qualifies, and the remainder targets Ritz variance. The
uniform fallback preserves the shot budget when either functional is
deterministic or the overlap frame is unresolved. If the margin is resolved,
ordinary Ritz targeting continues until confirmation and SE qualify.

`rank_stop` refuses shot-calibrated cutoffs and requires a declared SE target
and enough scheduled looks. The guards do not modify the projected solver or
allocate using exact spectra. When the bootstrap half-width guard is enabled,
it re-solves only SE/rank-eligible looks, uses a reproducible distinct seed at
each look, and reuses the last assessment instead of drawing it twice. Failed
bootstrap draws cannot authorize a stop. The final bootstrap at an ineligible
terminal look is diagnostic only. No bootstrap is needed for rank-only
stopping; optional terminal bootstrap behavior is otherwise preserved.

The statistical separation guard is heuristic: the derivative uses
data-derived eigenvectors, may be inaccurate at degeneracy, and the grouped
histograms are adaptively collected. It adds a necessary diagnostic, not a
finite-sample, variational or anytime-valid certificate. All intervals retain
their previous evidence labels. It tests the declared projected energy, not
full molecular chemical accuracy. A deliberately near-cutoff independent
mode may require substantial extra shots or remain unresolved at exhaustion.

## Preregistered validation

The [result-free declaration](rank_aware_stop_declaration.json) binds the
previous stressed LiH CAS(2e,5o) and BeH2 CAS(4e,6o) banks and result bytes.
Use the same full four-direction projected energy estimands, ideal joint
QWC shots, diagonal symmetry-corrected controller, five-look schedule and
1.6 mHa SE threshold. The new seed root is independent of prior studies.
The normalization derivative is checked against finite differences and
small-bank regression tests before the evaluation. The development
namespace (501) never enters the fresh sample.

Four policies are compared with 16 independent replicas per molecule/policy
(128 acquisitions), matching the pilot RNG stream across policies:

| Policy | Stop condition | Allocation |
| --- | --- | --- |
| SE only | SE <= 1.6 mHa | Corrected Ritz targeting |
| Rank/margin | SE and two consecutive separated full ranks | Overlap-margin/Ritz split while unresolved |
| Bootstrap width | SE and adjusted bootstrap half-width <= 1.6 mHa | Corrected Ritz targeting |
| Fixed budget | No early stop; spend the declared schedule | Corrected Ritz targeting |

All use 16 terminal bootstrap resamples; width checks cost an additional
16 draws at each eligible nonterminal look. Count every attempt, cost and
solve; distinguish numerical failures, exhaustion, true precision stops,
false precision stops and diagnostic terminal accuracy. A true stop meets
all active guards and has actual absolute projected error <= 1.6 mHa;
failure or exhaustion is never a true stop. The fixed-budget arm is not
an adaptive success even if its terminal error is small. Report interval
coverage and widths at the actual interval family (five looks except fixed
budget, one), and paired shot costs. Comparison is descriptive.

Rank/margin feasibility requires each molecule to have at least 12/16 true
precision stops, at most one false stop, and fewer false stops than SE-only.
This is a preregistered descriptive gate, not a statistical superiority test;
it avoids falsely declaring success by abstaining on nearly every attempt.
The bootstrap-width arm is a distinct comparator, not tuned after seeing
the evaluation. A combined policy is tested mechanically but not evaluated
as a fifth scientific arm. Limited replica count and engineered rank
instability prevent broader validation claims.

## Reproduction

Run at the published result-free declaration commit under the pinned stack.
Use an empty checkpoint folder outside the checkout; resume with `--resume`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/check_rank_aware_stop.py --declaration-only --full-anchor
python benchmarks/run_rank_aware_stop.py --execute --checkpoint-dir /tmp/guarded-stop --jobs 8
python benchmarks/check_rank_aware_stop.py --report
python benchmarks/check_rank_aware_stop.py --spot-replay
python benchmarks/check_rank_aware_stop.py --verify-history
```

## Results

The result-free declaration was published at
`1661bfefec6d95621deaa96669801ef93e7e0d2d` before evaluation. The
[compressed record](rank_aware_stop_results.json.gz) contains every one of
128 acquisitions, 491 measured solves, 2,688 bootstrap solves and
3,702,893,500 simulated physical shots. No acquisition or bootstrap failed.
The independent parent occupation-space matrices and native frozen-core
anchors agree, reproducing the previous exact full-rank projected energies
(-7.763632943522596 Ha LiH, -15.200464376073494 Ha BeH2). All 96 adaptive
attempts met their respective guards; all 32 fixed-budget attempts reached
the schedule horizon. The fixed-budget terminal points are not credited as
adaptive stops.

The independent checker verified the full 128-attempt ledgers, the
paired pilot fingerprints, all 3,179 recorded solve rank decisions, margin
arithmetic and bootstrap seeds. All eight designated spot replays (replica
zero of each molecule/policy) passed. Focused regression and CI-gate tests:
290 passed, one older version-pinned replay skipped. Ruff, mypy, manuscript
and reproduction-document checks passed. Routine CI performs only the short
history, occupation-space anchor and ledger verification.

### Precision stopping and cost

| Molecule | Policy | True precision stops | False precision stops | Reduced-rank stops | Terminal error <= 1.6 mHa | Mean shots (M) | RMSE (mHa) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LiH | SE only | 12/16 | 4/16 | 2 | 12/16 | 2.474 | 121.578 |
| LiH | Rank/margin | 15/16 | 1/16 | 0 | 15/16 | 13.347 | 0.670 |
| LiH | Bootstrap width | 16/16 | 0/16 | 0 | 16/16 | 18.870 | 0.339 |
| LiH | Fixed budget | — | — | — | 16/16 | 58.912 | 0.134 |
| BeH2 | SE only | 13/16 | 3/16 | 2 | 13/16 | 5.640 | 95.069 |
| BeH2 | Rank/margin | 16/16 | 0/16 | 0 | 16/16 | 11.985 | 0.833 |
| BeH2 | Bootstrap width | 16/16 | 0/16 | 0 | 16/16 | 29.963 | 0.533 |
| BeH2 | Fixed budget | — | — | — | 16/16 | 90.240 | 0.193 |

Both rank/margin cells pass the frozen descriptive feasibility criteria:
LiH true stops 15/16 with one false stop versus four SE-only false stops;
BeH2 true stops 16/16 with zero false versus three SE-only false stops.
Across the two banks the guard reduces false stops from 7/32 to 1/32 and
eliminates reduced-rank stops (4 to 0). The remaining LiH false stop has
full confirmed rank but actual error 2.212 mHa. Thus rank stability addresses
the catastrophic truncation mechanism without turning an estimated SE into
an absolute error bound. With only 16 independent replicas per cell, even
16/16 true precision stops has Wilson 95% interval 80.6–100%; 15/16 has
71.7–98.9%. This is descriptive evidence, not proof of universal reliability.

Rank-aware acquisition costs 5.40x (LiH) and 2.13x (BeH2) the SE-only mean
shots, but reduces point RMSE by factors 181.5 and 114.2 respectively.
Compared with the full fixed budget it uses 77.3% and 86.7% fewer shots;
compared with width-only it uses 29.3% and 60.0% fewer shots, respectively.
The rank-aware rounds allocate a total 27,154,750 and 18,330,000 shots
specifically to the overlap margin across the two cells (included in shot
cost, no extra shots on top of each declared round). This avoids continuing
to buy only energy-sensitive measurements in a truncated space.

### Interval and classical computation trade-offs

| Molecule | Policy | 95% delta coverage | 95% bootstrap coverage | Median bootstrap full width (mHa) | Measured solves | Bootstrap solves |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LiH | SE only | 13/16 | 15/16 | 16.391 | 41 | 256 |
| LiH | Rank/margin | 15/16 | 16/16 | 2.377 | 59 | 256 |
| LiH | Bootstrap width | 15/16 | 16/16 | 1.383 | 64 | 624 |
| LiH | Fixed budget | 15/16 | 15/16 | 0.526 | 80 | 256 |
| BeH2 | SE only | 14/16 | 14/16 | 7.287 | 48 | 256 |
| BeH2 | Rank/margin | 16/16 | 16/16 | 3.697 | 54 | 256 |
| BeH2 | Bootstrap width | 16/16 | 16/16 | 2.157 | 65 | 528 |
| BeH2 | Fixed budget | 15/16 | 16/16 | 0.729 | 80 | 256 |

All intervals were available. Report actual schedule-adjusted 95% intervals
for adaptive policies and single-look 95% intervals for the fixed horizon.
The width-only arm attained 32/32 true precision stops, with zero false
stops, at greater mean shots than rank/margin and 1,152 bootstrap solves
versus 512 terminal-only rank/margin solves. On these cases rank resolution
is the cheaper first guard; width is useful when a narrow reported interval
is essential, not a replacement for checking structural rank. In particular,
a finite bootstrap can miss a rank regime and cannot certify it absent.

### Scientific decision

Adopt the two optional policy controls as research features, not defaults
or certificates. The rank-aware policy passes this frozen near-cutoff
stress validation and materially reduces catastrophic errors without full
budget expenditure. Width-only attains every point error target in this
small sample at higher cost; the combined rank-and-width API is tested
mechanically, but has not been subjected to a distinct fresh-seed campaign.
The next test should use naturally ill-conditioned, larger banks and noisy
hardware models, with more outer and inner draws and a declared combined
policy. If the full bank is structurally dependent, repair/reparameterize
it before using the full-rank guard; a reduced-rank stopping policy would
require a separately justified estimand and criterion.

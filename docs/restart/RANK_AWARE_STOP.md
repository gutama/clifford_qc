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
```

## Results

Pending fresh evaluation. No evaluation outcomes exist in this declaration commit.

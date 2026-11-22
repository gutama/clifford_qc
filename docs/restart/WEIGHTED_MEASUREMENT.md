# Variance-first, Ritz-weighted, and noise-gated measurement planning

`clifford_qc.measurement.weighted_planning` adds a variance-first planning layer
on top of the NumPy-only `graph_state` readout compiler and the block-commuting
($k$-commutativity) hierarchy. It does not change any frozen historical records,
elimination circuits, or prior negative campaign conclusions.

## Preferred entry point and backward compatibility

Use `plan_measurement` for new measurement planning workflows:

```python
from clifford_qc.measurement import (
    DeviceCard,
    NoiseGatePolicy,
    evaluate_plan_variance,
    plan_measurement,
)

plan = plan_measurement(n, codes, block_size, weights=hamiltonian)
assert plan.method == "graph_state"
```

- **Default readout compiler:** `plan_measurement` defaults to
  `method="graph_state"`, which is pure NumPy and requires no Stim installation.
- **Frozen historical replay:** `compile_block_measurement_plan` and
  `synthesize_block_settings` retain their default `method="elimination"` so
  byte-hash-pinned producers and historical elimination records replay
  unchanged.
- **Default sampled provider stability:** `SampledProjectedProvider` keeps its
  default `grouping="largest_degree"` (with `method="graph_state"`) so frozen
  `circuit_identity` values and historical Step 5/cross-domain records are
  unchanged; weighted grouping and noise gating are opt-in.

## Weighted $k$-commuting grouping and Ritz sensitivity

Unweighted largest-conflict-degree (`"largest_degree"`) minimizes setting counts
without regard to observable weights, which can split large-magnitude Pauli
terms across separate settings and inflate shot variance.

- `word_weights(target, *, aggregation="l2", weights=None)` extracts nonnegative
  per-word weights (excluding the deterministic identity code `0`) from a
  `WordFunctional`, `MV`, `PauliSum`, `{code: weight}` mapping, or a sequence of
  targets aggregated by `"l2"`, `"l1"`, or `"max"`.
- `sorted_insertion_partition(n, codes, block_size, weights=None, *, noise_gate=None)`
  inserts words into the first compatible $k$-commuting setting in descending
  weight magnitude (breaking ties by descending conflict degree, then ascending
  Pauli code). When `weights=None` and `noise_gate=None`, it matches
  `block_commuting_partition`.
- `ritz_sensitivity_weights(bank, indices, ritz_vector, energy, *, overlap_functional=None, overlap_weight=0.0, aggregation="l2")`
  derives first-order Ritz sensitivity weights $|g_w|$ from
  $c^\dagger (H - E S) c$ (optionally blended with an overlap-mode functional
  such as the weakest-mode derivative from `rank_margin`). It reads only the
  operator bank's algebraic coefficients and a caller-supplied
  `(ritz_vector, energy)` (for instance, from an independent pilot); it touches
  no exact reference state or density matrix.

## GALIC-style per-group noise gating

`NoiseGatePolicy` enforces a per-setting two-qubit gate acceptance rule during
grouping. Under independent two-qubit error rate $p$ (`two_qubit_error`) and
relative bias target $\varepsilon$ (`max_relative_bias`), a setting with routed
two-qubit gate count $N_{2\mathrm{q}}$ satisfies $(1 - p)^{N_{2\mathrm{q}}} \ge 1 - \varepsilon$
if and only if:

$$N_{2\mathrm{q}} \le \left\lfloor \frac{\log(1 - \varepsilon)}{\log(1 - p)} \right\rfloor.$$

- `NoiseGatePolicy.from_device_card(card, *, max_relative_bias=0.05, enforce_fidelity_floor=False)`
  reads `eps_2q` and `routing_2q_multiplier` from a declared `DeviceCard` and
  computes both `max_routed_two_qubit_gates` and the exact rational-arithmetic
  `max_logical_two_qubit_gates`.
- During `sorted_insertion_partition(..., noise_gate=gate)`, a candidate merge
  into an existing setting is admitted only if the merged `graph_state` circuit
  satisfies `gate.admits(setting_resources, n)`. When `max_routed_two_qubit_gates == 0`,
  the planner automatically falls back to local QWC ($N_{2\mathrm{q}} = 0$).
- Setting `enforce_fidelity_floor=True` additionally requires every merged
  setting to satisfy `setting_fidelity(card, resources, n) >= card.fidelity_floor`.
  When `noise_gate` is passed to `plan_measurement`, every compiled multi-word
  setting is re-checked against the policy after final compilation.

## Variance-first metrics and rung comparison

`evaluate_plan_variance(plan, target, *, cache=None, reference=None, strategy="custom", noise_gate=None, aggregation="l2", weights=None)`
returns a `GroupingMetrics` dataclass reporting:

- Crawford's state-independent optimal shot ratio under disjoint assignment,
  $$\hat{R} = \left(\frac{\sum_w |a_w|}{\sum_g \sqrt{\sum_{w \in g} a_w^2}}\right)^2,$$
  together with `ungrouped_variance` and `grouped_variance`;
- Optional covariance-aware state-dependent optimal variance and shot ratio $R$
  (`state_dependent_variance`, `state_dependent_ungrouped_variance`,
  `state_dependent_r`, `state_dependent_source`), computed either from a
  measured `GroupedWordCache` (supporting both `pooling="assigned"` and
  `pooling="shots"`) or offline from an exact `MV` density `reference`;
- Two-qubit gate totals, means, maxima, and scheduled depths
  (`two_qubit_gates_total`, `two_qubit_gates_mean`, `two_qubit_gates_max`,
  `two_qubit_depth_mean`, `two_qubit_depth_max`), compatibility density, and
  optional `noise_gate_admissible_settings`.

`compare_block_rungs(n, codes, target, *, block_sizes=None, strategy="sorted_insertion", method="graph_state", noise_gate=None, reference=None)`
evaluates the dyadic $k$-commutativity ladder ($k = 1, 2, 4, \dots, n$) in one
call so variance and entangling-gate cost are inspected side by side.

## Density-free integration with `SampledProjectedProvider`

`SampledProjectedProvider` accepts opt-in `grouping="sorted_insertion"`,
`grouping_weights=...`, and `noise_gate=...`, and adds two methods:

- `provider.plan_metrics(target=None, *, cache=None)` reports `GroupingMetrics`
  against the provider's matrix functionals (or a custom target such as
  `ritz_functional(...)`).
- `provider.with_ritz_plan(ritz_vector, energy, *, block_size=None, overlap_functional=None, overlap_weight=0.0, noise_gate=..., method=None)`
  returns a fresh `SampledProjectedProvider` replanned by first-order Ritz
  sensitivity (plus optional overlap-mode sensitivity) without mutating `self`
  or touching any exact/density oracle. The replanned provider plugs directly
  into `acquire_precision` and `run_workflow(..., path="sampled_precision", ...)`.

To preserve unbiased evaluation when replanning from pilot data, treat the
pilot batch used to estimate `(ritz_vector, energy)` as separate from fresh
evaluation shots drawn under the replanned settings.

## Reproducible illustrative diagnostics

Running the NumPy-only example script reproduces three finite diagnostics from
committed inputs:

```bash
python examples/measurement_planning_upgrade.py
```

### 1. H4 STO-3G Hamiltonian ($n=8$, 184 non-identity Pauli words, `graph_state`)

| Rung | LDF settings | LDF $\hat{R}$ | LDF CZ total (max) | SI settings | SI $\hat{R}$ | SI CZ total (max) | $\hat{R}$ ratio (SI / LDF) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $k=1$ (QWC) | 68 | 2.8990 | 0 (0) | 69 | 7.6480 | 0 (0) | 2.64× |
| $k=2$ | 60 | 2.7746 | 36 (2) | 67 | 8.2793 | 16 (4) | 2.98× |
| $k=4$ | 21 | 6.5922 | 60 (6) | 30 | 13.7249 | 112 (9) | 2.08× |
| $k=8$ (FC) | 9 | 17.3779 | 48 (10) | 10 | 20.7946 | 47 (9) | 1.20× |

Notice that under unweighted LDF, increasing from $k=1$ to $k=2$ reduces
settings (68 → 60) while *lowering* $\hat{R}$ (2.8990 → 2.7746), illustrating
why setting count alone is not a variance metric. Weighted sorted insertion
uses slightly more settings at every rung while achieving higher $\hat{R}$
across all four rungs.

### 2. BeH2 STO-3G Hamiltonian ($n=8$, 60 words) under `superconducting-like` noise gate

With `superconducting-like.json` ($p = 0.01$, routing multiplier $3.0$) and
$\varepsilon = 0.10$, the gate allows at most `10` routed 2Q gates (`3` logical
CZs per setting):

| Plan | Settings | $\hat{R}$ | Grouped variance | CZ total (max) | Admissible settings |
| --- | ---: | ---: | ---: | ---: | ---: |
| QWC ($k=1$, SI) | 13 | 21.3313 | 0.6387 | 0 (0) | 13 / 13 |
| Unconstrained FC ($k=8$, SI) | 4 | 26.8665 | 0.5071 | 18 (6) | 1 / 4 |
| GALIC noise-gated FC ($k=8$, SI) | 7 | 25.1569 | 0.5416 | 18 (3) | 7 / 7 |

At a tighter $\varepsilon = 0.05$ (`max_routed_two_qubit_gates = 5`,
`max_logical_two_qubit_gates = 1`), the gate falls back cleanly to 0–1 CZ
settings (QWC on BeH2).

### 3. Density-free Ritz-weighted replanning on 4-qubit TFIM ($k=2$)

Using an independent 200-shot-per-setting pilot (`seed=11`, pilot energy
`-4.02007`) and replanning with `with_ritz_plan(..., noise_gate=gate)` on the
same `superconducting-like` gate improves the Ritz-functional $\hat{R}$ from
`3.2622` (9 settings, grouped variance `35.0380`) to `4.6041` (11 settings,
grouped variance `24.8260`, 11/11 admissible).

## Scope and non-claims

- $\hat{R}$ is a state-independent Neyman-allocation variance proxy under
  disjoint word assignment; actual shot savings depend on state covariances
  (reportable via `evaluate_plan_variance(..., cache=...)`), shot-allocation
  policy, and hardware noise.
- `superconducting-like.json` is an illustrative accounting card, not a
  calibrated device characterization or hardware execution claim.
- These diagnostics verify software behavior and algebraic/variance tradeoffs;
  they are not a new preregistered accuracy campaign and do not overturn the
  closed Step 5 or Phase 14B conclusions.

## Primary references and validation

- O. Crawford, B. van Straaten, D. Wang, T. Parks, E. Campbell, and S. Brierley,
  *Efficient quantum measurement of Pauli operators in the presence of finite
  sampling error*, Quantum **5**, 385 (2021).
- B. DalFavero, R. Sarkar, J. Rowland, D. Camps, N. P. D. Sawaya, and R. LaRose,
  *Measurement reduction for expectation values via fine-grained commutativity*,
  Phys. Rev. A **112** (2025), [doi:10.1103/y6hl-gzr4](https://doi.org/10.1103/y6hl-gzr4),
  [arXiv:2312.11840](https://arxiv.org/abs/2312.11840).
- M. X. Burns, C. Liu, S. A. Stein, B. Peng, K. Kowalski, and A. Li,
  *GALIC: Hybrid Multi-Qubitwise Pauli Grouping for Quantum Computing Measurement*,
  [arXiv:2409.00576](https://arxiv.org/abs/2409.00576) (2024).

```bash
python -m pytest -q tests/test_weighted_measurement_planning.py
python examples/measurement_planning_upgrade.py
```

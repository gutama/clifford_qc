# Phase 16A preregistration: time-evolved QSCI against selection at equal size

Phase 16A shipped an input (`subspace/time_evolution.py`): configurations for
QSCI sampled from `exp(-iHt)|reference⟩`, either exactly propagated (`oracle`)
or prepared by a validated Trotter circuit (`implementable`). An input is not a
result. This declaration fixes, before anything is sampled, the one comparison
that decides whether the input is worth carrying forward. Its config is
`benchmarks/configs/phase16a_te_qsci.json`, and its gate is
`benchmarks/check_phase16a_preregistration.py`.

## 1. Question (Q15)

On an active space where the reference determinant misses chemical accuracy:

1. does QSCI on configurations pooled from Trotter-circuit time-evolved
   reference states reach `1.6e-3` Ha, as a median over replicas, within the
   frozen shot grid; and
2. at the first budget where it does, is its determinant set better than the
   one classical selected CI picks at the same size without seeing the sample?

The second half is the project's standing question about QSCI (`PLAN.md` §1,
§12): whether a sampled subspace adds anything classical selection does not
already give. A time-evolved input that reaches the target only by sampling
the configurations a laptop would have chosen anyway has not earned a place.

## 2. The control, and why it is not ADAPT-VQE

The primary control is `run_control(kind="matched_selected_ci")`. It anchors on
the reference, ranks the determinants coupled to it by Epstein–Nesbet score,
and keeps as many as the candidate sampled in the same replica. It never reads
the sample's contents, only its size, so any margin over it belongs to *which*
configurations time evolution found. It is classical, and it consumes no
sampling input, so placing it beside an implementable arm does not break §8D's
split between oracle and implementable inputs.

An ADAPT-VQE sampling input was the first choice of comparator and was
dropped for a measured reason. The package's exact ADAPT runs on density
multivectors. On the 12-qubit H₂O space, with its 640-word qubit pool, it did
not complete four operators in ten minutes. That would put the gate past CI's
timeout and make every replica cost more than the experiment it serves. A
comparison against ADAPT inputs needs a statevector ADAPT first, and that is a
separate piece of work.

## 3. Instances and admission

Admission reads zero-noise quantities only: the exact sector spectrum, the
declared states' exact probabilities, and one exact solve. It never reads a
sampled draw, the control, or a paired comparison. A required instance must
satisfy three clauses:

- **the reference determinant misses the target;**
- **zero-noise reachability:** with `p` the equal-shot mixture of the
  candidate's Trotter states, the configurations whose expected raw count
  `65536 · p(c)` is at least one give a QSCI error within the target in exact
  arithmetic. Without this the comparison is censored by construction, the
  v1 failure of Phase 16B;
- **non-saturation:** that expected support is smaller than the sector, so a
  matched selection still has to choose.

| instance | qubits | sector | reference error (Ha) | expected support | its exact error (Ha) | admitted |
|---|---:|---:|---:|---:|---:|---|
| H₂O CAS(8e,6o) | 12 | 225 | 4.956e-2 | 59 | 9.2e-6 | yes |
| H₄ chain, 0.9 Å | 8 | 36 | 5.606e-2 | 20 | ~0 | yes |
| BeH₂ CAS(4e,4o) | 8 | 36 | 5.899e-3 | 6 | ~0 | yes |
| LiH CAS(4e,4o) | 8 | 36 | 1.196e-3 | — | — | no: the reference already meets the target |
| Hubbard 2×3, U = 4 | 12 | 400 | 3.62 t | 337 | 0.149 t | no: censored by construction, and not in Ha |

All three required instances are committed FCIDUMPs, so the gate and the
producer need no chemistry extra, and the gate runs in CI's structural matrix.

**One clause changed after the structural screen, and this says so.** The
non-saturation clause was first written as "sector dimension ≥ 100", which
would have left H₂O alone. After the screen, and before any sampled draw, it
became the support clause above, which admits H₄ and BeH₂. No sampled quantity
informed the change. The config labels it `declared_after_structural_screen`.

Two small instances are still a narrow base. A verdict here is evidence about
these three active spaces, not about chemistry at large.

## 4. Time grid and circuits

`t_k = c_k · τ` with `τ = 1/σ_ref`, where `σ_ref² = ⟨ref|H²|ref⟩ − ⟨ref|H|ref⟩²` on
the declared sector, and `c ∈ {1/4, 1/2, 1, 2}`. First-order weight leaving the
reference is `σ_ref² t²`: 1/16 at the first time and 1/4 at the second, and the
last two lie past the perturbative regime. The grid follows from that
argument, and no sampled draw informed it.

Each time is prepared by the second-order `trotter_program` in Pauli-code
order. Its step count is the smallest of 1, 2, 4, …, 256 whose circuit state
reaches fidelity 0.999 with exact propagation of the same reference. The
selected counts (H₂O `2, 4, 8, 16`; H₄ `1, 2, 4, 8`; BeH₂ `2, 2, 4, 16`) are
frozen in `measured_before_freezing`, and the gate recomputes them. It also
refuses a choice whose fidelity sits within `1e-6` of the threshold; the
smallest margin is `1.7e-4`, on H₄. Circuits run in the full register, and the
small sector leakage of single rotors is post-selected (discards count against
the budget).

## 5. Arms

| arm | role | category | what it is |
|---|---|---|---|
| `te_trotter_pooled` | primary candidate | implementable | the four Trotter states, `total/4` shots each, pooled by `sample_state_inputs`, solved by `run_qsci` |
| `matched_selected_ci` | primary control | classical | `matched_selected_ci` at the candidate replica's configuration count |
| `te_exact_pooled` | diagnostic | oracle | the same times, exactly propagated: separates Trotter error from the method |
| `exact_ground_oracle` | diagnostic | oracle | the exact ground state, all shots on one state: the sampled-QSCI ceiling |
| `reference_determinant` | floor | implementable | one configuration; the error admission requires to miss |

Diagnostics and the floor cannot promote. Oracle and implementable arms are
reported in separate tables.

## 6. Shots, replicas, seeds

Total budgets are `2^6 … 2^16`, split equally over the four times, with 100
replicas per cell. Each sampled stream is
`SeedSequence(20260926, spawn_key=(instance, arm, budget, replica))`, so no two
arms or cells share draws. The control is deterministic given `M_r`.

## 7. Decision rule

- **Shots-to-target** is the smallest tested budget at which the candidate's
  median error meets the target. There is no interpolation.
- **At that budget,** the per-replica paired difference is
  `error(candidate) − error(control at the same size)`.

| instance status | condition |
|---|---|
| PASS | shots-to-target exists and the median paired difference is `< −1e-10` Ha |
| FAIL | shots-to-target exists and the median paired difference is `≥ −1e-10` Ha |
| UNDETERMINED | no tested budget reaches the target |

All three PASS gives **GO**, all three FAIL gives **NO_GO**, and anything else
is **CONDITIONAL**. An UNDETERMINED required instance can never yield GO.
A failed deterministic check makes an instance INVALID. Such a check is
admission no longer holding at execution, a step count or support differing
from the frozen one, a configuration outside the sector, a missing replica, or
missing provenance. No follow-up is permitted: another time grid, shot grid,
control or instance set would be a new declaration.

## 8. What this can and cannot conclude

A GO means that on these three active spaces a Trotterized time-evolved input
reaches chemical accuracy within `2^16` shots. It also means that its
configurations beat what a sample-blind selection of the same size finds. A
FAIL means it reaches the target by sampling what selection would have picked.
Excluded for every arm: circuit depth and two-qubit counts (rotor counts are
recorded, not priced), classical cost, and device noise. The draws come from
exact probabilities. No hardware, scaling or quantum-advantage claim follows,
and the record must carry `quantum_advantage_claim: false`.

## 9. Disclosure

While 16A was implemented, pooled time-evolved QSCI was smoke-tested on H₄ and
`hubbard(4)` at `t ∈ {0.5, 1, 2}`, with 200 shots per time, one seed and no
control. That used another time grid, with no replicas and no matched control,
so none of the quantities the decision rule reads was observed. H₄ stays
required on the Phase 16B v1 precedent for disclosed exploratory pilots.

## 10. Order

The config and this gate land first and pass before any producer exists. The
producer (`run_phase16a_te_qsci.py`) must re-run the gate and refuse to draw
if it fails. The gate binds the FCIDUMPs, their provenance and
`time_evolution.py` by SHA-256, so a changed input cannot reach a record under
this declaration. Once the record exists, the gate checks from git history
that the config's commit strictly precedes it.

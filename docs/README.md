# Documentation guide

This guide connects the package's current interfaces to the contracts, research
results and reproduction instructions that describe them. The project is research
software: an implemented interface, a passing test, a reproduced record and a
scientific advantage are different claims.

## Start here

| Need | Primary document |
| --- | --- |
| Install, try examples and choose a solver | [Project README](../README.md) |
| Understand layers, execution paths and responsibilities | [Architecture](../ARCHITECTURE.md) |
| Check Pauli, state, sector and numerical conventions | [Conventions](../CONVENTIONS.md) |
| Prepare and solve a molecular problem | [Molecular pipeline](../molecular/PIPELINE.md); [simulation suite](../molecular/README.md) |
| Inspect the current restart scope and compatibility | [Restart roadmap](restart/README.md), [contract](restart/CONTRACT.md), [compatibility](restart/COMPATIBILITY.md) |
| Reproduce a committed result | [Reproduction ledger](../REPRODUCING.md), starting with the campaign links below |

The README is the entry point, not a complete API or evidence specification.
The architecture describes design boundaries; conventions define mathematical
semantics; workflow guides give usage and scope. For a particular experimental
claim, use its declaration, result record and checker together.

## Workflows and contracts

| Task | Guide | Boundary |
| --- | --- | --- |
| Reuse exact molecular inputs | [Molecular pipeline](../molecular/PIPELINE.md) | Preparation, solve and optional validation are separate; storage choices do not bound total RSS. |
| Use the shared result and export | [Unified workflow](restart/UNIFIED_WORKFLOW.md) | Opt-in wrapper; provider preparation, independent validation and persistent storage remain caller responsibilities. |
| Recover roots with sector actions | [Sector actions](restart/SECTOR_ACTIONS.md) | Bounded numerical recovery and residuals do not by themselves identify the ground root. |
| Sample projections and ratios | [Sampled ratio](restart/SAMPLED_RATIO.md), [cross-domain validation](restart/CROSS_DOMAIN_RATIO.md) | Independent pilot/evaluation stages, denominator checks and declared input scope matter. |
| Compile graph readouts | [Graph readouts](restart/GRAPH_READOUTS.md) | Logical circuit results are distinct from matched-accuracy shot costs or measured hardware gains. |
| Plan weighted and noise-gated measurements | [Weighted measurement](restart/WEIGHTED_MEASUREMENT.md) | Variance-first `R_hat` and GALIC 2Q-gate budgets are software/accounting diagnostics, not hardware or matched-accuracy claims. |
| Report sampled second moments and real-time responses | [Phase 15/16 completion](restart/PHASE15_16_COMPLETION.md), [completion contract](restart/PHASE15_16_COMPLETION_CONTRACT.md) | The sampled and polynomial workflows have declared input, uncertainty, truncation and conditioning limits. |
| Record, resume and validate executions | [Runner and artifacts](restart/RUNNER_ARTIFACTS.md), [replay verification](restart/REPLAY_VERIFICATION.md) | Immutable receipts and recorded numerical replay do not imply byte-identical strict replay. |
| Choose a phase convention | [Deterministic phase pivot](deterministic_phase_pivot.md) | The tolerance-based convention is opt-in and changes the representative, not the state ray. |

## Research evidence and reproduction

Each declaration fixes the relevant comparison and cost boundary; the linked
result or workflow gives its conclusions and limitations. The
[reproduction ledger](../REPRODUCING.md) holds detailed commands, pinned
environments, provenance and checker scopes. Consult the specific sections
before attempting a replay. The [independent-check priorities](restart/INDEPENDENT_CHECK_PRIORITIES.md)
identify what is checked independently and what remains outside those checks.

| Research question | Declaration or protocol | Result and interpretation |
| --- | --- | --- |
| Do sampled ratios reduce cost across declared inputs? | [Original protocol](restart/SAMPLED_RATIO.md), [cross-domain protocol](restart/CROSS_DOMAIN_RATIO.md) | Both all-input 20% cost hypotheses closed negative; a TFIM-only saving is not a molecular or ground-state-accuracy claim. |
| Does noise-aware selection improve held-out equal-shot performance? | [Noise-aware selection](restart/NOISE_AWARE_SELECTION.md) | The declared held-out criterion closed negative. |
| Do intervals on measured Ritz values reach their nominal coverage? | [Interval-coverage declaration and result](restart/INTERVAL_COVERAGE.md) | No construction was calibrated for every estimator on three four-qubit banks; the basic bootstrap came closest on thresholded solves. Small systems and ideal shots only. |
| Does targeting shots at the Ritz value save shots, and is a stopped interval valid? | [Target-aware acquisition declaration and result](restart/TARGET_ACQUISITION.md) | Never dearer at matched error and 1.4 to 2.7 times cheaper in five of eight cells. The stopped interval over-covered on three four-qubit banks and under-covered on the held-out six-qubit bank. Small systems and ideal shots only. |
| Does a bootstrap interval at the acquisition stop restore coverage? | [Bootstrap interval at the acquisition stop](restart/BOOTSTRAP_STOP.md) (criteria fixed before a fresh-seed run, not a Git-provable declaration) | Yes on the declared study's four spin banks: 99% to 100% coverage on the held-out six-qubit bank, where the delta interval covered 84.5% to 93.5%. Median width 1.2 to 2.0 times the delta interval there and up to 7.2 times on another bank. Stopping only when the bootstrap also meets the target cost up to 3.2 times the shots and was not adopted. |
| Does refitted symmetry correction retain molecular interval coverage? | [Preregistered molecular interval coverage](restart/MOLECULAR_INTERVAL_COVERAGE.md) | LiH/H4, fixed shots, 256 paired outer replicas: no rejection of 95% corrected-spread coverage, median width 22–46% smaller and point RMSE 18–50% lower. 68% spread coverage descriptively below nominal in all cells. |
| Does adaptive symmetry-corrected acquisition stopping work on molecules? | [Preregistered molecular adaptive stopping](restart/MOLECULAR_ADAPTIVE_STOP.md) | 256 acquisitions on LiH/H4; all meet the 1.6 mHa estimated-SE target. Primary 95% CV schedule-spread coverage 62/64 and 61/64; neither rejects nominal coverage. Mean shot savings 16.4% and 34.7%; H4 RMSE rises 12.4%. Stable rank, ideal shots only, not an anytime-valid interval. |
| Does the SE stop retain precision when molecular rank changes? | [Preregistered stretched molecular rank stress](restart/STRETCHED_MOLECULAR_STOP.md) | Negative for precision: ten-/twelve-qubit LiH/BeH2, rank changes in 55/128; all meet the 1.6 mHa estimated-SE target but 39/128 miss it in absolute error (all 24 reduced-rank stops). Neither CV 95% spread primary cell rejects coverage; LiH CV median width 531.6 mHa. Engineered stress, ideal shots only. |
| Can an opt-in rank/margin or bootstrap-width guard address false precision stops? | [Preregistered rank-aware stopping comparison](restart/RANK_AWARE_STOP.md) | Both rank/margin feasibility cells pass: 31/32 true stops vs 25/32 for SE-only, no reduced-rank stops; 77–87% fewer mean shots than fixed budget. Width-only 32/32 true stops at higher shot/solve cost. Small engineered stress, not a guarantee. |
| Do weighted $k$-commuting grouping, Ritz replanning, and GALIC noise gating improve density-free adaptive precision? | [Preregistered weighted precision study](restart/WEIGHTED_PRECISION.md) | Result-free declaration committed; evaluation pending execution at the declaration commit. |
| Can symmetry shifts cut the shots a measured Ritz value needs? | [Symmetry-shift control variates](restart/SYMMETRY_SHIFTS.md) (exploratory, not declared) | 1.4 to 7 times lower mean-squared error at fixed shots on three spin-adapted molecular banks. With targeting on a six-qubit bank, 9 to 13 times fewer shots to a standard-error target, with the stopped interval covering. On H4, 10.5 times fewer shots with targeting alone and 12.6 times with the correction too (40 replicas). Needs generators that conserve the charge; none of the declared spin banks does. |
| Can observables or projected quantities remain stable under noise? | [Observable-noise investigation](restart/OBSERVABLE_NOISE.md) | Exploratory bounded study; stability of an observable or overlap does not establish stable Ritz energies. |
| Do residual packets improve the classical work/accuracy tradeoff? | [Packet budget declaration](../benchmarks/PACKET_BUDGET_CONVERGENCE_DECLARATION.md) | [Budget report](../benchmarks/results/packet_budget_convergence.md): mixed compactness and substantial action cost; the packet performance line is closed. |
| What do sparse growth and caching change? | [Sparse packet declaration](../benchmarks/SPARSE_PACKET_WORK_DECLARATION.md), [cache declaration](../benchmarks/EXACT_PROJECTION_CACHE_DECLARATION.md) | [Sparse report](../benchmarks/results/sparse_packet_work.md) and [cache report](../benchmarks/results/exact_projection_cache.md) separate timing, memory, accuracy and held-out limitations. |
| How does a small molecular residual-growth comparison behave? | [H2O declaration](../benchmarks/H2O_RESIDUAL_COMPARISON_DECLARATION.md), [reuse declaration](../benchmarks/PACKET_WORK_REUSE_DECLARATION.md) | [H2O report](../benchmarks/results/h2o_residual_comparison.md) and [reuse report](../benchmarks/results/packet_work_reuse.md) are small-sector classical evidence. |

For other campaigns, use the [reproduction ledger](../REPRODUCING.md),
[benchmark declarations](../benchmarks/) and [result reports](../benchmarks/results/).
The [phase status](../PHASE_STATUS.json) tracks the research phases; the
[research plan](../PLAN.md) is a detailed phase ledger, not a short current
user guide. The separate [interoperability roadmap](../simple_plan.md) uses
its own phase numbering. The original [restart blueprint](restart/blueprint_2026-10-24.md)
and frozen research declarations retain their historical context and should
not be silently rewritten to match a later result.

## Papers and citations

| Material | Scope |
| --- | --- |
| [Architecture paper](../paper_architecture/README.md) | Software architecture and evidence; [arXiv:2609.23059](https://arxiv.org/abs/2609.23059). |
| [Compactness paper](../paper_compactness/README.md) | Draft analysis of residual-packet compactness and classical costs. |
| [Measurement/ADAPT manuscript](../paper/README.md) | Measurement and adaptive-selection research. |
| [A-CASE source snapshot](../paper_acase/README.md) | Historical paper source and its own provenance boundary. |
| [DA-CASE manuscript](../paper_dacase/README.md) | Separate manuscript and validation scope. |
| [Citation metadata](../CITATION.cff) | Software and research citations. |

Identify the method and software commit when reporting a result. Paper versions,
frozen inputs, code revisions and current implementations need not coincide.

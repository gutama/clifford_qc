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

# Priorities for independent evidence checks

The repository census counts same-stem producer/gate pairs; a missing pair is a
screening signal, not proof that a result has no check. This triage ranks claims
by publication use, consequence and remaining independence gap as of 1 November
2026. It does not rewrite any historical records.

| Priority | Claim/producer | Existing coverage | Remaining action |
| --- | --- | --- | --- |
| 1 | Published H4 FCIDUMP `run_fcidump_h4.py`: 9-direction energy, complete SD and Pauli-word count | Frozen paper input and producer replay test; independently supplied PySCF FCI energy | Independently reconstruct occupation-sector Hamiltonian, every selected-prefix energy and complete-SD energy; independently multiply Pauli letters for word count and largest H-row support. Do not claim an independent reconstruction of the original adaptive selection decisions or Python heap bytes. |
| 2 | Sampled cross-domain ratio cost/accuracy (`run_cross_domain_ratio.py`) | Existing independent occupation/tensor checker, all outcomes, full-seed replay and negative all-input result | Retain as a required gate; audit new scientific claims against its explicit scope. Avoid duplicating its checker. |
| 3 | Noise-stable observable claims (`run_observable_noise.py`) | Existing independent dense Kraus/adjoint checker, controls, full grid and tests | Retain as a required gate. For any new selector, rebuild the outcome from the frozen declaration with an independent implementation. |
| 4 | Stretched H2O memory feasibility (`run_packed_h2o_feasibility.py`) | Retained historical observed runs and cost boundaries; repeating the original 15-GiB workload needs its declared hardware | Audit records and monitoring methodology separately; numerical replay on a smaller host cannot certify the original process peak or OOM. Do not claim a new memory replication without a separately declared controlled run. |
| 5 | Legacy ladder/calibration and exploratory drivers | Mixture of producer replays, structural and paper gates | Triage individual downstream claims before allocating rebuild work. Do not infer that all missing same-stem checkers are uncovered or equally important. |

The new H4 checker is deliberately scoped: it shares FCIDUMP parsing and the
existing independent occupation-sign oracle, and verifies mathematical values
and word-set construction rather than exact timing, memory or the adaptive
winner sequence. Its evidence improves confidence in the committed observations;
it is not a new campaign or a new performance claim.

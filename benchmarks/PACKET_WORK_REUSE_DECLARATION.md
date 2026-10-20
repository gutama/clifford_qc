# Residual packet work reuse and screening

Exploratory follow-up to the known H2O comparison: packets reached 1.6 mHa
with 12 columns and 213 H actions; spin-adapted selected CI used 14 and 105.
Freeze this declaration and configs/packet_work_reuse.json before implementing
the new modules or observing molecular outcomes. Preserve earlier evidence.

Use exactly the existing 225-determinant H2O STO-3G CAS(8e,6o) input, singlet
frame, initial CAS(2e,2o) on [3,4], diagonal preconditioner, orthogonalization,
energy-lowering score and stopping rules from the H2O comparison. Five arms:
original packets, frame-action reuse only, shortlisting only, both, and the
existing spin-adapted selected-CI control. Run three fresh processes per arm
in rotated order, one numerical thread, 120 seconds per worker. No selective
reruns. Failures count in every success denominator.

Reuse: cache H applied to fixed singlet-frame columns lazily by group, seed
the initial frame columns from already counted HV. For each refreshed packet,
combine cached HF actions and propagate both vector and action through two
whole-span orthogonalization passes, normalization and canonical phase. Keep
all populated actions; report their payload and the additional frame snapshot.
The H-action ceiling is the complete frame dimension (105), not a runtime or
memory guarantee. Match the unshortlisted original scientific trajectory up
to numerical ties. No cache-eviction policy is tested in this first comparison.

Shortlist: build all admissible directions without their H actions. For each
group use squared norm of its fixed-frame projection of the UNPRECONDITIONED
full Ritz residual. Sort descending by this indicator, breaking exact ties by
integer group key, and retain the smallest prefix covering at least 80% of
the admissible groups' total indicator. If the total is zero, retain all.
Every fifth growth decision is a full-screen audit: evaluate all admissible
candidates, select the full-screen winner, and record whether it was omitted
and the maximum score lost by shortlisting. Between audits, if the best
shortlisted exact lowering is <=1e-12 Ha, evaluate the omitted candidates and
select the full-screen winner. Charge every audited/fallback action. Empty
shortlists must never masquerade as full convergence. This is a heuristic;
no adaptive-finite-element theorem is claimed for it.

Record every prefix's energies, true residual, spin, selected corrections,
support, total H calls, exact scores, cheap candidate counts, cache indices,
frame/cache/V-HV payloads, cumulative time and OS process high-water RSS.
Each decision records indicators, shortlist/evaluated keys, audit/fallback
flags, scores and audit loss. Setup is included in time-to-target; imports
and process startup are priced separately. Cache payload is not process RSS.
Only after all timed workers finish, evaluate the independent occupation-H
reference and mark the fixed 1.6 mHa target. Report successes, first-target
costs, complete terminal costs, audit misses and trajectory agreement. Do not
tune theta, audit frequency or tolerances after observing outcomes.

Before execution require clean committed sources (including untracked files),
the repository record environment, input/design digests and full Git history.
Bind execution to complete source digests and declaration ancestry. Commit
observations separately, bind their complete bytes in a publication receipt,
and preserve required history in a digest-bound Git bundle. Independently
reconstruct occupation-H, spin, packet vectors, shortlists, decisions, cache
population, work counts and every prefix. Include adversarial record tests.

These are explicitly stored classical small-sector vectors and dense singlet
setup. Full HF caching can approach a complete singlet action table. No
large-sector scaling, physical shots, quantum state preparation, hardware,
optimized SHCI/DMRG superiority or quantum advantage is established.

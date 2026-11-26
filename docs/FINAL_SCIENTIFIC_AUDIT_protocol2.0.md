# Final Scientific Audit of protocol 2.0 (HISTORICAL, body preserved unchanged, only this title line edited; superseded by FINAL_SCIENTIFIC_AUDIT.md)

Branch `audit/final-scientific-validation` (from `audit/federated-lifecycle-fixes`). Machine-readable counterpart: `docs/final_validation_report.json` (regenerate: `python scripts/final_validation.py --probe-episodes 3`). Nothing here is a performance result: **no comparative experiment has been run, and no claim of superiority is made.** The model and methodology were not redesigned.

Full suite: 292 passed, 0 failed (CPU, Python 3.13); validate_manifest exit 0; compileall clean.

## Recommendation: **NO-GO** for launching the publication-scale protocol as currently configured
The software is validated (all validators pass, see below). The NO-GO is about the *experimental design*, which needs author decisions (Section 8) and a small non-publication pilot first. It flips to GO when those are closed and the manifest is re-frozen.

## 1. Episode-count discrepancy (200 vs 500)
Manifest said 200, YAML said 500. Resolved to **500** (Paper 2 Table I: 500 episodes, checkpoint every 25); 200 had no source. Introduced `experiment.protocol_version: "2.0"`, a changelog, mirrored fields (eval_every, n_train_seeds, checkpoint_every, episode_steps) and a pinned `config_hash` (`7f4a0e4f66569528`). `scripts/validate_manifest.py` now exits 1 ("MANIFEST INVALID") on any critical mismatch including hash drift, so any later YAML change forces a version bump. Tests: `tests/test_manifest_validator.py`.

## 2. Adverse-condition lifecycle (deterministic, executed)
Eight scenarios, each checking references, counters, statistics settlement and invariants (check counts): delayed upload 18, stale rejection 15, server rejection 15, expiration 31, duplicate delivery 17, client reconnection 22, asynchronous aggregation 16, episode transition with unresolved updates 15. All pass; runs are deterministic; mutation tests (e.g. receipt advancing the reference) confirm the checks can fail; two tests drive the real `Trainer` across episode boundaries. Probe: a persistently rejected client re-uploads immediately (30 full-model uploads in 120 s) — no back-off exists (risk R4).

## 3. Semantic relevance
- Attribution: the AUV1 example (batch collected by AUV1 at t=100, delivered t=250 while AUV0 delivers its own) gives AUV1 rho_sem = 0.212 from its own batch only and AUV0 0.779; no cross-attribution.
- Real-environment mass balance: |U_ledger − accounted| = 2.5e-11, 1.2e-10, 1.6e-10 (U ≈ 2.2–2.6e4) in 3 scenarios. Partial/incomplete/late batches, claim-once and censoring are covered by tests; rho_sem ∈ [0,1] with no clipping (inconsistent ledger raises).
- Weight stability: relevance factor ∈ [0.1, 1.0] (eps_M=0.1, eta=1) so relevance alone can change a relative weight ≤10×; analytic |dw/dM| bound 2.25, empirical max 1.21; weights valid everywhere.
- Maturity-threshold sensitivity (eps 0.01–0.5, descriptive, **not used to choose**; 0.05 fixed a priori): pooled rho_sem 0.1095–0.1121; Spearman vs eps=0.05 0.96–1.0; censored value 2.9%→1.1%; late-unit share 0.17%→3.0%. The metric is insensitive to eps.

## 4. MATD3 and federation
Verified numerically: twin critics with clipped double-Q, target smoothing with noise clip, delayed actor, Polyak, actor-only federation (critics/targets/optimisers untouched), critic-sync policy, weighting/staleness, parameter sync, finiteness, no silent actor corruption; 5 mutation tests. Probe (3 episodes, full length): all parameters finite, accounting balanced.

## 5. Baseline fairness
Numerically equal across the 11 methods: observations, action constraints, environment, step budget, evaluation scenarios, seeds. Declared non-equivalences N1–N10 (full text in the JSON), the material ones:
- N2/N3/N4: MAPPO has ~31× fewer gradient minibatches (499,500 vs ~16,000), a Gaussian shared actor.
- N1: greedy/Hungarian use a different allocator.
- N5: federated methods scored with the global actor.
- N8: CTDE critics are not communication-constrained.
- N9/N10: independent TD3 and MADDPG are deliberately weaker by definition.
Differences against MAPPO must be reported with these caveats.

## 6. Statistics
The precision-loss warning came from constant paired differences (t = inf). Now handled: degenerate cases return NaN p-values with a reason; the test passes under `-W error`. Seeds are matched by key; paired t and Wilcoxon, t-CI of the mean difference, Cohen d_z, Hedges g, Holm per metric and per test. With 5 seeds the Wilcoxon minimum p is 0.0625 (cannot reach 0.05); Holm over 10 comparisons needs ≥9 seeds (≥6 uncorrected). Observations from nearly identical policies are not independent evidence; seeds, not episodes, are the unit. No significance is fabricated.

## 7. Measured cost (probe, not a result)
~12 s/episode (3 episodes, 12 updates created/aggregated, 0 rejected/expired, 7 rounds). ≈1.7 h per 500-episode run; ≈90 h sequential for 55 runs; double at 10 seeds.

## 8. Remaining risks and decisions needing the author
- R1 Seeds: 5 is underpowered for the planned tests. Recommend ≥10 (decision needed; protocol 2.1 if changed).
- R2 Relevance may barely discriminate: rho_sem ≈ 0.09–0.14, probe mean M = 0.0116, floor eps_M = 0.1. Ablations may show no effect.
- R3 43% of probe rounds had a single update, where staleness/relevance weights are inert and a lone stale update is applied at full strength.
- R4 No retry back-off after rejection (bandwidth waste). Fix would be a protocol change: needs approval.
- R5 clip_norm=None would accept a huge finite update.
- R6 No evidence that any learner improves at 500 episodes; no comparative experiment exists.
- R7 Actor Adam state is not reset on global install; critics are not communication-constrained.
- R8 Open first-round decisions: maturity eps=0.05; drop undefined relevance components vs score 0; return vs discard rejected stats; client-side expiry after last bit sent.
- R9 No DP, privacy or secure-aggregation claim is made.

## 9. Path to GO
(1) Decide seeds and R4/R8. (2) Small non-publication pilot (a few episodes per method, 2 seeds) to confirm learning signal and relevance dynamic range. (3) Re-freeze manifest (2.1 if anything changed) and re-run `final_validation.py`. (4) Launch full protocol only on explicit approval.

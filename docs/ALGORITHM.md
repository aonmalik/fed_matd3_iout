# Algorithm (S3A-FMATD3)

```
init: identical global actor theta^0 at USV and all AUVs; per-AUV critics/replay/optimisers; server version r=0
for episode e:                                   # federation clock t_g continues across episodes
  reset env (fresh layout/arrivals/fading streams)
  for step:
    a_i = clip(mu_i(o_i) + N(0, sigma_e));  env.step(a)  (arrival -> allocation -> motion/service -> link -> uplink)
    store joint transition once; AUV i admits it to its replay only if it was steering toward a goal (in transit)
    for each AUV i (rate scaled by client_speed):  one MATD3 step if |replay_i| >= warmup
        y_i = r_i + g(1-d) min_k Q'_ik(s', a'_1..a'_M), a'_k smoothed from ALL target actors; critic step;
        every d-th step: actor DPG step + Polyak targets
    protocol.step(t_g):
      for each AUV i:
        if update ready (>= local_steps_per_update, >= min_local_transitions) and no outstanding update and
           scheduler.may_start(link up [, airtime fits window]) :  Delta_i = theta_i - theta_ref; start upload
        upload advances only while link up and the fading draw succeeds; last bit + propagation => USV.receive()
        USV rejects shape-mismatch / non-finite / over-stale; otherwise buffers
        ACK transfer back; reference := snapshot only if no global model was installed meanwhile
        if server version > base_version and link up: download global actor; install (replace or blend)
      USV aggregates when >= K accepted updates are buffered or the oldest waited agg_timeout (sync mode: all clients or timeout):
        Z_i = n_i^a exp(-k D_i) [e + (1-e) M_i]^h ;  w = capped_normalise(Z) ;
        theta <- theta + eta_g sum_i w_i Clip_C(Delta_i) ;  r <- r+1
evaluation: deterministic actors (global actor on all AUVs, and each local actor) on matched scenarios, no learning
```
Aggregation kinds: `fedavg` (Z=n), `stale` (n e^{-k D}), `semantic` (+rho_sem), `service` (+rho_svc), `s3a` (+beta rho_sem+(1-beta) rho_svc).
Time-limit truncation is not terminal (bootstrapped); `d=1` handling is unit-tested.

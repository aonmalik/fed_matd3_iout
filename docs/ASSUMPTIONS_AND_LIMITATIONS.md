# Assumptions and Limitations

## Honest status
* **No scientific result is claimed.** Unit/integration tests and a smoke test pass; a 25-episode pilot (single seed) showed the learned policies did **not** yet solve the task (<1 Mbit
  delivered vs ~85 Mbit for the goal-seeking reference controller) - far below Paper 2's 500-episode budget. Whether S3A-FMATD3 helps is **untested**.
* `~/dt_smarl_step8/` was not available in the build workspace, so Phase 1 inspection and code reuse were not possible; models were re-implemented from the equations in the two papers.
  The frozen Paper-2 manifest was not read (`configs/experiment_manifest.json` is new).

## Parameters taken from the papers (verified)
Region 200 m, Q=2 Mbit / 5000 units, rates {3,5,8,12}x3, AUV and sensor depth 50 m, speeds [1.2,2.2] (v_nom 1.7), R_dc 6 m, d_safe 5 m, 1000 steps, STA weights (1,1,0.05), gamma 0.97,
hidden/batch 128, lr 1e-3, tau 1e-3, replay 20000, policy delay 2, noise 0.8, 500 episodes (P2 Table I); p_c .3, p_h .4, gamma_crit 2.5, gamma_low .5, mu .05/.01, omega .3/.6/1,
f 12 kHz, B 4 kHz, k 1.5, Thorp, Rayleigh, v_s 1500, USV 5 m/s (SAC-TD3).

## Assumptions (not in the papers; all exposed in `configs/`)
Sensor-link frequency 25 kHz; all transmit powers, efficiencies, SNR thresholds, modem rate caps (0.05 Mbit/s); AUV-USV range 110 m; Gilbert-Elliott outage; onboard buffer 20 Mbit and
offload thresholds; initial buffer law U(0,0.5); mu of the 'high' category; propulsion coefficients (P0=20 W, kappa=8); tx/rx/compute powers; max_hover_steps=60; USV centroid relay rule
(P2 uses FIM-based relocation); static semantic hotspot weights (P1 uses a Markov chain); model payload = parameters x 4 bytes.

## Known limitations
* Propulsion/communication/computation constants are placeholders: **calibrate** before interpreting energy results. Evaluation episodes contain no learning, so E_comp = 0 there and model traffic is absent.
* Straight-line, constant-speed travel-time prediction; exact backlog expectation conditional on that T.
* Centralised critics use simulator state (CTDE); critics are not federated.
* Relevance statistics are windowed; rho_sem may saturate; own-sensor overflow depends on the allocator. The weighting changes the effective objective - **not unbiased FedAvg**; no convergence proof.
* No privacy mechanism or threat model; clipping without noise gives no privacy guarantee.
* Potential-based shaping is not strictly policy-invariant under target switching.
* Single link process per AUV, no inter-AUV acoustic interference; unsafe-proximity counts are indicators, not verified collisions.
* The federation clock continues across episodes while layouts reset each episode.
* Reward differs from Paper 2 Eq.(14); only the collision term has a dedicated reward unit test.

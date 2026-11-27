"""Multi-AUV / single-USV IoUT data-collection environment (parallel multi-agent API, CTDE-compatible).

API (documented, PettingZoo-parallel-like, array based):
    obs, info = env.reset(episode_offset)                    obs:  (M, obs_dim)
    obs, rew, terminated, truncated, info = env.step(act)    act:  (M, 2) in [-1,1];  rew: (M,)
    env.state()                                              global simulator state (M-independent layout, used ONLY by
                                                             centralised critics at TRAINING time)
Information available
    * Execution (actors): the local observation only (own position, goal, believed neighbour positions that are
      refreshed ONLY while the AUV-USV link is up, link flag, onboard fill, ...).
    * Training (critics):  global state + joint actions + all local observations (simulator-provided).  Hence training is
      centralised-critic CTDE -- NOT fully decentralised.  Federation exchanges actor parameters only.

Step event order (never changed): arrivals -> (event-triggered) allocation -> motion/service -> AUV->USV link/uplink
-> USV motion -> reward/observation.   Sensor queue: arrival BEFORE service (Paper 2 Eq. (1)-(5)).
"""
from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from ..allocation.semantic_service_cost import AllocState
from ..allocation.service_arrival_predictor import certainty_equivalent_backlog, travel_time_s
from ..allocation.task_allocator import TaskAllocator
from ..communication.acoustic_channel import AcousticLink
from ..communication.packet_delivery import LinkProcess, LinkState
from ..semantic.attribution import BatchBook
from ..semantic.mission_criticality import build_categories, mean_semantic_value_density
from ..semantic.semantic_utility import UtilityLedger
from ..utils.seed import make_rng
from .auv import AUV
from .energy import EnergyAccount, propulsion_power
from .movement import action_to_vel, step_position
from .safety import ProximityTracker, proximity_penalty
from .sensor_node import SensorNode
from .sensor_queue import SensorQueue
from .usv import USV


class MultiAUVIoUTEnv:
    # additive per-client window statistics (all merged by return_window_stats)
    WINDOW_KEYS = ("coll_mbit", "owned_loss_mbit", "steps",
                   "U_mature", "V_mature", "units_del_mature", "units_coll_mature", "n_mature_batches")

    def __init__(self, cfg):
        self.cfg = cfg
        e = cfg["env"]
        self.M = int(e["n_auvs"])
        self.N = int(e["n_sensors"])
        self.L = float(e["region_m"])
        self.dt = float(e["dt_s"])
        self.T = int(e["episode_steps"])
        self.z_auv = float(e["auv_depth_m"])
        self.z_sen = float(e["sensor_depth_m"])
        self.Dn = math.hypot(self.L, self.L)
        self.cats = build_categories(cfg["semantic"])
        self.use_sem = bool(cfg["semantic"]["use_in_decisions"])
        self.sensor_link = AcousticLink(cfg["acoustic"]["sensor_link"], cfg["acoustic"])
        self.usv_link = AcousticLink(cfg["acoustic"]["usv_link"], cfg["acoustic"])
        self.q = float(e["buffer_mbit"]) / float(e["buffer_units"])         # Mbit per data unit (Eq. Q6)
        self.b_max = float(e["buffer_units"])
        hover_d3 = math.sqrt(float(e["r_dc"]) ** 2 + (self.z_sen - self.z_auv) ** 2)
        self.sensor_rate = float(self.sensor_link.rate_mbps(hover_d3))       # Mbit/s at hover (worst case in range)
        self.prop_s2a = float(self.sensor_link.prop_delay(hover_d3))
        self.allocator = TaskAllocator(cfg["allocation"], use_semantic=self.use_sem)
        self.allocator.v_nom = float(cfg["allocation"]["v_nom"])
        self.obs_dim = 2 * (self.M - 1) + 15
        self.state_dim = 4 * self.M + 2 + 2 * self.N + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.M, self.obs_dim), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.M, 2), np.float32)
        self.act_dim = 2
        self.model_traffic_active = np.zeros(self.M, dtype=bool)
        self.model_uplink_active = np.zeros(self.M, dtype=bool)     # an upload is in progress (direction-aware reservation)
        self.sensor_uplink_airtime_s = np.zeros(self.M)             # airtime the sensor-data uplink used in the LATEST env step
        self.episode_offset = 0
        self.step_idx = 0
        # Unique id of the current episode: increments on EVERY reset (episode_offset can repeat, e.g. in evaluation).
        # Window statistics carry it so that statistics of one episode can never be merged into another (audit item 3).
        self.episode_id = 0
        self.maturity_eps = float(cfg["federated"]["maturity_eps"])
        self.book: BatchBook | None = None
        self.censored_V_total = 0.0            # unclaimed collected value discarded at episode resets (diagnostic)
        self.stats_returned = 0                # window-statistics returns merged / refused (diagnostics)
        self.stats_refused_episode = 0
        # Episode-boundary diagnostics (read-only bookkeeping; no effect on dynamics, statistics or attribution):
        self.dropped_window = {k: 0.0 for k in self.WINDOW_KEYS}      # partial-window content discarded by an episode reset
        self.stats_refused_episode_mass = {k: 0.0 for k in self.WINDOW_KEYS}   # finite statistics refused because their episode ended
        self.episode_boundary_log: list[dict] = []

    # ------------------------------------------------------------------ reset
    def reset(self, episode_offset: int = 0):
        cfg, e = self.cfg, self.cfg["env"]
        s = cfg["seeds"]
        if self.book is not None:                                  # censor the previous episode's unclaimed batches
            self._log_episode_boundary()                           # diagnostics only: records what the next two lines discard
            self.book.reset_episode()
            self.censored_V_total += self.book.censored_V
        self.episode_id += 1
        self.book = BatchBook(self.M, self.maturity_eps)
        self.episode_offset = int(episode_offset)
        self.rng_layout = make_rng(s["layout"], "layout", episode_offset)
        self.rng_arr = make_rng(s["arrivals"], "arrivals", episode_offset)
        self.rng_fade = make_rng(s["fading"], "fading", episode_offset)
        self.rng_comm = make_rng(s["comm"], "comm", episode_offset)
        self.step_idx = 0
        rl = self.rng_layout
        # --- sensors
        pos = rl.uniform(0.0, self.L, size=(self.N, 2))
        tr = cfg["traffic"]
        base = np.array(tr["base_rates"], dtype=float) * float(tr["scale"])
        om = np.array(cfg["semantic"]["hotspot_weights"], dtype=float)
        if tr["mode"] == "regional":                       # non-IID: rate by x band (W->E), semantic weight by y band (S->N)
            xb = np.minimum((pos[:, 0] / self.L * len(base)).astype(int), len(base) - 1)
            yb = np.minimum((pos[:, 1] / self.L * len(om)).astype(int), len(om) - 1)
            rates, omegas = base[xb], om[yb]
        else:
            rates = base[rl.integers(0, len(base), self.N)]
            omegas = om[rl.integers(0, len(om), self.N)]
        lo, hi = e["initial_fill"]
        self.sensors: list[SensorNode] = []
        for j in range(self.N):
            qu = SensorQueue(j, self.b_max, self.q, float(omegas[j]), self.cats)
            qu.initialise(0.0, rl.uniform(lo, hi) * self.b_max, self.rng_arr)
            self.sensors.append(SensorNode(j, pos[j], self.z_sen, float(rates[j]), qu))
        dens = np.array([mean_semantic_value_density(s_.queue.omega, self.cats) for s_ in self.sensors])
        self.density = dens / max(dens.max(), 1e-12)
        # --- vehicles
        a_pos = rl.uniform(0.1 * self.L, 0.9 * self.L, size=(self.M, 2))
        self.usv = USV(np.array([self.L / 2, self.L / 2]), cfg["usv"]["speed"], cfg["usv"]["policy"])
        self.auvs = [AUV(i, a_pos[i].copy(), self.z_auv, float(e["onboard_capacity_mbit"])) for i in range(self.M)]
        for a in self.auvs:
            a.belief_pos = a_pos.copy()
        self.link_proc = LinkProcess(self.M, self.usv_link, cfg["acoustic"]["usv_link"], self.rng_fade, self.rng_comm)
        self.links: list[LinkState] = self.link_proc.step(self._slant())      # initial link snapshot
        self.energy = EnergyAccount(self.M)
        self.ledger = UtilityLedger()
        self.prox = ProximityTracker(self.M)
        # accounting
        self.delivered_units = 0.0
        self.collected_units_by_auv = np.zeros(self.M)
        self.boundary_viol = np.zeros(self.M)
        self.travel_by_auv = np.zeros(self.M)
        # Window statistics.  coll_mbit / owned_loss_mbit / steps are accumulated per step.  The *_mature keys are NOT
        # accumulated here: they come from BatchBook.claim() when an update is created, and are only re-filled by
        # return_window_stats() when a non-committed update hands its statistics back (same episode only).
        self.win = {k: np.zeros(self.M) for k in self.WINDOW_KEYS}
        self._prev_comm_e = np.zeros(self.M)       # audit fix: energy-delta baseline must restart with the new EnergyAccount
        self.need_alloc = set(range(self.M))
        self.comm_tx_s = np.zeros(self.M)
        self.comm_rx_s = np.zeros(self.M)
        self.model_traffic_active[:] = False
        self.model_uplink_active[:] = False
        self.sensor_uplink_airtime_s[:] = 0.0
        self.last_realloc = 0
        self.ep_comp = np.zeros(self.M)
        self.rew_comp_sum = {k: 0.0 for k in ("utility", "overflow", "energy", "collision", "boundary", "shape")}
        self._allocate()
        return self._obs_all(), {"state": self.state()}

    # ------------------------------------------------------------------ helpers
    def _apos(self) -> np.ndarray:
        return np.array([a.pos for a in self.auvs])

    def _slant(self) -> np.ndarray:
        h = np.linalg.norm(self._apos() - self.usv.pos[None, :], axis=1)
        return np.sqrt(h ** 2 + self.z_auv ** 2)

    def goal_pos(self, i: int) -> np.ndarray | None:
        a = self.auvs[i]
        if a.to_usv:
            return self.usv.pos
        if a.target >= 0:
            return self.sensors[a.target].pos
        return None

    def sensor_lams(self) -> np.ndarray:
        tr = self.cfg["traffic"]
        return np.array([s.lam(self.step_idx, int(tr["rate_change_step"]), float(tr["rate_change_factor"]))
                         for s in self.sensors])

    # ------------------------------------------------------------------ allocation
    def _alloc_state(self) -> AllocState:
        load = self.collected_units_by_auv.copy()
        load = load / max(load.max(), 1.0) if load.max() > 0 else np.zeros(self.M)
        return AllocState(
            auv_pos=self._apos(), auv_load=load,
            auv_free_mbit=np.array([a.cap_mbit - a.onboard_mbit for a in self.auvs]),
            cur_target=np.array([a.target for a in self.auvs]),
            sensor_pos=np.array([s.pos for s in self.sensors]),
            b=np.array([s.queue.occ for s in self.sensors]), lam=self.sensor_lams(), b_max=self.b_max, q=self.q,
            density=self.density, last_served=np.array([s.last_served_step for s in self.sensors]),
            t_step=self.step_idx, episode_steps=self.T, rate_mbps=self.sensor_rate, d_max=self.Dn)

    def _release_target(self, a: AUV):
        if a.target >= 0 and self.sensors[a.target].owner == a.aid:
            self.sensors[a.target].owner = -1
        a.target = -1

    def _allocate(self):
        """Event-triggered allocation (t=0, hover completion, goal change, idle retry)."""
        if not self.need_alloc:
            return
        freed = sorted(self.need_alloc)
        self.need_alloc = set()
        hovering_locked = {a.hover_sensor for a in self.auvs if a.hover_left > 0}
        method = self.allocator.method
        if method in ("sta", "hungarian_priority"):
            seekers = [a.aid for a in self.auvs if (a.aid in freed or a.target >= 0) and not a.to_usv and a.hover_left == 0]
            st = self._alloc_state()                                # cur_target kept => assignment-continuity term
            for i in seekers:                                       # joint reassignment of freed + in-transit AUVs
                self._release_target(self.auvs[i])
            free = [j for j in range(self.N) if j not in hovering_locked and
                    (self.sensors[j].owner < 0)]
            res = self.allocator.allocate(st, seekers, free)
            for i, j in res.items():
                self._assign(i, j)
        else:                                                       # sequential baselines: only freed AUVs
            for i in freed:
                a = self.auvs[i]
                if a.to_usv or a.hover_left > 0:
                    continue
                self._release_target(a)
                free = [j for j in range(self.N) if j not in hovering_locked and self.sensors[j].owner < 0]
                st = self._alloc_state()
                res = self.allocator.allocate(st, [i], free, freed=i)
                self._assign(i, res[i])

    def _assign(self, i: int, j: int):
        a = self.auvs[i]
        a.target = int(j)
        if j >= 0:
            self.sensors[j].owner = i

    # ------------------------------------------------------------------ dynamics
    def step(self, actions: np.ndarray):
        e, cfg = self.cfg["env"], self.cfg
        t, dt = self.step_idx, self.dt
        actions = np.clip(np.asarray(actions, dtype=float).reshape(self.M, 2), -1, 1)
        en = cfg["energy"]
        p0, kap = float(en["p0_w"]), float(en["kappa"])
        rcfg = cfg["reward"]
        loss_units = 0.0
        self._du_step = 0.0
        self._du_auv = np.zeros(self.M)
        self._loss_owned_step = np.zeros(self.M)
        # 1) arrivals (arrival BEFORE service)
        lams = self.sensor_lams()
        for j, s in enumerate(self.sensors):
            A = self.rng_arr.poisson(lams[j] * dt)
            _, loss = s.queue.arrive(t * dt, A, self.rng_arr)
            loss_units += loss
            if loss > 0 and s.owner >= 0:
                self.win["owned_loss_mbit"][s.owner] += loss * self.q
                self._loss_owned_step[s.owner] += loss
        # 2) goal updates + allocation events
        for a in self.auvs:
            self._update_mode(a)
        if any(a.target < 0 and not a.to_usv and a.hover_left == 0 for a in self.auvs) and t - self.last_realloc >= 10:
            self.need_alloc |= {a.aid for a in self.auvs if a.target < 0 and not a.to_usv and a.hover_left == 0}
            self.last_realloc = t
        self._allocate()
        # 3) motion / service
        dist_before = np.zeros(self.M)
        dist_after = np.zeros(self.M)
        moved_flag = np.zeros(self.M, dtype=bool)
        collected_units = np.zeros(self.M)
        for a in self.auvs:
            i = a.aid
            goal = self.goal_pos(i)
            a.boundary_flag = 0
            if a.hover_left > 0:
                v, th = 0.0, a.theta
            elif goal is None:
                v, th = 0.0, a.theta                               # idle: hold position
            else:
                v, th = action_to_vel(actions[i], float(e["v_min"]), float(e["v_max"]))
                dist_before[i] = float(np.linalg.norm(goal - a.pos))
                new, moved, bviol, trav = step_position(a.pos, v, th, dt, self.L, e["sea_current"])
                a.pos = new
                moved_flag[i] = True
                a.boundary_flag = int(bviol)
                self.boundary_viol[i] += int(bviol)
                self.travel_by_auv[i] += trav
                a.travelled += trav
                dist_after[i] = float(np.linalg.norm(goal - a.pos))
                if bviol:
                    v = 0.0                                          # move rejected -> no propulsion distance
            a.v, a.theta = v, th
            a.transit_steps += int(a.hover_left == 0 and goal is not None)
            # start hover on entering r_dc of the target sensor
            if a.hover_left == 0 and (not a.to_usv) and a.target >= 0:
                s = self.sensors[a.target]
                if float(np.linalg.norm(a.pos - s.pos)) <= float(e["r_dc"]):
                    units_now = s.queue.occ
                    h = max(1, int(math.ceil(self.q * units_now / (max(self.sensor_rate, 1e-12) * dt))))   # Eq. (6), once
                    a.hover_left = min(h, int(e["max_hover_steps"]))
                    a.hover_sensor = a.target
                    s.visits += 1
            if a.hover_left > 0:
                collected_units[i] = self._service(a, t)
                a.hover_steps += 1
        # 4) AUV <-> USV link and data uplink
        self.links = self.link_proc.step(self._slant())
        dU_total = 0.0
        dU_auv = np.zeros(self.M)
        self.sensor_uplink_airtime_s[:] = 0.0
        for a in self.auvs:
            i = a.aid
            l = self.links[i]
            a.link_up = l.up
            if l.up:
                a.belief_pos = self._apos().copy()
                a.belief_age = 0.0
            else:
                a.belief_age += dt
            if l.up and l.success and a.onboard:
                share = self._sensor_uplink_share(i)
                cap_mbit = l.rate_mbps * dt * share
                sent_mbit, du = self._uplink(a, cap_mbit, t, l)
                dU_total += du
                dU_auv[i] = du
                airtime = sent_mbit / max(l.rate_mbps, 1e-12)             # time on air at the full rate (<= share * dt)
                self.sensor_uplink_airtime_s[i] = airtime                 # the protocol may only grant model traffic the rest
                self.energy.tx[i] += float(en["p_tx_elec_w"]) * airtime
        # propulsion energy ledger (Eq. E1/E2); hover => v = 0 => only baseline power P0
        for a in self.auvs:
            self.energy.prop[a.aid] += propulsion_power(a.v, p0, kap) * dt
        # 5) USV motion
        self.usv.step(self._apos(), dt, self.L)
        # 6) safety
        pos = self._apos()
        self.prox.update(pos, float(e["d_safe"]))
        pen = proximity_penalty(pos, float(e["d_safe"])) / float(e["d_safe"])
        # 7) reward
        E_prop_step = np.array([propulsion_power(a.v, p0, kap) * dt for a in self.auvs])
        E_step_total = E_prop_step + self._step_comm_energy()
        M = self.M
        gam = float(self.cfg["matd3"]["gamma"])
        shape = np.where(moved_flag, (dist_before - gam * dist_after) / self.Dn, 0.0)
        du_term = dU_total if self.use_sem else self._du_step
        w = rcfg
        r_team = (w["w_utility"] * du_term / (w["u_ref"] * M)
                  - w["w_overflow"] * loss_units / (w["l_ref"] * M)
                  - w["w_energy"] * E_step_total.sum() / (w["e_ref_j"] * M)
                  - w["w_collision"] * pen.mean()
                  - w["w_boundary"] * np.mean([a.boundary_flag for a in self.auvs]))
        dui = dU_auv if self.use_sem else self._du_auv
        r_loc = (w["w_utility"] * dui / w["u_ref"]
                 - w["w_overflow"] * np.array([self._owned_loss_step(i) for i in range(M)]) / w["l_ref"]
                 - w["w_energy"] * E_step_total / w["e_ref_j"]
                 - w["w_collision"] * pen
                 - w["w_boundary"] * np.array([a.boundary_flag for a in self.auvs], dtype=float))
        zeta = float(self.cfg["federated"]["local_reward_weight"])
        r = (1.0 - zeta) * r_team + zeta * r_loc + w["w_shape"] * shape
        r = np.clip(r, -w["reward_clip"], w["reward_clip"])
        for k, v in (("utility", w["w_utility"] * du_term / (w["u_ref"] * M)),
                     ("overflow", -w["w_overflow"] * loss_units / (w["l_ref"] * M)),
                     ("energy", -w["w_energy"] * E_step_total.sum() / (w["e_ref_j"] * M)),
                     ("collision", -w["w_collision"] * pen.mean()),
                     ("boundary", -w["w_boundary"] * np.mean([a.boundary_flag for a in self.auvs])),
                     ("shape", w["w_shape"] * float(shape.mean()))):
            self.rew_comp_sum[k] += float(v)
        self.step_idx += 1
        self.win["steps"] += 1
        truncated = self.step_idx >= self.T
        in_transit = np.array([a.hover_left == 0 and self.goal_pos(a.aid) is not None for a in self.auvs])
        info = {"state": self.state(), "in_transit": in_transit, "links": self.links, "r_team": r_team, "r_local": r_loc,
                "dU": dU_total, "loss_units": loss_units, "collected_units": collected_units,
                "boundary": np.array([a.boundary_flag for a in self.auvs]), "obs_all": None}
        obs = self._obs_all()
        info["obs_all"] = obs
        return obs, r.astype(np.float32), False, truncated, info

    def _sensor_uplink_share(self, i: int) -> float:
        """Fraction of one step's airtime the sensor-data uplink of AUV ``i`` may use.

        Sensor data and model traffic share the AUV's USV-link modem.  Half duplex: any model transfer (either direction) reserves
        ``model_share``; full duplex: only an upload does (the downlink is a separate budget).  Reflects the transfers that were
        active after the previous protocol step; the protocol additionally caps the model grant by the airtime already used here.
        """
        ul = self.cfg["acoustic"]["usv_link"]
        reserved = self.model_traffic_active[i] if str(ul.get("duplex", "half")) == "half" else self.model_uplink_active[i]
        return 1.0 - float(ul["model_share"]) if reserved else 1.0

    def _owned_loss_step(self, i: int) -> float:
        return float(self._loss_owned_step[i])

    def _step_comm_energy(self) -> np.ndarray:
        """Per-step communication energy (data uplink tx already accumulated in energy.tx; return the step delta)."""
        cur = np.array(self.energy.tx) + np.array(self.energy.rx) + np.array(self.energy.comp)
        prev = getattr(self, "_prev_comm_e", np.zeros(self.M))
        self._prev_comm_e = cur
        return np.maximum(cur - prev, 0.0)

    def _update_mode(self, a: AUV):
        e = self.cfg["env"]
        f = a.fill()
        if not a.to_usv and f >= float(e["offload_hi"]) and a.hover_left == 0:
            a.to_usv = True
            self._release_target(a)
            self.need_alloc |= {x.aid for x in self.auvs if not x.to_usv and x.hover_left == 0}
        elif a.to_usv and (f <= float(e["offload_lo"])):
            a.to_usv = False
            self.need_alloc.add(a.aid)

    def _service(self, a: AUV, t: int) -> float:
        """One hover/service step at sensor a.hover_sensor (rate-limited, onboard-capacity-limited, FIFO)."""
        s = self.sensors[a.hover_sensor]
        free_units = (a.cap_mbit - a.onboard_mbit) / self.q
        max_units = min(self.sensor_rate * self.dt / self.q, free_units)          # Eq. (4)
        got = s.queue.serve(max_units)
        units = 0.0
        for b in got:
            b.status, b.t_collect, b.collected_by = "onboard", (t + 1) * self.dt, a.aid
            a.onboard.append(b)
            units += b.units
            self.book.on_collect(a.aid, b)             # identity of the collected batch is kept for later attribution
        mbit = units * self.q
        a.onboard_mbit += mbit
        a.collected_mbit += mbit
        a.served_units += units
        a.service_time_s += self.dt if units > 0 else 0.0
        self.win["coll_mbit"][a.aid] += mbit
        self.collected_units_by_auv[a.aid] += units
        self.energy.rx[a.aid] += float(self.cfg["acoustic"]["rx_power_w"]) * (mbit / max(self.sensor_rate, 1e-12))
        s.last_served_step = t
        a.hover_left -= 1
        if a.hover_left <= 0 or (a.cap_mbit - a.onboard_mbit) <= 1e-9 or s.queue.occ <= 1e-9 and units <= 0:
            a.hover_left = 0
            self._release_target(a)
            a.hover_sensor = -1
            self.need_alloc.add(a.aid)
        return units

    def _uplink(self, a: AUV, cap_mbit: float, t: int, l: LinkState) -> tuple[float, float]:
        """FIFO upload of onboard data to the USV: end-to-end delivery, utility, age (incl. propagation)."""
        sent = 0.0
        dU = 0.0
        t_del = (t + 1) * self.dt + l.prop_delay_s + self.prop_s2a
        du_units = 0.0
        while a.onboard and sent < cap_mbit - 1e-12:
            head = a.onboard[0]
            hm = head.units * self.q
            if hm <= cap_mbit - sent + 1e-12:
                a.onboard.popleft()
                chunk = head
                sent += hm
            else:
                take = (cap_mbit - sent) / self.q
                chunk, rest = head.split(take)
                a.onboard[0] = rest
                sent = cap_mbit
            chunk.status = "delivered"
            u = self.ledger.record(chunk, t_del - chunk.t_gen, auv=a.aid)
            self.book.on_deliver(a.aid, chunk, u)      # utility goes to the batch it was COLLECTED in, not to "now"
            dU += u
            du_units += chunk.units
        a.onboard_mbit = max(0.0, a.onboard_mbit - sent)
        if not a.onboard:
            a.onboard_mbit = 0.0
        self.delivered_units += du_units
        self._du_step += du_units            # per-step delivered units (used only by the semantics-off reward variant)
        self._du_auv[a.aid] += du_units
        return sent, dU

    # ------------------------------------------------------------------ observation / state
    def _obs_one(self, i: int) -> np.ndarray:
        a = self.auvs[i]
        Dn = self.Dn
        others = [k for k in range(self.M) if k != i]
        bp = a.belief_pos
        rel = [(bp[k] - a.pos) for k in others]
        order = np.argsort([np.linalg.norm(r) for r in rel]) if rel else []
        rel_sorted = np.concatenate([rel[k] / Dn for k in order]) if rel else np.zeros(0)
        goal = self.goal_pos(i)
        gvec = (goal - a.pos) / Dn if goal is not None else np.zeros(2)
        gd = float(np.linalg.norm(goal - a.pos)) / Dn if goal is not None else 0.0
        b_goal, d_goal = 0.0, 0.0
        if a.target >= 0 and not a.to_usv:
            s = self.sensors[a.target]
            T = float(travel_time_s(np.linalg.norm(s.pos - a.pos), self.allocator.v_nom))
            b_goal = float(certainty_equivalent_backlog(s.queue.occ, s.lam(self.step_idx, int(self.cfg["traffic"]["rate_change_step"]),
                                                                          float(self.cfg["traffic"]["rate_change_factor"])), T, self.b_max)) / self.b_max
            d_goal = float(self.density[a.target]) if self.use_sem else 0.0
        n_sat = sum(1 for s in self.sensors if s.queue.fill >= 0.99) / self.N
        o = np.concatenate([
            rel_sorted, gvec, a.pos / self.L, [gd, float(a.to_usv), a.fill(), b_goal, d_goal, float(a.link_up),
                                                min(a.belief_age / 100.0, 1.0), n_sat, float(a.boundary_flag)],
            (self.usv.pos - a.pos) / Dn])
        assert o.shape[0] == self.obs_dim, (o.shape, self.obs_dim)
        return o.astype(np.float32)

    def _obs_all(self) -> np.ndarray:
        return np.stack([self._obs_one(i) for i in range(self.M)])

    def state(self) -> np.ndarray:
        parts = []
        for a in self.auvs:
            parts += [a.pos[0] / self.L, a.pos[1] / self.L, a.fill(), float(a.to_usv)]
        parts += [self.usv.pos[0] / self.L, self.usv.pos[1] / self.L]
        for s, d in zip(self.sensors, self.density):
            parts += [s.queue.fill, float(d) if self.use_sem else 0.0]
        parts.append(self.step_idx / max(self.T, 1))
        return np.array(parts, dtype=np.float32)

    def in_transit_mask(self) -> np.ndarray:
        """True for AUVs whose action will actually steer motion this step (moving toward a goal)."""
        return np.array([a.hover_left == 0 and self.goal_pos(a.aid) is not None for a in self.auvs])

    # ------------------------------------------------------------------ external hooks (federated layer)
    def add_compute_energy(self, i: int, joules: float):
        self.energy.comp[i] += joules

    def add_comm_energy(self, i: int, tx_s: float = 0.0, rx_s: float = 0.0):
        en = self.cfg["energy"]
        self.energy.tx[i] += float(en["p_tx_elec_w"]) * tx_s
        self.energy.rx[i] += float(self.cfg["acoustic"]["rx_power_w"]) * rx_s
        self.comm_tx_s[i] += tx_s
        self.comm_rx_s[i] += rx_s

    def pop_window_stats(self, i: int) -> dict:
        """Take client ``i``'s window statistics for a new update (they now belong to that update, exactly once).

        Per-step quantities (collected Mbit, owned-sensor overflow) are popped from the running window.  Semantic
        quantities come from the batch ledger: only MATURE collected batches are claimed (see semantic/attribution.py);
        immature ones stay open for a later update.  Anything previously handed back by a non-committed update is
        already inside the window and is therefore included here.  The caller must record ``self.episode_id`` with
        the update so the statistics can be refused by a later episode.
        """
        out = {k: float(v[i]) for k, v in self.win.items()}
        for v in self.win.values():
            v[i] = 0.0
        c = self.book.claim(i, self.step_idx * self.dt)
        for k in ("U_mature", "V_mature", "units_del_mature", "units_coll_mature", "n_mature_batches"):
            out[k] += c[k]
        out["n_immature_batches"] = c["n_immature_batches"]          # diagnostic only (not additive, not returned)
        return out

    def _log_episode_boundary(self) -> None:
        """Record (without changing anything) the partial-window statistics and unclaimed batches that this reset discards."""
        part = {k: float(self.win[k].sum()) for k in self.WINDOW_KEYS}
        for k in self.WINDOW_KEYS:
            self.dropped_window[k] += part[k]
        opn = [e for a in range(self.M) for e in self.book.open[a].values()]
        self.episode_boundary_log.append({
            "ended_episode_id": int(self.episode_id), "steps_run": int(self.step_idx),
            "partial_window_discarded": part,
            "partial_window_coll_mbit_by_auv": [float(x) for x in self.win["coll_mbit"]],
            "partial_window_owned_loss_mbit_by_auv": [float(x) for x in self.win["owned_loss_mbit"]],
            "censored_open_batches": {"n_entries": len(opn), "V": float(sum(e.V for e in opn)), "units": float(sum(e.units for e in opn)),
                                      "U_delivered_so_far": float(sum(e.U for e in opn))}})

    def episode_boundary_diagnostics(self) -> dict:
        """Totals over all resets so far.  Descriptive: what was discarded/censored at episode boundaries, NOT a correction of it."""
        return {"n_boundaries": len(self.episode_boundary_log), "dropped_partial_window": dict(self.dropped_window),
                "censored_V_total": float(self.censored_V_total),
                "stats_refused_episode_count": int(self.stats_refused_episode),
                "stats_refused_episode_mass": dict(self.stats_refused_episode_mass),
                "current_episode_id": int(self.episode_id), "log": list(self.episode_boundary_log)}

    def return_window_stats(self, i: int, stats: dict, episode_id: int) -> bool:
        """Hand the statistics of a NON-committed update (rejected / expired) back to the client's window.

        Policy: statistics follow the model progress they describe.  A rejected/expired update did not commit its
        progress (the client's reference is unchanged, so the next delta still contains it); its statistics are
        therefore merged into the current window so the next update accounts for them -- but ONLY if the episode that
        produced them is still running.  Otherwise they are discarded (the old episode's batches no longer exist) and
        counted in ``stats_refused_episode``.  Returns True iff merged.  The caller guarantees single use per update.
        """
        if int(episode_id) != self.episode_id:
            self.stats_refused_episode += 1
            for k in self.WINDOW_KEYS:                                   # diagnostics only: mass of the statistics that were refused
                if k in stats and math.isfinite(float(stats[k])):
                    self.stats_refused_episode_mass[k] += float(stats[k])
            return False
        if any(k in stats and not math.isfinite(float(stats[k])) for k in self.WINDOW_KEYS):
            self.stats_refused_nonfinite = getattr(self, "stats_refused_nonfinite", 0) + 1     # never poison the running window
            return False
        for k in self.WINDOW_KEYS:
            if k in stats:
                self.win[k][i] += float(stats[k])
        self.stats_returned += 1
        return True

    # ------------------------------------------------------------------ accounting
    def accounting(self) -> dict:
        G = sum(s.queue.generated for s in self.sensors) * self.q
        coll = sum(s.queue.collected for s in self.sensors) * self.q
        lost = sum(s.queue.lost for s in self.sensors) * self.q
        res = sum(s.queue.occ for s in self.sensors) * self.q
        onboard = sum(a.onboard_mbit for a in self.auvs)
        deliv = self.delivered_units * self.q
        return dict(G=G, collected=coll, lost=lost, residual=res, onboard=onboard, delivered=deliv,
                    sensor_balance_err=G - (coll + lost + res), e2e_balance_err=G - (deliv + onboard + lost + res))

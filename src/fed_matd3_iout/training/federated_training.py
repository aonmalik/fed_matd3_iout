"""Builds the federation (clients + USV server + protocol) from a configuration."""
from __future__ import annotations

import numpy as np

from ..agents.actor import flatten_params, load_flat_params
from ..agents.replay_buffer import ClientReplay
from ..federated.communication_protocol import FederatedProtocol
from ..federated.federated_client import FederatedClient
from ..federated.usv_server import USVServer


def build_federation(cfg, agents, store, rng_replay: np.random.Generator, use_semantics: bool):
    f, m = cfg["federated"], cfg["matd3"]
    speeds = list(f["client_speed"])
    clients = []
    for i, ag in enumerate(agents):
        rep = ClientReplay(store, int(m["replay_capacity"]), np.random.default_rng(rng_replay.integers(1 << 31)))
        clients.append(FederatedClient(i, ag, rep, f, int(m["batch_size"]), float(speeds[i % len(speeds)])))
    if f["enabled"]:
        theta0 = flatten_params(agents[0].actor)
        for ag in agents[1:]:                                    # common initial global actor
            load_flat_params(ag.actor, theta0)
            load_flat_params(ag.actor_t, theta0)
        for c in clients:
            c.reference = theta0.clone()
        server = USVServer(theta0, f, len(agents), use_semantics)
        proto = FederatedProtocol(clients, server, f, cfg["acoustic"]["usv_link"], float(cfg["env"]["dt_s"]))
        return clients, server, proto
    return clients, None, None

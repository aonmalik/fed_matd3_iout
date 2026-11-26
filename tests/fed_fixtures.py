"""Re-export of the shared deterministic fixtures (kept so existing tests keep working)."""
from fed_matd3_iout.validation.fixtures import *   # noqa: F401,F403
from fed_matd3_iout.validation.fixtures import Harness, FakeEnv, link, make_clients, perturb, KEYS  # noqa: F401
from fed_matd3_iout.federated.communication_protocol import FederatedProtocol
from fed_matd3_iout.federated.usv_server import USVServer
from fed_matd3_iout.agents.actor import flatten_params


def build(cfg, n=2, mode="async", **fed):
    h = Harness(cfg, n, mode, copy_cfg=False, **fed)    # tests mutate/inspect the shared cfg object
    return h.clients, h.server, h.proto, h.env


def run(proto, env, t0, t1):
    for t in range(t0, t1):
        proto.step(env, float(t))
    return t1

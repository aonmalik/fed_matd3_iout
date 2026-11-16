"""Local MATD3 learning step for one client (runs whether or not the AUV is connected to the USV)."""
from __future__ import annotations

from ..federated.federated_client import FederatedClient


def local_update(client: FederatedClient, target_actors: list, device, warmup: int, batch_size: int,
                 e_grad_j: float, env) -> bool:
    """Do at most one gradient step, scaled by the client's relative compute speed. Returns True if a step ran."""
    client.credit += client.speed
    if client.credit < 1.0:
        return False
    if len(client.replay) < max(warmup, batch_size):
        client.credit = min(client.credit, 1.0)
        return False
    client.credit -= 1.0
    batch = client.replay.sample(batch_size, device)
    client.agent.update(batch, target_actors)
    client.steps_since += 1
    client.total_steps += 1
    env.add_compute_energy(client.id, e_grad_j)          # E_comp (Eq. E4), counted once per gradient step
    return True

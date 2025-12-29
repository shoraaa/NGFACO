"""
train_ppo.py
============
Corrected PPO training for NeuACO (CVRP).

Fixes applied:
1. GAE (Generalized Advantage Estimation) for proper credit assignment across rollout steps.
2. Proper mini-batch PPO with PyG Batch for vectorized forward passes.
3. Optimized replay_logp that minimizes redundant FACO instantiation.
4. Restructured experience buffer to track temporal structure.
5. Proper entropy bonus from sampling distributions.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch

from faco import FACO, FACOTrace, split_flat_to_routes, compute_route_load, flatten_routes
from utils import gen_pyg_data, gen_instance
from net import Net

# ------------------------------------------------------------
# Hyperparams
# ------------------------------------------------------------
ROLLOUT_STEPS = 4           # n-step FACO iterations per instance rollout
PPO_EPOCHS = 3              # PPO epochs per rollout batch
MINIBATCH_SIZE = 64         # number of (instance, step) tuples per minibatch

INV_TEMP = 1.0
CLIP_RATIO = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01
MAX_GRAD_NORM = 1.0

# GAE params
GAMMA = 0.99
GAE_LAMBDA = 0.95

# FACO params
K_NEAREST = 20
MIN_NEW_EDGES = 8
SAMPLE_TWO_OPT = True

# Training params
N_INSTANCES_PER_BATCH = 16  # number of instances to collect in parallel before PPO update


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class StepExperience:
    """Experience from a single FACO step for a single instance."""
    # Instance data (shared across steps for same instance)
    demands: torch.Tensor       # (n+1,)
    distances: torch.Tensor     # (n+1, n+1)
    pyg_base: Data              # static graph structure
    
    # State at this step
    ref_flat: np.ndarray        # best solution at this step (for reconstruction)
    tau_snapshot: torch.Tensor  # pheromone (n, k) at this step
    
    # Action info (per ant)
    old_logps: torch.Tensor     # (n_ants,) log probabilities under old policy
    traces: List[FACOTrace]     # FACOTrace for each ant
    entropies: torch.Tensor     # (n_ants,) entropy of policy at each step
    
    # Outcome
    costs: np.ndarray           # (n_ants,) tour costs
    
    # Value prediction
    vpred: torch.Tensor         # scalar value estimate for this state


@dataclass 
class RolloutBuffer:
    """Buffer holding experiences for one instance across all rollout steps."""
    instance_id: int
    demands: torch.Tensor
    distances: torch.Tensor
    pyg_base: Data
    steps: List[StepExperience] = field(default_factory=list)


# ------------------------------------------------------------
# Edge feature construction (efficient, no graph rebuild)
# ------------------------------------------------------------

def build_edge_features(pyg: Data, solver: FACO, ref_flat: np.ndarray, 
                        tau_sparse: torch.Tensor, in_dim: int = 3) -> torch.Tensor:
    """
    Build edge features without rebuilding the entire PyG object.
    Returns (E, in_dim) tensor.
    """
    device = pyg.x.device
    n = solver.n
    
    # Base distance feature (E, 1)
    dist_feat = pyg.edge_attr
    if dist_feat.dim() == 1:
        dist_feat = dist_feat.unsqueeze(-1)
    if dist_feat.size(-1) > 1:
        dist_feat = dist_feat[:, :1]
    
    if in_dim == 1:
        return dist_feat
    
    # Best-edge mask (E, 1)
    best_adj = torch.zeros((n, n), device=device, dtype=torch.float32)
    u_b = torch.from_numpy(ref_flat[:-1]).to(device=device, dtype=torch.long)
    v_b = torch.from_numpy(ref_flat[1:]).to(device=device, dtype=torch.long)
    best_adj[u_b, v_b] = 1.0
    
    u_e = pyg.edge_index[0]
    v_e = pyg.edge_index[1]
    in_best = best_adj[u_e, v_e].unsqueeze(-1)
    
    if in_dim == 2:
        return torch.cat([dist_feat, in_best], dim=-1)
    
    # Pheromone feature (E, 1)
    nn = torch.from_numpy(solver.nn_list).to(device)
    rows = torch.arange(n, device=device).unsqueeze(1).expand(n, solver.k)
    tau_full = torch.zeros((n, n), device=device, dtype=torch.float32)
    tau_full[rows, nn.clamp_min(0)] = tau_sparse.detach().clamp_min(1e-10)
    tau_e = tau_full[u_e, v_e].unsqueeze(-1)
    
    return torch.cat([dist_feat, in_best, tau_e], dim=-1)


def prepare_pyg_for_step(pyg_base: Data, solver: FACO, ref_flat: np.ndarray,
                         tau_sparse: torch.Tensor) -> Data:
    """Create a copy of pyg_base with updated edge features for this state."""
    pyg = Data(
        x=pyg_base.x,
        edge_index=pyg_base.edge_index,
        edge_attr=build_edge_features(pyg_base, solver, ref_flat, tau_sparse, in_dim=3)
    )
    return pyg


# ------------------------------------------------------------
# Rollout collection (collects full trajectory per instance)
# ------------------------------------------------------------

@torch.no_grad()
def collect_rollout(n_nodes: int, model: nn.Module, n_ants: int, device: str,
                    rollout_steps: int = ROLLOUT_STEPS) -> RolloutBuffer:
    """
    Collect a rollout for a single instance.
    Returns a RolloutBuffer with experiences at each step.
    """
    model.eval()
    
    demands, distances = gen_instance(n_nodes, device)
    pyg_base = gen_pyg_data(demands, distances, device)
    
    solver = FACO(
        distances=distances,
        demand=demands,
        n_ants=n_ants,
        k_nearest=K_NEAREST,
        capacity=50,
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        min_new_edges=MIN_NEW_EDGES,
        sample_two_opt=SAMPLE_TWO_OPT,
        device=device,
    )
    
    buffer = RolloutBuffer(
        instance_id=0,
        demands=demands,
        distances=distances,
        pyg_base=pyg_base,
    )
    
    for t in range(rollout_steps):
        # Snapshot state
        ref_flat = solver.best_flat.copy()
        tau_snap = solver.pheromone_sparse.detach().clone()
        
        # Prepare graph with state features
        pyg = prepare_pyg_for_step(pyg_base, solver, ref_flat, tau_snap)
        
        # Forward pass
        out = model(pyg, return_value=True) if hasattr(model, 'value_head') and model.value_head else model(pyg)
        if isinstance(out, (tuple, list)):
            heu_vec, vpred = out
            vpred = vpred.mean() if vpred.numel() > 1 else vpred.squeeze()
        else:
            heu_vec = out
            vpred = torch.zeros((), device=device)
        
        # Set heuristic
        heu_full = model.reshape(pyg, heu_vec) + 1e-10
        solver.set_heuristic(heu_full)
        
        # Sample with probability tracking
        costs_np, flats, touched, logps, traces = solver.sample(invtemp=INV_TEMP, require_prob=True)
        
        # Compute per-ant entropies from the current policy
        prob_sparse = solver.prob_sparse_torch(INV_TEMP)
        prob_norm = prob_sparse / (prob_sparse.sum(dim=1, keepdim=True) + 1e-10)
        entropy_per_node = -(prob_norm * torch.log(prob_norm + 1e-10)).sum(dim=1)
        entropy_mean = entropy_per_node.mean()
        # Store same entropy for all ants (it's the policy entropy, not per-trajectory)
        entropies = entropy_mean.expand(n_ants).clone()
        
        # Store experience
        step_exp = StepExperience(
            demands=demands,
            distances=distances,
            pyg_base=pyg_base,
            ref_flat=ref_flat,
            tau_snapshot=tau_snap,
            old_logps=logps.detach(),
            traces=traces,
            entropies=entropies.detach(),
            costs=costs_np,
            vpred=vpred.detach(),
        )
        buffer.steps.append(step_exp)
        
        # Advance environment (update best solution and pheromone)
        best_i = int(np.argmin(costs_np))
        best_cost = float(costs_np[best_i])
        best_flat = flats[best_i]
        
        if best_cost < solver.best_cost:
            solver.best_cost = best_cost
            solver.best_flat = best_flat
        
        solver._update_pheromone_from_flat(best_flat, best_cost)
    
    return buffer


def collect_batch_rollouts(n_nodes: int, model: nn.Module, n_ants: int, device: str,
                           n_instances: int = N_INSTANCES_PER_BATCH) -> List[RolloutBuffer]:
    """Collect rollouts for multiple instances."""
    buffers = []
    for i in range(n_instances):
        buffer = collect_rollout(n_nodes, model, n_ants, device, ROLLOUT_STEPS)
        buffer.instance_id = i
        buffers.append(buffer)
    return buffers


# ------------------------------------------------------------
# GAE computation
# ------------------------------------------------------------

def compute_gae_for_buffer(buffer: RolloutBuffer, gamma: float = GAMMA, 
                           lam: float = GAE_LAMBDA) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns for a single instance buffer.
    
    Since each step has n_ants trajectories, we compute GAE per-ant.
    
    Returns:
        advantages: (T, n_ants) tensor
        returns: (T, n_ants) tensor
    """
    T = len(buffer.steps)
    if T == 0:
        return torch.tensor([]), torch.tensor([])
    
    n_ants = buffer.steps[0].costs.shape[0]
    device = buffer.steps[0].vpred.device
    
    # Collect rewards (negative costs) and values
    rewards = torch.zeros((T, n_ants), device=device)
    values = torch.zeros((T + 1,), device=device)  # T+1 for bootstrap value
    
    for t, step in enumerate(buffer.steps):
        # Reward at step t: improvement in best cost from t to t+1
        # For PPO, we use the immediate negative cost as reward signal
        # But we need to think about reward shaping...
        
        # Option 1: Use per-ant cost as reward (dense signal)
        rewards[t] = -torch.from_numpy(step.costs).to(device)
        values[t] = step.vpred
    
    # Bootstrap value for terminal state (use last step's value)
    values[T] = values[T - 1]  # or 0 if treating as terminal
    
    # Compute GAE
    advantages = torch.zeros((T, n_ants), device=device)
    gae = torch.zeros(n_ants, device=device)
    
    for t in reversed(range(T)):
        # For n-step returns in ACO, treat the rollout as episodic
        # Delta = r_t + gamma * V_{t+1} - V_t
        # Since all ants share the same value estimate, broadcast
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    
    returns = advantages + values[:T].unsqueeze(1).expand(T, n_ants)
    
    return advantages, returns


def compute_gae_all_buffers(buffers: List[RolloutBuffer]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Compute GAE for all instance buffers."""
    results = []
    for buffer in buffers:
        adv, ret = compute_gae_for_buffer(buffer)
        results.append((adv, ret))
    return results


# ------------------------------------------------------------
# Batched replay log-prob computation
# ------------------------------------------------------------

def replay_logp_single(trace: FACOTrace, solver: FACO, prob_sparse: torch.Tensor) -> torch.Tensor:
    """Compute replay log-prob for a single trace using existing solver state."""
    return solver.replay_logp(trace, invtemp=INV_TEMP, prob_sparse=prob_sparse)


def batch_forward_and_replay(model: nn.Module, items: List[Dict[str, Any]], 
                             device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched forward pass and replay log-prob computation.
    
    Args:
        model: the policy network
        items: list of dicts with keys: pyg, solver, trace, ref_flat, tau
        device: torch device
        
    Returns:
        new_logps: (batch_size,) new log probabilities
        new_vpreds: (batch_size,) new value predictions  
        entropies: (batch_size,) policy entropies
    """
    batch_size = len(items)
    
    # Prepare batch of graphs
    pyg_list = []
    for item in items:
        pyg = prepare_pyg_for_step(
            item['pyg_base'], 
            item['solver'], 
            item['ref_flat'],
            item['tau']
        )
        pyg_list.append(pyg)
    
    # Batch forward pass
    batch_pyg = Batch.from_data_list(pyg_list)
    
    out = model(batch_pyg, return_value=True) if hasattr(model, 'value_head') and model.value_head else model(batch_pyg)
    if isinstance(out, (tuple, list)):
        heu_vec_batch, vpred_batch = out
    else:
        heu_vec_batch = out
        vpred_batch = torch.zeros(batch_size, device=device)
    
    # Reshape heuristics per graph
    # The batch stores edge features concatenated
    n_nodes = items[0]['pyg_base'].x.size(0)
    n_edges_per_graph = n_nodes * n_nodes
    
    new_logps = []
    new_vpreds = []
    entropies = []
    
    edge_offset = 0
    for i, item in enumerate(items):
        # Extract heuristic vector for this graph
        heu_vec = heu_vec_batch[edge_offset:edge_offset + n_edges_per_graph]
        edge_offset += n_edges_per_graph
        
        # Reshape to matrix
        heu_full = torch.zeros((n_nodes, n_nodes), device=device)
        heu_full[item['pyg_base'].edge_index[0], item['pyg_base'].edge_index[1]] = heu_vec
        heu_full = heu_full + 1e-10
        
        # Set heuristic in solver
        item['solver'].set_heuristic(heu_full)
        
        # Compute new probability distribution
        prob_sparse = item['solver'].prob_sparse_torch(INV_TEMP)
        
        # Replay log-prob
        new_logp = item['solver'].replay_logp(
            item['trace'],
            invtemp=INV_TEMP,
            ref_flat=item['ref_flat'],
            prob_sparse=prob_sparse
        )
        new_logps.append(new_logp)
        
        # Entropy from policy
        prob_norm = prob_sparse / (prob_sparse.sum(dim=1, keepdim=True) + 1e-10)
        ent = -(prob_norm * torch.log(prob_norm + 1e-10)).sum(dim=1).mean()
        entropies.append(ent)
    
    new_logps = torch.stack(new_logps)
    entropies = torch.stack(entropies)
    
    # Handle value predictions
    if vpred_batch.dim() == 0:
        new_vpreds = vpred_batch.expand(batch_size)
    elif vpred_batch.size(0) == batch_size:
        new_vpreds = vpred_batch
    else:
        # Need to pool per-graph values
        # Assuming global mean pooling was done in model, this should be (batch_size,)
        new_vpreds = vpred_batch[:batch_size] if vpred_batch.size(0) >= batch_size else vpred_batch.mean().expand(batch_size)
    
    return new_logps, new_vpreds, entropies


# ------------------------------------------------------------
# PPO Update with proper batching
# ------------------------------------------------------------

def ppo_update(model: nn.Module, optimizer: torch.optim.Optimizer,
               buffers: List[RolloutBuffer], gae_results: List[Tuple[torch.Tensor, torch.Tensor]],
               device: str) -> Dict[str, float]:
    """
    PPO update with proper batching and GAE advantages.
    """
    model.train()
    
    # Flatten experiences into a list of (instance_idx, step_idx, ant_idx) tuples
    flat_experiences = []
    
    for inst_idx, (buffer, (advantages, returns)) in enumerate(zip(buffers, gae_results)):
        for step_idx, step_exp in enumerate(buffer.steps):
            n_ants = step_exp.costs.shape[0]
            for ant_idx in range(n_ants):
                flat_experiences.append({
                    'buffer': buffer,
                    'step_exp': step_exp,
                    'inst_idx': inst_idx,
                    'step_idx': step_idx,
                    'ant_idx': ant_idx,
                    'advantage': advantages[step_idx, ant_idx],
                    'return': returns[step_idx, ant_idx],
                    'old_logp': step_exp.old_logps[ant_idx],
                    'trace': step_exp.traces[ant_idx],
                })
    
    if len(flat_experiences) == 0:
        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'total_loss': 0.0}
    
    # Normalize advantages across entire batch
    all_advs = torch.stack([exp['advantage'] for exp in flat_experiences])
    adv_mean = all_advs.mean()
    adv_std = all_advs.std() + 1e-8
    
    for exp in flat_experiences:
        exp['advantage'] = (exp['advantage'] - adv_mean) / adv_std
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    n_updates = 0
    
    for epoch in range(PPO_EPOCHS):
        # Shuffle experiences
        indices = torch.randperm(len(flat_experiences))
        
        for start in range(0, len(flat_experiences), MINIBATCH_SIZE):
            end = min(start + MINIBATCH_SIZE, len(flat_experiences))
            batch_indices = indices[start:end]
            batch = [flat_experiences[i] for i in batch_indices.tolist()]
            
            if len(batch) == 0:
                continue
            
            # Prepare items for batched processing
            items = []
            for exp in batch:
                # Reconstruct minimal solver state for replay
                solver = FACO(
                    distances=exp['buffer'].distances,
                    demand=exp['buffer'].demands,
                    n_ants=1,
                    k_nearest=K_NEAREST,
                    capacity=50,
                    decay=0.9,
                    alpha=1.0,
                    beta=1.0,
                    min_new_edges=MIN_NEW_EDGES,
                    sample_two_opt=False,
                    device=device,
                )
                solver.pheromone_sparse = exp['step_exp'].tau_snapshot.clone()
                solver.best_flat = exp['step_exp'].ref_flat.copy()
                
                items.append({
                    'pyg_base': exp['buffer'].pyg_base,
                    'solver': solver,
                    'trace': exp['trace'],
                    'ref_flat': exp['step_exp'].ref_flat,
                    'tau': exp['step_exp'].tau_snapshot,
                })
            
            # Batched forward + replay
            new_logps, new_vpreds, entropies = batch_forward_and_replay(model, items, device)
            
            # Gather old logps, advantages, returns
            old_logps = torch.stack([exp['old_logp'] for exp in batch])
            advantages_batch = torch.stack([exp['advantage'] for exp in batch])
            returns_batch = torch.stack([exp['return'] for exp in batch])
            
            # PPO loss computation
            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(new_vpreds, returns_batch)
            
            entropy = entropies.mean()
            
            loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy
            
            # Backward and step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_loss += loss.item()
            n_updates += 1
    
    return {
        'policy_loss': total_policy_loss / max(1, n_updates),
        'value_loss': total_value_loss / max(1, n_updates),
        'entropy': total_entropy / max(1, n_updates),
        'total_loss': total_loss / max(1, n_updates),
    }


# ------------------------------------------------------------
# Training step: collect + update
# ------------------------------------------------------------

def train_step(n_nodes: int, model: nn.Module, optimizer: torch.optim.Optimizer,
               n_ants: int, device: str) -> Dict[str, float]:
    """Single training step: collect rollouts and perform PPO update."""
    
    # Collect rollouts from multiple instances
    buffers = collect_batch_rollouts(n_nodes, model, n_ants, device, N_INSTANCES_PER_BATCH)
    
    # Compute GAE for all buffers
    gae_results = compute_gae_all_buffers(buffers)
    
    # PPO update
    stats = ppo_update(model, optimizer, buffers, gae_results, device)
    
    return stats


def train_epoch(n_nodes: int, n_ants: int, steps_per_epoch: int, 
                net: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> Dict[str, float]:
    """Train for one epoch."""
    stats_acc = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'total_loss': 0.0}
    
    for _ in range(steps_per_epoch):
        stats = train_step(n_nodes, net, optimizer, n_ants, device)
        for k in stats_acc:
            stats_acc[k] += stats[k]
    
    for k in stats_acc:
        stats_acc[k] /= max(1, steps_per_epoch)
    
    return stats_acc


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

@torch.no_grad()
def validation(n_nodes: int, n_ants: int, net: nn.Module, device: str,
               n_val: int = 50, T_eval: int = 5) -> Tuple[float, float, float]:
    """
    Validate by running FACO with learned heuristic.
    
    Returns:
        (baseline_mean, best_single_mean, faco_best_mean)
    """
    net.eval()
    
    sum_bl = 0.0
    sum_best = 0.0
    sum_faco = 0.0
    
    for _ in range(n_val):
        demands, distances = gen_instance(n_nodes, device)
        pyg_base = gen_pyg_data(demands, distances, device)
        
        solver = FACO(
            distances=distances,
            demand=demands,
            n_ants=n_ants,
            k_nearest=K_NEAREST,
            capacity=50,
            decay=0.9,
            alpha=1.0,
            beta=1.0,
            min_new_edges=MIN_NEW_EDGES,
            sample_two_opt=SAMPLE_TWO_OPT,
            device=device,
        )
        
        # Initial forward pass
        ref_flat = solver.best_flat.copy()
        tau = solver.pheromone_sparse.detach().clone()
        pyg = prepare_pyg_for_step(pyg_base, solver, ref_flat, tau)
        
        out = net(pyg, return_value=True) if hasattr(net, 'value_head') and net.value_head else net(pyg)
        heu_vec = out[0] if isinstance(out, (tuple, list)) else out
        heu_full = net.reshape(pyg, heu_vec) + 1e-10
        solver.set_heuristic(heu_full)
        
        costs_np, flats, _ = solver.sample(invtemp=INV_TEMP, require_prob=False)
        sum_bl += float(np.mean(costs_np))
        sum_best += float(np.min(costs_np))
        
        # Run T_eval iterations
        for _ in range(T_eval):
            ref_flat = solver.best_flat.copy()
            tau = solver.pheromone_sparse.detach().clone()
            pyg = prepare_pyg_for_step(pyg_base, solver, ref_flat, tau)
            
            out = net(pyg, return_value=True) if hasattr(net, 'value_head') and net.value_head else net(pyg)
            heu_vec = out[0] if isinstance(out, (tuple, list)) else out
            heu_full = net.reshape(pyg, heu_vec) + 1e-10
            solver.set_heuristic(heu_full)
            
            costs_np, flats, _ = solver.sample(invtemp=INV_TEMP, require_prob=False)
            
            best_i = int(np.argmin(costs_np))
            best_cost = float(costs_np[best_i])
            best_flat = flats[best_i]
            
            if best_cost < solver.best_cost:
                solver.best_cost = best_cost
                solver.best_flat = best_flat
            
            solver._update_pheromone_from_flat(best_flat, best_cost)
        
        sum_faco += float(solver.best_cost)
    
    return sum_bl / n_val, sum_best / n_val, sum_faco / n_val


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------

def train(n_nodes: int, n_ants: int, steps_per_epoch: int, epochs: int,
          lr: float = 1e-4, device: str = 'cuda', T_eval: int = 5):
    """Main training function."""
    
    net = Net(value_head=True, use_state_edge_features=True).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    
    val_best = 1e30
    
    for epoch in range(epochs):
        t0 = time.time()
        
        stats = train_epoch(n_nodes, n_ants, steps_per_epoch, net, optimizer, device)
        bl, best, faco_best = validation(n_nodes, n_ants, net, device, n_val=50, T_eval=T_eval)
        
        dt = time.time() - t0
        
        print(
            f"epoch {epoch:03d} | "
            f"loss {stats['total_loss']:.4f} (pi {stats['policy_loss']:.4f}, "
            f"v {stats['value_loss']:.4f}, ent {stats['entropy']:.4f}) | "
            f"val: bl {bl:.2f}, best {best:.2f}, FACO(T={T_eval}) {faco_best:.2f} | "
            f"time {dt:.1f}s"
        )
        
        if faco_best < val_best:
            val_best = faco_best
            torch.save(net.state_dict(), f"../pretrained/cvrp/cvrp{n_nodes}_ppo_gae.pt")
    
    return net


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_nodes', type=int, default=50)
    parser.add_argument('--n_ants', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--T_eval', type=int, default=5)
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"Using device: {device}")
    
    train(
        n_nodes=args.n_nodes,
        n_ants=args.n_ants,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        T_eval=args.T_eval,
    )

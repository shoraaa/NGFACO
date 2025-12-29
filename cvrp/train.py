

import copy
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
import wandb

from faco import FACO, FACOTrace, split_flat_to_routes, compute_route_load, flatten_routes
from utils import gen_pyg_data, gen_instance
from net import Net


# ------------------------------------------------------------
# Hyperparams
# ------------------------------------------------------------
ROLLOUT_STEPS = 4           # n-step FACO iterations per instance rollout
PPO_EPOCHS = 3              # PPO epochs per rollout batch
MINIBATCH_SIZE = 64         # number of (instance, step, ant) tuples per minibatch

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
USE_FULL_GRAPH = False      # If True, use full n² edges (for ablation); if False, use kNN

# Training params
N_INSTANCES_PER_BATCH = 16  # number of instances to collect before PPO update


# ------------------------------------------------------------
# Data structures for structured experience collection
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
    best_cost_snapshot: float   # best cost at this step (for solver reconstruction)
    
    # Action info (per ant)
    old_logps: torch.Tensor     # (n_ants,) log probabilities under old policy
    traces: List[FACOTrace]     # FACOTrace for each ant
    
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
    bootstrap_value: Optional[torch.Tensor] = None  # V(S_T) for GAE bootstrap


# ------------------------------------------------------------
# Edge feature construction (efficient, no graph rebuild)
# ------------------------------------------------------------

def build_edge_features(pyg: Data, solver: FACO, ref_flat: np.ndarray, 
                        tau_sparse: torch.Tensor, in_dim: int = 3) -> torch.Tensor:
    """Build edge features without rebuilding the entire PyG object."""
    device = pyg.x.device
    n_nodes = pyg.num_nodes  # Use graph's node count, not solver.n
    
    # Base distance feature (E, 1)
    dist_feat = pyg.edge_attr
    if dist_feat.dim() == 1:
        dist_feat = dist_feat.unsqueeze(-1)
    if dist_feat.size(-1) > 1:
        dist_feat = dist_feat[:, :1]
    
    if in_dim == 1:
        return dist_feat
    
    # Best-edge mask (E, 1) with bounds checking
    best_adj = torch.zeros((n_nodes, n_nodes), device=device, dtype=torch.float32)
    u_b = torch.from_numpy(ref_flat[:-1]).to(device=device, dtype=torch.long)
    v_b = torch.from_numpy(ref_flat[1:]).to(device=device, dtype=torch.long)
    # Guard against out-of-bounds indices
    mask = (u_b >= 0) & (u_b < n_nodes) & (v_b >= 0) & (v_b < n_nodes)
    best_adj[u_b[mask], v_b[mask]] = 1.0
    
    u_e = pyg.edge_index[0]
    v_e = pyg.edge_index[1]
    in_best = best_adj[u_e, v_e].unsqueeze(-1)
    
    if in_dim == 2:
        return torch.cat([dist_feat, in_best], dim=-1)
    
    # Pheromone feature (E, 1) - properly handle -1 padding in nn_list
    nn = torch.from_numpy(solver.nn_list).to(device)  # (solver.n, k)
    nn_rows = nn.size(0)
    nn_cols = nn.size(1)
    
    # Create full pheromone matrix, handling padding
    tau_full = torch.zeros((n_nodes, n_nodes), device=device, dtype=torch.float32)
    
    # Only fill valid entries (where nn >= 0)
    rows = torch.arange(nn_rows, device=device).unsqueeze(1).expand(nn_rows, nn_cols)
    valid_mask = nn >= 0
    
    if valid_mask.any():
        valid_rows = rows[valid_mask]
        valid_cols = nn[valid_mask]
        valid_tau = tau_sparse.detach()[valid_mask].clamp_min(1e-10)
        # Ensure indices are within bounds for the graph
        bounds_mask = (valid_rows < n_nodes) & (valid_cols < n_nodes)
        tau_full[valid_rows[bounds_mask], valid_cols[bounds_mask]] = valid_tau[bounds_mask]
    
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
    """Collect a rollout for a single instance. Returns a RolloutBuffer."""
    model.eval()
    
    demands, distances = gen_instance(n_nodes, device)
    
    # For full graph mode: use all edges; for kNN mode: use K_NEAREST edges
    # Keep solver graph consistent with pyg graph
    k_graph = None if USE_FULL_GRAPH else K_NEAREST
    k_solver = None if USE_FULL_GRAPH else K_NEAREST
    
    pyg_base = gen_pyg_data(demands, distances, device, k_nearest=k_graph)
    
    solver = FACO(
        distances=distances,
        demand=demands,
        n_ants=n_ants,
        k_nearest=k_solver,
        capacity=50,
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        min_new_edges=MIN_NEW_EDGES,
        sample_two_opt=SAMPLE_TWO_OPT,
        use_full_graph=USE_FULL_GRAPH,
        device=device,
    )
    
    # Validate node ID consistency between PyG and FACO
    assert pyg_base.num_nodes == distances.size(0), \
        f"PyG nodes ({pyg_base.num_nodes}) != distance matrix ({distances.size(0)})"
    if not USE_FULL_GRAPH:
        assert solver.nn_list.max() < pyg_base.num_nodes, \
            f"nn_list has out-of-range node ids: max={solver.nn_list.max()}, num_nodes={pyg_base.num_nodes}"
    
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
        
        # Validate output shape matches edge count
        E = pyg.edge_index.size(1)
        assert heu_vec.numel() == E, \
            f"Model output size ({heu_vec.numel()}) != edge count ({E})"
        
        # Set heuristic
        heu_full = model.reshape(pyg, heu_vec) + 1e-10
        solver.set_heuristic(heu_full)
        
        # Sample with probability tracking
        costs_np, flats, touched, logps, traces = solver.sample(invtemp=INV_TEMP, require_prob=True)
        
        # Store experience (entropy is recomputed during PPO update)
        step_exp = StepExperience(
            demands=demands,
            distances=distances,
            pyg_base=pyg_base,
            ref_flat=ref_flat,
            tau_snapshot=tau_snap,
            best_cost_snapshot=solver.best_cost,
            old_logps=logps.detach(),
            traces=traces,
            costs=costs_np,
            vpred=vpred.detach(),
        )
        buffer.steps.append(step_exp)
        
        # Advance environment
        best_i = int(np.argmin(costs_np))
        best_cost = float(costs_np[best_i])
        best_flat = flats[best_i]
        
        if best_cost < solver.best_cost:
            solver.best_cost = best_cost
            solver.best_flat = best_flat
        
        solver._update_pheromone_from_flat(best_flat, best_cost)
    
    # Compute bootstrap value V(S_T) for the final state (for GAE)
    ref_flat = solver.best_flat.copy()
    tau_snap = solver.pheromone_sparse.detach().clone()
    pyg_final = prepare_pyg_for_step(pyg_base, solver, ref_flat, tau_snap)
    
    out = model(pyg_final, return_value=True) if hasattr(model, 'value_head') and model.value_head else model(pyg_final)
    if isinstance(out, (tuple, list)):
        _, v_final = out
        v_final = v_final.mean() if v_final.numel() > 1 else v_final.squeeze()
    else:
        v_final = torch.zeros((), device=device)
    
    buffer.bootstrap_value = v_final.detach()
    
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
    
    Uses improvement-based reward: r_t = best_cost_{t-1} - min_cost_t
    This rewards the policy for finding better solutions.
    
    Returns:
        advantages: (T, n_ants) tensor
        returns: (T, n_ants) tensor
    """
    T = len(buffer.steps)
    if T == 0:
        return torch.tensor([]), torch.tensor([])  
    
    n_ants = buffer.steps[0].costs.shape[0]
    device = buffer.steps[0].vpred.device
    dtype = buffer.steps[0].vpred.dtype  # Ensure consistent dtype
    
    # Compute improvement-based rewards
    # r_t = improvement from previous best to current best found
    rewards = torch.zeros((T, n_ants), device=device, dtype=dtype)
    values = torch.zeros((T + 1,), device=device, dtype=dtype)
    
    # Track best cost across rollout for improvement reward
    prev_best_cost = float('inf')
    
    for t, step in enumerate(buffer.steps):
        costs_t = torch.from_numpy(step.costs).to(device=device, dtype=dtype)
        min_cost_t = costs_t.min().item()
        
        if t == 0:
            # First step: reward is negative of cost (no previous best)
            # Use per-ant cost relative to mean as baseline
            mean_cost = costs_t.mean()
            rewards[t] = -(costs_t - mean_cost)  # Centered reward
            prev_best_cost = min_cost_t
        else:
            # Improvement reward: positive if we found better solution
            improvement = prev_best_cost - min_cost_t
            improvement = max(0.0, improvement)  # Only positive improvements
            # Broadcast improvement to all ants, plus per-ant bonus
            mean_cost = costs_t.mean()
            rewards[t] = improvement - (costs_t - mean_cost) * 0.1  # Small per-ant signal
            prev_best_cost = min(prev_best_cost, min_cost_t)
        
        values[t] = step.vpred
    
    # Bootstrap value for terminal state (from network prediction of final state)
    if buffer.bootstrap_value is not None:
        values[T] = buffer.bootstrap_value.to(dtype=dtype)
    else:
        values[T] = values[T - 1]  # Fallback if not available
    
    # Compute GAE (backwards iteration)
    advantages = torch.zeros((T, n_ants), device=device, dtype=dtype)
    gae = torch.zeros(n_ants, device=device, dtype=dtype)
    
    for t in reversed(range(T)):
        # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
        # Values are per-state (scalar), rewards are per-ant
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    
    returns = advantages + values[:T].unsqueeze(1).expand(T, n_ants)
    
    return advantages, returns


def compute_gae_all_buffers(buffers: List[RolloutBuffer]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Compute GAE for all instance buffers."""
    return [compute_gae_for_buffer(buffer) for buffer in buffers]


# ------------------------------------------------------------
# PPO Update with proper batching and GAE
# ------------------------------------------------------------

def ppo_update(model: nn.Module, optimizer: torch.optim.Optimizer,
               buffers: List[RolloutBuffer], gae_results: List[Tuple[torch.Tensor, torch.Tensor]],
               device: str) -> Dict[str, float]:
    """
    PPO update with instance-level batching for efficiency.
    
    Instead of shuffling individual ant experiences (which requires reconstructing
    FACO solvers for each sample), we batch by instance. This allows reusing
    the same graph structure for all ants/steps of an instance.
    """
    model.train()
    
    # Get actual n_ants from data (don't hardcode)
    n_ants = buffers[0].steps[0].costs.shape[0] if buffers and buffers[0].steps else 20
    
    # Precompute normalized advantages for all experiences
    all_advantages = []
    all_returns = []
    for buffer, (advantages, returns) in zip(buffers, gae_results):
        all_advantages.append(advantages.flatten())  # (T * n_ants,)
        all_returns.append(returns.flatten())
    
    all_advs_flat = torch.cat(all_advantages)
    adv_mean = all_advs_flat.mean()
    adv_std = all_advs_flat.std() + 1e-8
    
    # Normalize advantages per buffer
    normalized_gae_results = []
    for advantages, returns in gae_results:
        norm_advs = (advantages - adv_mean) / adv_std
        normalized_gae_results.append((norm_advs, returns))
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    n_updates = 0
    
    # Create list of (buffer_idx, step_idx) pairs for shuffling
    step_indices = []
    for buf_idx, buffer in enumerate(buffers):
        for step_idx in range(len(buffer.steps)):
            step_indices.append((buf_idx, step_idx))
    
    for epoch in range(PPO_EPOCHS):
        # Shuffle at step level (all ants for a step stay together)
        perm = torch.randperm(len(step_indices))
        shuffled_indices = [step_indices[i] for i in perm.tolist()]
        
        # Process in mini-batches of steps (use actual n_ants, not hardcoded)
        n_steps_per_batch = max(1, MINIBATCH_SIZE // n_ants)
        
        for start in range(0, len(shuffled_indices), n_steps_per_batch):
            end = min(start + n_steps_per_batch, len(shuffled_indices))
            batch_step_indices = shuffled_indices[start:end]
            
            if len(batch_step_indices) == 0:
                continue
            
            # PHASE 1: Prepare all items and graphs for batched forward pass
            items = []
            for buf_idx, step_idx in batch_step_indices:
                buffer = buffers[buf_idx]
                step_exp = buffer.steps[step_idx]
                advantages, returns = normalized_gae_results[buf_idx]
                
                # Create solver for this (buffer, step) - MUST match rollout settings
                k_solver = None if USE_FULL_GRAPH else K_NEAREST
                solver = FACO(
                    distances=buffer.distances,
                    demand=buffer.demands,
                    n_ants=1,
                    k_nearest=k_solver,
                    capacity=50,
                    decay=0.9,
                    alpha=1.0,
                    beta=1.0,
                    min_new_edges=MIN_NEW_EDGES,
                    sample_two_opt=SAMPLE_TWO_OPT,  # MUST match rollout!
                    use_full_graph=USE_FULL_GRAPH,  # MUST match rollout!
                    device=device,
                )
                solver.pheromone_sparse = step_exp.tau_snapshot.clone()
                solver.best_flat = step_exp.ref_flat.copy()
                solver.best_cost = step_exp.best_cost_snapshot  # Restore best_cost for correct replay
                
                items.append({
                    'buffer': buffer,
                    'step_exp': step_exp,
                    'step_idx': step_idx,
                    'solver': solver,
                    'pyg_base': buffer.pyg_base,
                    'ref_flat': step_exp.ref_flat,
                    'tau': step_exp.tau_snapshot,
                    'advantages': advantages,
                    'returns': returns,
                })
            
            # PHASE 2: Batched forward pass
            pyg_list = [
                prepare_pyg_for_step(item['pyg_base'], item['solver'], item['ref_flat'], item['tau'])
                for item in items
            ]
            batch_pyg = Batch.from_data_list(pyg_list)
            
            out = model(batch_pyg, return_value=True) if hasattr(model, 'value_head') and model.value_head else model(batch_pyg)
            if isinstance(out, (tuple, list)):
                heu_vec_batch, vpred_batch = out
            else:
                heu_vec_batch = out
                vpred_batch = torch.zeros(len(items), device=device)
            
            # PHASE 3: Process each graph's output and replay log-probs
            # Note: Each graph may have different edge count, compute E_i per graph
            
            # Ensure heu_vec_batch is 1D for proper slicing
            if heu_vec_batch.dim() > 1:
                heu_vec_batch = heu_vec_batch.squeeze(-1)
            
            batch_new_logps = []
            batch_old_logps = []
            batch_advantages = []
            batch_returns = []
            step_entropies = []  # One entropy per step (not per ant)
            
            # For value loss: collect per-step mean returns and vpreds
            step_vpreds = []
            step_mean_returns = []
            
            edge_offset = 0
            for i, item in enumerate(items):
                # Extract heuristic vector for this graph (each graph may have different E)
                pyg_i = pyg_list[i]
                E_i = pyg_i.edge_index.size(1)
                heu_vec = heu_vec_batch[edge_offset:edge_offset + E_i]
                edge_offset += E_i
                
                # Get value prediction for this graph
                if vpred_batch.dim() == 0:
                    vpred = vpred_batch
                elif vpred_batch.size(0) > i:
                    vpred = vpred_batch[i]
                else:
                    vpred = vpred_batch.mean()
                vpred = vpred.mean() if vpred.numel() > 1 else vpred.squeeze()
                
                # Use model.reshape() for proper heuristic construction (matches rollout)
                # pyg_i already defined above when computing E_i
                heu_full = model.reshape(pyg_i, heu_vec) + 1e-10
                item['solver'].set_heuristic(heu_full)
                
                # Compute probability distribution
                prob_sparse = item['solver'].prob_sparse_torch(INV_TEMP)
                
                # Entropy (per-step, not per-ant, for proper regularization)
                prob_norm = prob_sparse / (prob_sparse.sum(dim=1, keepdim=True) + 1e-10)
                ent = -(prob_norm * torch.log(prob_norm + 1e-10)).sum(dim=1).mean()
                step_entropies.append(ent)  # One entropy per step
                
                # Process all ants for this step
                step_exp = item['step_exp']
                step_idx = item['step_idx']
                advantages = item['advantages']
                returns = item['returns']
                n_ants_step = step_exp.costs.shape[0]
                
                # Collect mean return for value loss (one target per state, not per ant)
                step_vpreds.append(vpred)
                step_mean_returns.append(returns[step_idx].mean())
                
                for ant_idx in range(n_ants_step):
                    # Replay log-prob for this ant
                    new_logp = item['solver'].replay_logp(
                        step_exp.traces[ant_idx],
                        invtemp=INV_TEMP,
                        ref_flat=item['ref_flat'],
                        prob_sparse=prob_sparse
                    )
                    
                    batch_new_logps.append(new_logp)
                    batch_old_logps.append(step_exp.old_logps[ant_idx])
                    batch_advantages.append(advantages[step_idx, ant_idx])
                    batch_returns.append(returns[step_idx, ant_idx])
            
            # Verify all edges were consumed (catches edge count misalignment)
            assert edge_offset == heu_vec_batch.numel(), \
                f"Edge offset mismatch: consumed {edge_offset}, total {heu_vec_batch.numel()}"
            
            if len(batch_new_logps) == 0:
                continue
            
            # Stack tensors
            new_logps = torch.stack(batch_new_logps)
            old_logps = torch.stack(batch_old_logps)
            advantages_batch = torch.stack(batch_advantages)
            
            # Entropy: per-step mean (not per-ant) for proper regularization
            # This makes ENTROPY_COEFF independent of n_ants
            step_entropies_tensor = torch.stack(step_entropies)
            entropy = step_entropies_tensor.mean()
            
            # Value loss: use mean return per step (not per ant) to avoid noisy gradients
            vpreds_for_value = torch.stack(step_vpreds)
            returns_for_value = torch.stack(step_mean_returns)
            
            # PPO loss
            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(vpreds_for_value, returns_for_value)
            
            loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy
            
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
    buffers = collect_batch_rollouts(n_nodes, model, n_ants, device, N_INSTANCES_PER_BATCH)
    gae_results = compute_gae_all_buffers(buffers)
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
    """Validate by running FACO with learned heuristic."""
    net.eval()
    
    sum_bl = 0.0
    sum_best = 0.0
    sum_faco = 0.0
    
    for _ in range(n_val):
        demands, distances = gen_instance(n_nodes, device)
        
        # For full graph mode: use all edges; for kNN mode: use K_NEAREST edges
        # Keep solver graph consistent with pyg graph
        k_graph = None if USE_FULL_GRAPH else K_NEAREST
        k_solver = None if USE_FULL_GRAPH else K_NEAREST
        
        pyg_base = gen_pyg_data(demands, distances, device, k_nearest=k_graph)
        
        solver = FACO(
            distances=distances,
            demand=demands,
            n_ants=n_ants,
            k_nearest=k_solver,
            capacity=50,
            decay=0.9,
            alpha=1.0,
            beta=1.0,
            min_new_edges=MIN_NEW_EDGES,
            sample_two_opt=SAMPLE_TWO_OPT,
            use_full_graph=USE_FULL_GRAPH,
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
          lr: float = 1e-4, device: str = 'cuda', T_eval: int = 5,
          wandb_project: str = 'neuaco-cvrp', wandb_run_name: str = None):
    """Main training function."""
    
    # Initialize wandb
    config = {
        'n_nodes': n_nodes,
        'n_ants': n_ants,
        'steps_per_epoch': steps_per_epoch,
        'epochs': epochs,
        'lr': lr,
        'device': device,
        'T_eval': T_eval,
        'rollout_steps': ROLLOUT_STEPS,
        'ppo_epochs': PPO_EPOCHS,
        'minibatch_size': MINIBATCH_SIZE,
        'clip_ratio': CLIP_RATIO,
        'value_coeff': VALUE_COEFF,
        'entropy_coeff': ENTROPY_COEFF,
        'gamma': GAMMA,
        'gae_lambda': GAE_LAMBDA,
        'k_nearest': K_NEAREST,
        'min_new_edges': MIN_NEW_EDGES,
        'n_instances_per_batch': N_INSTANCES_PER_BATCH,
        'use_full_graph': USE_FULL_GRAPH,
    }
    wandb.init(
        project=wandb_project,
        name=wandb_run_name or f'cvrp{n_nodes}_ppo_gae',
        config=config,
        save_code=False,
    )
    
    net = Net(value_head=True, use_state_edge_features=True).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    
    val_best = 1e30
    
    for epoch in range(epochs):
        t0 = time.time()
        
        stats = train_epoch(n_nodes, n_ants, steps_per_epoch, net, optimizer, device)
        bl, best, faco_best = validation(n_nodes, n_ants, net, device, n_val=50, T_eval=T_eval)
        
        dt = time.time() - t0
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/total_loss': stats['total_loss'],
            'train/policy_loss': stats['policy_loss'],
            'train/value_loss': stats['value_loss'],
            'train/entropy': stats['entropy'],
            'val/baseline_mean': bl,
            'val/best_single': best,
            'val/faco_best': faco_best,
            'time/epoch_seconds': dt,
        })
        
        print(
            f"epoch {epoch:03d} | "
            f"loss {stats['total_loss']:.4f} (pi {stats['policy_loss']:.4f}, "
            f"v {stats['value_loss']:.4f}, ent {stats['entropy']:.4f}) | "
            f"val: bl {bl:.2f}, best {best:.2f}, FACO(T={T_eval}) {faco_best:.2f} | "
            f"time {dt:.1f}s"
        )
        
        if faco_best < val_best:
            val_best = faco_best
            torch.save(net.state_dict(), f"./pretrained/cvrp/cvrp{n_nodes}_ppo_gae.pt")
            wandb.log({'val/best_so_far': val_best})
    
    wandb.finish()
    return net


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_nodes', type=int, default=20)
    parser.add_argument('--n_ants', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--T_eval', type=int, default=5)
    parser.add_argument('--wandb_project', type=str, default='neuaco-cvrp')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--full_graph', action='store_true', 
                        help='Use full n² graph instead of kNN (for ablation study)')
    
    args = parser.parse_args()
    
    # Update module-level flag based on argument
    if args.full_graph:
        USE_FULL_GRAPH = True
    
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"Using device: {device}")
    print(f"Graph mode: {'FULL (n² edges)' if USE_FULL_GRAPH else f'kNN (k={K_NEAREST})'}")
    
    train(
        n_nodes=args.n_nodes,
        n_ants=args.n_ants,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        T_eval=args.T_eval,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

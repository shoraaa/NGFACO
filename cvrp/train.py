

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

from faco import FACO, FACOTrace, split_flat_to_routes, _build_positions_cycle, two_opt_focused_cycle, flatten_routes
from utils import gen_pyg_data, gen_instance
from net import Net


# ------------------------------------------------------------
# Hyperparams
# ------------------------------------------------------------
ROLLOUT_STEPS = 4           # n-step FACO iterations per instance rollout
PPO_EPOCHS = 4              # PPO epochs per rollout batch
MINIBATCH_SIZE = 64         # number of (instance, step, ant) tuples per minibatch

INV_TEMP = 1.0
CLIP_RATIO = 0.1
VALUE_COEFF = 0.0          # Reduced from 0.5 to prevent value loss domination
ENTROPY_COEFF = 0; #0.001         # Reduced from 0.02 - high entropy coef keeps policy uniform
MAX_GRAD_NORM = 1.0

# GAE params
GAMMA = 0.0
GAE_LAMBDA = 0.95

# ------------------------------------------------------------
# Curriculum / warmup (later-stage state distribution)
# ------------------------------------------------------------
CURRICULUM_ENABLED = True

# Linear schedule: warmup_iters(epoch) = min(WARMUP_MAX, epoch * WARMUP_PER_EPOCH)
WARMUP_PER_EPOCH = 2
WARMUP_MAX = 60

# Keep some early-stage states in the batch for stability
CURRICULUM_EARLY_FRAC = 0.25  # 25% start from t=0 always

# Optional: adaptive "hardness" criterion using gap = (mean_cost - best_cost)/best_cost
ADAPTIVE_WARMUP_ENABLED = True
TARGET_GAP_START = 0.60   # easy (big gap) early
TARGET_GAP_END   = 0.08   # hard (small gap) late

# FACO params
K_NEAREST = 20
MIN_NEW_EDGES = 8
SAMPLE_TWO_OPT = False        # Disable internal 2-opt; use external deterministic local_two_opt
USE_FULL_GRAPH = True        # If True, use full n² edges (for ablation); if False, use kNN

# Training params
N_INSTANCES_PER_BATCH = 16    # number of instances to collect before PPO update


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
    heu_sparse_snapshot: torch.Tensor  # heuristic (n, k) at this step (for exact replay)
    best_cost_snapshot: float   # best cost at this step (for solver reconstruction)
    
    # Action info (per ant)
    old_logps: torch.Tensor     # (n_ants,) log probabilities under old policy
    traces: List[FACOTrace]     # FACOTrace for each ant
    touched: List[np.ndarray]   # touched nodes per ant for local 2-opt
    
    # Outcome (both raw and post-2opt for curriculum)
    costs_raw: np.ndarray       # (n_ants,) tour costs before local 2-opt
    costs_post: np.ndarray      # (n_ants,) tour costs after local 2-opt
    
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
# Deterministic Local 2-opt (external, for PPO consistency)
# ------------------------------------------------------------

def route_cost(route: List[int], dist: np.ndarray) -> float:
    """Compute cost of a single route (depot -> customers -> depot)."""
    if len(route) == 0:
        return 0.0
    cost = dist[0, route[0]]  # depot to first
    for i in range(len(route) - 1):
        cost += dist[route[i], route[i + 1]]
    cost += dist[route[-1], 0]  # last to depot
    return float(cost)


def flat_cost(flat: np.ndarray, dist: np.ndarray) -> float:
    """Compute total cost of a flat tour representation."""
    assert flat[0] == 0 and flat[-1] == 0
    return float(dist[flat[:-1], flat[1:]].sum())


def local_two_opt(flat: np.ndarray, touched: np.ndarray, distances: torch.Tensor,
                  window: int = 15, max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Deterministic local 2-opt focused on touched nodes.
    
    This is an "environment step" that is reproducible given (flat, touched, distances).
    Only considers intra-route 2-opt swaps near touched positions.
    
    Args:
        flat: tour in flat representation (0, c1, c2, ..., 0, c3, ..., 0)
        touched: array of touched customer node IDs
        distances: (n+1, n+1) distance matrix
        window: how many positions around touched nodes to consider
        max_iterations: max improvement iterations
        
    Returns:
        improved_flat: improved tour
        improved_cost: cost of improved tour
    """
    touched = np.asarray(touched, dtype=np.int64)
    if touched.size == 0:
        cost = flat_cost(flat, distances.cpu().numpy() if isinstance(distances, torch.Tensor) else distances)
        return flat.copy(), cost
    
    dist = distances.cpu().numpy() if isinstance(distances, torch.Tensor) else distances
    
    # Split into routes
    routes = split_flat_to_routes(flat)
    
    # Build touched set
    touched_set = set(int(x) for x in touched.tolist())
    
    # Find which routes contain touched nodes
    cust_route = {}
    for ridx, r in enumerate(routes):
        for v in r:
            cust_route[v] = ridx
    
    affected_routes = set()
    for v in touched_set:
        if v in cust_route:
            affected_routes.add(cust_route[v])
    
    # Apply 2-opt to affected routes
    for ridx in sorted(affected_routes):
        r = routes[ridx]
        if len(r) < 3:
            continue
        
        # Find positions of touched nodes in this route
        touched_positions = []
        for i, v in enumerate(r):
            if v in touched_set:
                touched_positions.append(i)
        
        if len(touched_positions) == 0:
            continue
        
        # Determine candidate positions (within window of touched)
        candidate_set = set()
        for pos in touched_positions:
            for delta in range(-window, window + 1):
                idx = pos + delta
                if 0 <= idx < len(r):
                    candidate_set.add(idx)
        candidates = sorted(candidate_set)
        
        # 2-opt improvement loop (deterministic: fixed iteration order)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for ii, i in enumerate(candidates):
                for j in candidates[ii + 2:]:  # j > i + 1
                    if j <= i + 1 or j >= len(r):
                        continue
                    
                    # Current edges: (i-1, i), (j, j+1) or depot connections
                    # After reversal: (i-1, j), (i, j+1)
                    
                    # Compute delta (gain from reversal)
                    # Before: dist[pred_i, r[i]] + dist[r[j], succ_j]
                    # After:  dist[pred_i, r[j]] + dist[r[i], succ_j]
                    pred_i = 0 if i == 0 else r[i - 1]
                    succ_j = 0 if j == len(r) - 1 else r[j + 1]
                    
                    before = dist[pred_i, r[i]] + dist[r[j], succ_j]
                    after = dist[pred_i, r[j]] + dist[r[i], succ_j]
                    
                    if after < before - 1e-9:  # Strict improvement (deterministic tie-break)
                        # Reverse segment [i, j]
                        r[i:j + 1] = r[i:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        
        routes[ridx] = r
    
    # Rebuild flat
    improved_flat = flatten_routes(routes)
    improved_cost = flat_cost(improved_flat, dist)
    
    return improved_flat, improved_cost

def local_two_opt_fast(flat: np.ndarray,
                       touched: np.ndarray,
                       dist_np: np.ndarray,              # IMPORTANT: pass numpy float32 once
                       n_nodes: int,
                       max_iterations: int = 2000) -> tuple[np.ndarray, float]:
    touched = np.asarray(touched, dtype=np.int64)
    if touched.size == 0:
        return flat.copy(), float(dist_np[flat[:-1], flat[1:]].sum())

    routes = split_flat_to_routes(flat)
    touched_set = set(int(x) for x in touched.tolist())

    # map customer->route id using array (faster than dict)
    n_nodes += 1  # include depot
    cust_route = np.full(n_nodes, -1, dtype=np.int32)
    for ridx, r in enumerate(routes):
        for v in r:
            cust_route[v] = ridx

    affected_routes = set()
    for v in touched_set:
        ridx = int(cust_route[v])
        if ridx >= 0:
            affected_routes.add(ridx)

    for ridx in affected_routes:
        r = routes[ridx]
        if len(r) < 4:
            continue

        # Tour representation: [0, c1, c2, ..., cm]
        tour = np.empty(len(r) + 1, dtype=np.uint16)
        tour[0] = 0
        tour[1:] = np.asarray(r, dtype=np.uint16)

        pos = _build_positions_cycle(tour, n_nodes)

        # fixed positions = touched nodes ± 1 (keep this SMALL like C++)
        fixed = []
        L = tour.shape[0]
        for u in r:
            if u not in touched_set:
                continue
            pu = int(pos[u])
            if pu <= 0:
                continue
            fixed.append(pu)
            if pu - 1 > 0: fixed.append(pu - 1)
            if pu + 1 < L: fixed.append(pu + 1)

        if not fixed:
            continue

        fixed = np.array(sorted(set(fixed)), dtype=np.int32)
        tour2 = two_opt_focused_cycle(dist_np.astype(np.float32),
                                      tour,
                                      fixed,
                                      fixed.shape[0],
                                      max_iterations=max_iterations)
        routes[ridx] = tour2[1:].astype(np.int64).tolist()

    improved_flat = flatten_routes(routes)
    improved_cost = float(dist_np[improved_flat[:-1], improved_flat[1:]].sum())
    return improved_flat, improved_cost


# ------------------------------------------------------------
# Curriculum warmup helpers
# ------------------------------------------------------------

def linear_target_gap(epoch: int, max_epoch: int) -> float:
    """Linearly anneal target gap from TARGET_GAP_START -> TARGET_GAP_END."""
    if max_epoch <= 1:
        return TARGET_GAP_END
    t = float(epoch) / float(max_epoch - 1)
    return (1 - t) * TARGET_GAP_START + t * TARGET_GAP_END


@torch.no_grad()
def faco_iter_with_model(
    solver: FACO,
    pyg_base: Data,
    model: nn.Module,
    distances: torch.Tensor,
    device: str,
    n_ants: int,
    n_nodes: int,
) -> float:
    """
    Run exactly ONE FACO iteration using current model heuristic + external deterministic 2-opt,
    update solver best + pheromone. Returns 'gap' (hardness proxy).
    """
    model.eval()

    ref_flat = solver.best_flat.copy()
    tau_snap = solver.pheromone_sparse.detach().clone()

    pyg = prepare_pyg_for_step(pyg_base, solver, ref_flat, tau_snap)

    out = model(pyg, return_value=True) if hasattr(model, 'value_head') and model.value_head else model(pyg)
    heu_vec = out[0] if isinstance(out, (tuple, list)) else out
    heu_full = model.reshape(pyg, heu_vec) + 1e-10
    solver.set_heuristic(heu_full)

    costs_raw, flats_raw, touched_list, _, _ = solver.sample(invtemp=INV_TEMP, require_prob=True)

    dist_np = distances.detach().cpu().numpy()
    flats_post, costs_post = [], []
    for a in range(n_ants):
        flat2, cost2 = local_two_opt_fast(flats_raw[a], touched_list[a], dist_np, n_nodes)
        flats_post.append(flat2)
        costs_post.append(cost2)
    costs_post_np = np.asarray(costs_post, dtype=np.float32)

    best_i = int(np.argmin(costs_post_np))
    best_cost = float(costs_post_np[best_i])
    best_flat = flats_post[best_i]

    if best_cost < solver.best_cost:
        solver.best_cost = best_cost
        solver.best_flat = best_flat

    solver._update_pheromone_from_flat(best_flat, best_cost)

    # "Hardness" proxy: smaller gap => harder (less room to improve)
    mean_cost = float(costs_post_np.mean())
    gap = (mean_cost - float(solver.best_cost)) / (float(solver.best_cost) + 1e-8)
    return gap


@torch.no_grad()
def warmup_solver_state(
    solver: FACO,
    pyg_base: Data,
    model: nn.Module,
    distances: torch.Tensor,
    device: str,
    n_ants: int,
    n_nodes: int,
    warmup_iters: int,
    target_gap: Optional[float] = None,
) -> None:
    """
    Advance solver forward before we start collecting PPO rollouts.
    If target_gap is set, stop early when gap <= target_gap.
    """
    if warmup_iters <= 0:
        return

    for _ in range(warmup_iters):
        gap = faco_iter_with_model(solver, pyg_base, model, distances, device, n_ants, n_nodes)
        if target_gap is not None and gap <= target_gap:
            break


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
def collect_rollout(
    n_nodes: int,
    model: nn.Module,
    n_ants: int,
    device: str,
    rollout_steps: int = ROLLOUT_STEPS,
    warmup_iters: int = 0,
    target_gap: Optional[float] = None,
) -> RolloutBuffer:
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
    
    # ---- Curriculum warmup to later-stage states ----
    if warmup_iters > 0:
        warmup_solver_state(
            solver=solver,
            pyg_base=pyg_base,
            model=model,
            distances=distances,
            device=device,
            n_ants=n_ants,
            n_nodes=n_nodes,
            warmup_iters=warmup_iters,
            target_gap=target_gap,
        )
    # -------------------------------------------------
    
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
        
        # Snapshot heuristic for exact replay
        heu_snap = solver.heuristic_sparse.detach().clone()
        
        # Sample with probability tracking (no internal 2-opt)
        costs_raw_np, flats_raw, touched_list, logps, traces = solver.sample(invtemp=INV_TEMP, require_prob=True)
        
        # Apply deterministic external local 2-opt (environment step)
        # Convert distances to numpy once (not per ant)
        dist_np = distances.detach().cpu().numpy()
        flats_post = []
        costs_post = []
        for a in range(n_ants):
            flat2, cost2 = local_two_opt_fast(flats_raw[a], touched_list[a], dist_np, n_nodes)
            flats_post.append(flat2)
            costs_post.append(cost2)
        costs_post_np = np.array(costs_post, dtype=np.float32)
        
        # Store experience with both raw and post costs
        step_exp = StepExperience(
            demands=demands,
            distances=distances,
            pyg_base=pyg_base,
            ref_flat=ref_flat,
            tau_snapshot=tau_snap,
            heu_sparse_snapshot=heu_snap,
            best_cost_snapshot=solver.best_cost,
            old_logps=logps.detach(),
            traces=traces,
            touched=touched_list,
            costs_raw=costs_raw_np,
            costs_post=costs_post_np,
            vpred=vpred.detach(),
        )
        buffer.steps.append(step_exp)
        
        # Advance environment using POST-2opt tours
        best_i = int(np.argmin(costs_post_np))
        best_cost = float(costs_post_np[best_i])
        best_flat = flats_post[best_i]
        
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


def collect_batch_rollouts(
    n_nodes: int,
    model: nn.Module,
    n_ants: int,
    device: str,
    n_instances: int = N_INSTANCES_PER_BATCH,
    warmup_iters: int = 0,
    target_gap: Optional[float] = None,
) -> List[RolloutBuffer]:
    """Collect rollouts for multiple instances."""
    buffers = []
    for i in range(n_instances):
        buffer = collect_rollout(
            n_nodes, model, n_ants, device, ROLLOUT_STEPS,
            warmup_iters=warmup_iters, target_gap=target_gap
        )
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
    
    Uses improvement-based reward: how much each ant's solution improves over
    the previous best. This preserves the global optimization signal.
    
    Returns:
        advantages: (T, n_ants) tensor
        returns: (T, n_ants) tensor
    """
    T = len(buffer.steps)
    if T == 0:
        return torch.tensor([]), torch.tensor([])  
    
    n_ants = buffer.steps[0].costs_post.shape[0]
    device = buffer.steps[0].vpred.device
    dtype = buffer.steps[0].vpred.dtype
    
    rewards = torch.zeros((T, n_ants), device=device, dtype=dtype)
    values = torch.zeros((T + 1,), device=device, dtype=dtype)
    
    for t, step in enumerate(buffer.steps):
        # Use post-2opt costs as reward signal
        costs_t = torch.from_numpy(step.costs_post).to(device=device, dtype=dtype)
        
        # Improvement-based reward: how much better than the previous best?
        # prev_best is the best cost AT THE START of this step (before sampling)
        prev_best = step.best_cost_snapshot
        
        batch_mean = costs_t.mean()
        rewards[t] = (batch_mean - costs_t) / (batch_mean + 1e-8)
        
        values[t] = step.vpred
    
    # Bootstrap value
    if buffer.bootstrap_value is not None:
        values[T] = buffer.bootstrap_value.to(dtype=dtype)
    else:
        values[T] = values[T - 1]
    
    # Compute GAE (backwards iteration)
    advantages = torch.zeros((T, n_ants), device=device, dtype=dtype)
    gae = torch.zeros(n_ants, device=device, dtype=dtype)
    
    for t in reversed(range(T)):
        delta = rewards[t] + 2 * values[t + 1] - values[t]
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
    n_ants = buffers[0].steps[0].costs_post.shape[0] if buffers and buffers[0].steps else 20
    
    # Precompute normalized advantages for all experiences
    all_advantages = []
    all_returns = []
    for buffer, (advantages, returns) in zip(buffers, gae_results):
        all_advantages.append(advantages.flatten())  # (T * n_ants,)
        all_returns.append(returns.flatten())
    
    all_advs_flat = torch.cat(all_advantages)
    adv_mean = all_advs_flat.mean()
    adv_std = all_advs_flat.std() + 1e-8
    
    # Normalize advantages per buffer (and normalize returns for stable value learning)
    normalized_gae_results = []
    all_returns_flat = torch.cat(all_returns)
    ret_mean = all_returns_flat.mean()
    ret_std = all_returns_flat.std() + 1e-8
    
    for advantages, returns in gae_results:
        norm_advs = (advantages - adv_mean) / adv_std
        normalized_gae_results.append((norm_advs, returns))
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    total_adv_mean = 0.0
    total_adv_std = 0.0
    total_ret_mean = 0.0
    total_ret_std = 0.0
    total_ratio_mean = 0.0
    total_clip_frac = 0.0
    total_approx_kl = 0.0
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
        effective_minibatch = max(MINIBATCH_SIZE, 4 * n_ants) 
        n_steps_per_batch = max(1, effective_minibatch // n_ants)
        
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
                solver.heuristic_sparse = step_exp.heu_sparse_snapshot.clone()  # Restore exact heuristic
                solver.best_flat = step_exp.ref_flat.copy()
                solver.best_cost = step_exp.best_cost_snapshot  # Restore best_cost for correct replay
                
                # DEBUG: Test reconstruction accuracy with stored heuristic (first minibatch only)
                if n_updates == 0 and len(items) == 0:
                    prob_test = solver.prob_sparse_torch(INV_TEMP)
                    for ant_idx in range(min(2, step_exp.costs_post.shape[0])):
                        lp_test, _, _ = solver.replay_logp(
                            step_exp.traces[ant_idx],
                            invtemp=INV_TEMP,
                            ref_flat=step_exp.ref_flat,
                            prob_sparse=prob_test,
                            return_entropy=True,
                            return_n_terms=True
                        )
                        diff = (lp_test.detach() - step_exp.old_logps[ant_idx].detach()).abs().item()
                        if diff > 1e-3:
                            print(f"[RECONSTRUCTION TEST] ant={ant_idx} diff={diff:.6f} (should be ~0)")
                
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
            
            # DEBUG: Test batched vs single forward (first minibatch only)
            if n_updates == 0:
                with torch.no_grad():
                    # Single forward for each graph
                    single_outs = []
                    for g in pyg_list[:min(2, len(pyg_list))]:
                        o = model(g, return_value=True) if hasattr(model, 'value_head') and model.value_head else model(g)
                        hv = o[0] if isinstance(o, (tuple, list)) else o
                        single_outs.append(hv.detach().cpu())
                    
                    # Batched forward
                    test_batch = Batch.from_data_list(pyg_list[:min(2, len(pyg_list))])
                    o = model(test_batch, return_value=True) if hasattr(model, 'value_head') and model.value_head else model(test_batch)
                    hvb = o[0] if isinstance(o, (tuple, list)) else o
                    hvb = hvb.detach().cpu().view(-1)
                    
                    # Compare outputs
                    off = 0
                    for i, (g, s) in enumerate(zip(pyg_list[:min(2, len(pyg_list))], single_outs)):
                        E = g.edge_index.size(1)
                        b = hvb[off:off+E]
                        off += E
                        s_flat = s.view(-1)
                        diff_mean = (b - s_flat).abs().mean().item()
                        diff_max = (b - s_flat).abs().max().item()
                        if diff_mean > 1e-4 or diff_max > 1e-3:
                            print(f"[BATCHING TEST] graph={i} mean_diff={diff_mean:.6f} max_diff={diff_max:.6f} (should be ~0)")
            
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
            
            # For value loss: collect per-step mean returns and vpreds (both old and new)
            step_vpreds = []
            step_old_vpreds = []
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
                assert heu_full.requires_grad, "heu_full lost gradient!"
                item['solver'].set_heuristic(heu_full)
                
                # Compute probability distribution
                prob_sparse = item['solver'].prob_sparse_torch(INV_TEMP)
                assert prob_sparse.requires_grad, "prob_sparse lost gradient!"

                # Process all ants for this step (entropy computed inside replay_logp from feasible set)
                step_exp = item['step_exp']
                step_idx = item['step_idx']
                advantages = item['advantages']
                returns = item['returns']
                n_ants_step = step_exp.costs_post.shape[0]
                
                # Collect mean return and old vpred for value loss
                step_vpreds.append(vpred)
                step_old_vpreds.append(step_exp.vpred)
                step_mean_returns.append(returns[step_idx].mean())
                
                for ant_idx in range(n_ants_step):
                    # Replay log-prob for this ant (also get entropy from feasible distribution)
                    new_logp, ant_ent, n_terms = item['solver'].replay_logp(
                        step_exp.traces[ant_idx],
                        invtemp=INV_TEMP,
                        ref_flat=item['ref_flat'],
                        prob_sparse=prob_sparse,
                        return_entropy=True,
                        return_n_terms=True
                    )
                    
                    # If replay produced no stochastic terms, neutralize for PPO (ratio=1)
                    # This prevents explosion when old_logp != 0 but new_logp == 0
                    if n_terms == 0:
                        old_logp = step_exp.old_logps[ant_idx].to(new_logp.dtype)
                        new_logp = old_logp + 0.0 * prob_sparse.sum()  # keeps graph, ratio=1
                        ant_ent = 0.0 * prob_sparse.sum()  # neutral entropy
                    
                    assert new_logp.requires_grad, f"new_logp lost gradient! (ant {ant_idx})"
                    
                    batch_new_logps.append(new_logp)
                    batch_old_logps.append(step_exp.old_logps[ant_idx])
                    batch_advantages.append(advantages[step_idx, ant_idx])
                    batch_returns.append(returns[step_idx, ant_idx])
                    step_entropies.append(ant_ent)  # Use feasible distribution entropy
            
            # Verify all edges were consumed (catches edge count misalignment)
            assert edge_offset == heu_vec_batch.numel(), \
                f"Edge offset mismatch: consumed {edge_offset}, total {heu_vec_batch.numel()}"
            
            if len(batch_new_logps) == 0:
                continue
            
            # Stack tensors
            new_logps = torch.stack(batch_new_logps)
            old_logps = torch.stack(batch_old_logps)
            advantages_batch = torch.stack(batch_advantages)
            
            # Check for bad logps (inf/nan) - these poison training
            bad_logp = torch.isinf(new_logps) | torch.isnan(new_logps)
            bad_logp_frac = bad_logp.float().mean().item()
            if bad_logp_frac > 0:
                print(f"  [WARNING] bad_logp_frac: {bad_logp_frac:.4f} ({bad_logp.sum().item()}/{new_logps.numel()})")
                # Filter out bad samples
                good_mask = ~bad_logp
                if good_mask.sum() == 0:
                    continue  # Skip this batch entirely
                new_logps = new_logps[good_mask]
                old_logps = old_logps[good_mask]
                advantages_batch = advantages_batch[good_mask]
                # Filter entropies too
                step_entropies = [e for i, e in enumerate(step_entropies) if good_mask[i]]
            
            # Entropy: mean over all ants (from feasible distribution)
            step_entropies_tensor = torch.stack(step_entropies)
            entropy = step_entropies_tensor.mean()
            
            # Value loss with clipping (PPO-style value clipping)
            vpreds_for_value = torch.stack(step_vpreds)
            old_vpreds = torch.stack(step_old_vpreds).to(vpreds_for_value.dtype)
            returns_for_value = torch.stack(step_mean_returns)
            
            # Clipped value loss
            v_clipped = old_vpreds + torch.clamp(
                vpreds_for_value - old_vpreds, -CLIP_RATIO, CLIP_RATIO
            )
            loss_unclipped = (vpreds_for_value - returns_for_value).pow(2)
            loss_clipped = (v_clipped - returns_for_value).pow(2)
            value_loss = torch.max(loss_unclipped, loss_clipped).mean()
            
            # PPO policy loss
            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute approx KL for monitoring and optional early stopping
            log_ratio = new_logps - old_logps
            ratio = torch.exp(log_ratio)
            approx_kl = (old_logps - new_logps).mean().item()

            clip_frac = ((ratio - 1).abs() > CLIP_RATIO).float().mean().item()
            
            loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # diff = (new_logps.detach() - old_logps.detach()).abs().mean().item()
            # print("replay diff", diff)

            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_loss += loss.item()
            total_adv_mean += advantages_batch.mean().item()
            total_adv_std += advantages_batch.std(unbiased=False).item()  # Use unbiased=False for robustness
            total_ret_mean += returns_for_value.mean().item()
            total_ret_std += returns_for_value.std(unbiased=False).item()  # Use unbiased=False for robustness
            total_ratio_mean += ratio.mean().item()
            total_clip_frac += clip_frac
            total_approx_kl += approx_kl
            n_updates += 1
    
    return {
        'policy_loss': total_policy_loss / max(1, n_updates),
        'value_loss': total_value_loss / max(1, n_updates),
        'entropy': total_entropy / max(1, n_updates),
        'total_loss': total_loss / max(1, n_updates),
        'adv_mean': total_adv_mean / max(1, n_updates),
        'adv_std': total_adv_std / max(1, n_updates),
        'ret_mean': total_ret_mean / max(1, n_updates),
        'ret_std': total_ret_std / max(1, n_updates),
        'ratio_mean': total_ratio_mean / max(1, n_updates),
        'clip_frac': total_clip_frac / max(1, n_updates),
        'approx_kl': total_approx_kl / max(1, n_updates),
    }



# ------------------------------------------------------------
# Training step: collect + update
# ------------------------------------------------------------

# Global step counter for wandb logging
_global_step = 0


def train_step(
    n_nodes: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_ants: int,
    device: str,
    epoch: int,
    max_epochs: int,
) -> Dict[str, float]:
    """Single training step: collect rollouts and perform PPO update."""
    global _global_step
    
    warm = 0
    tgt_gap = None
    
    if CURRICULUM_ENABLED:
        warm = min(WARMUP_MAX, epoch * WARMUP_PER_EPOCH)
        
        # Mix in early states for stability
        # (we randomize per batch; you can also do per-instance if you want)
        if np.random.rand() < CURRICULUM_EARLY_FRAC:
            warm = 0
        
        if ADAPTIVE_WARMUP_ENABLED:
            tgt_gap = linear_target_gap(epoch, max_epochs)
    
    buffers = collect_batch_rollouts(
        n_nodes, model, n_ants, device, N_INSTANCES_PER_BATCH,
        warmup_iters=warm, target_gap=tgt_gap
    )
    gae_results = compute_gae_all_buffers(buffers)
    stats = ppo_update(model, optimizer, buffers, gae_results, device)
    
    # Get current learning rate for logging
    current_lr = optimizer.param_groups[0]['lr']
    stats['lr'] = current_lr
    
    # Log per-step metrics to wandb
    wandb.log({
        'step/total_loss': stats['total_loss'],
        'step/policy_loss': stats['policy_loss'],
        'step/value_loss': stats['value_loss'],
        'step/entropy': stats['entropy'],
        'step/adv_mean': stats['adv_mean'],
        'step/adv_std': stats['adv_std'],
        'step/ret_mean': stats['ret_mean'],
        'step/ret_std': stats['ret_std'],
        'step/ratio_mean': stats['ratio_mean'],
        'step/clip_frac': stats['clip_frac'],
        'step/approx_kl': stats['approx_kl'],
        'step/lr': current_lr,
        
        # Curriculum logging
        'curriculum/warmup_iters': warm,
        'curriculum/target_gap': (tgt_gap if tgt_gap is not None else -1.0),
    }, step=_global_step)
    
    _global_step += 1
    return stats


def train_epoch(
    n_nodes: int,
    n_ants: int,
    steps_per_epoch: int,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int = 0,
    max_epochs: int = 1,
) -> Dict[str, float]:
    """Train for one epoch."""
    stats_keys = ['policy_loss', 'value_loss', 'entropy', 'total_loss', 
                  'adv_mean', 'adv_std', 'ret_mean', 'ret_std', 'ratio_mean', 'clip_frac', 'approx_kl', 'lr']
    stats_acc = {k: 0.0 for k in stats_keys}
    
    for _ in range(steps_per_epoch):
        stats = train_step(n_nodes, net, optimizer, n_ants, device, epoch=epoch, max_epochs=max_epochs)
        for k in stats_acc:
            stats_acc[k] += stats.get(k, 0.0)
    
    for k in stats_acc:
        stats_acc[k] /= max(1, steps_per_epoch)
    
    return stats_acc


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

@torch.no_grad()
def validation(n_nodes: int, n_ants: int, net: nn.Module, device: str,
               n_val: int = 100, T_eval: int = 5) -> Tuple[float, float, float]:
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
        
        # Sample with require_prob=True to get touched for external 2-opt
        costs_raw, flats_raw, touched_list, _, _ = solver.sample(invtemp=INV_TEMP, require_prob=True)
        
        # Apply external local 2-opt (consistent with training)
        dist_np = distances.detach().cpu().numpy()
        costs_post = []
        for a in range(n_ants):
            _, cost2 = local_two_opt_fast(flats_raw[a], touched_list[a], dist_np, n_nodes)
            costs_post.append(cost2)
        costs_np = np.array(costs_post, dtype=np.float32)
        
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
            
            # Sample with require_prob=True to get touched for external 2-opt
            costs_raw, flats_raw, touched_list, _, _ = solver.sample(invtemp=INV_TEMP, require_prob=True)
            
            # Apply external local 2-opt (consistent with training)
            flats_post = []
            costs_post = []
            for a in range(n_ants):
                flat2, cost2 = local_two_opt_fast(flats_raw[a], touched_list[a], dist_np, n_nodes)
                flats_post.append(flat2)
                costs_post.append(cost2)
            costs_np = np.array(costs_post, dtype=np.float32)
            
            best_i = int(np.argmin(costs_np))
            best_cost = float(costs_np[best_i])
            best_flat = flats_post[best_i]
            
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
        'sample_two_opt': SAMPLE_TWO_OPT,
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
        
        stats = train_epoch(n_nodes, n_ants, steps_per_epoch, net, optimizer, device, epoch=epoch, max_epochs=epochs)
        bl, best, faco_best = validation(n_nodes, n_ants, net, device, n_val=200, T_eval=T_eval)
        
        dt = time.time() - t0
        
        # Log to wandb (use current _global_step for consistency)
        wandb.log({
            'epoch': epoch,
            'train/total_loss': stats['total_loss'],
            'train/policy_loss': stats['policy_loss'],
            'train/value_loss': stats['value_loss'],
            'train/entropy': stats['entropy'],
            'train/approx_kl': stats['approx_kl'],
            'train/clip_frac': stats['clip_frac'],
            'val/baseline_mean': bl,
            'val/best_single': best,
            'val/faco_best': faco_best,
            'time/epoch_seconds': dt,
        }, step=_global_step)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"epoch {epoch:03d} | "
            f"loss {stats['total_loss']:.4f} (pi {stats['policy_loss']:.4f}, "
            f"v {stats['value_loss']:.4f}, ent {stats['entropy']:.4f}, kl {stats['approx_kl']:.5f}) | "
            f"val: bl {bl:.2f}, best {best:.2f}, FACO(T={T_eval}) {faco_best:.2f} | "
            f"lr {current_lr:.2e} | time {dt:.1f}s"
        )
        
        if faco_best < val_best:
            val_best = faco_best
            torch.save(net.state_dict(), f"./pretrained/cvrp/cvrp{n_nodes}_ppo_gae.pt")
            wandb.log({'val/best_so_far': val_best}, step=_global_step)
    
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
    parser.add_argument('--steps_per_epoch', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-5)
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
        # n_ants=args.n_ants,
        n_ants=args.n_nodes,  # Use n_nodes ants for CVRP
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        T_eval=args.T_eval,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

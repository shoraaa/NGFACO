"""
train.py - Neural Residual FACO Training with REINFORCE/A2C and BvB Reward

Trains a neural policy that biases FACO's deviation selection so that,
starting from a given FACO search state ("snapshot"), running FACO for the next
H iterations yields better best-so-far cost than vanilla FACO from the same snapshot.

Key features:
- Baseline-vs-Baseline (BvB) continuation reward
- Snapshot-based curriculum training
- Baseline caching for reduced variance
- Support for REINFORCE and A2C
"""

from __future__ import annotations
import os
import json
import pickle
import argparse
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data as PyGData
from copy import deepcopy
import time
import logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Local imports
from faco_tsp_unified import MFACO_TSP
from net import Net


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Stable Hashing for Reproducible Seeds
# =============================================================================

def stable_hash(key: Tuple) -> int:
    """
    Compute a stable hash that is consistent across Python runs.
    Unlike Python's hash(), this is not salted per process.
    
    Args:
        key: tuple of hashable items (e.g., (snapshot_id, H))
    
    Returns:
        Positive integer in [0, 2^31)
    """
    # Convert key to bytes
    key_str = str(key).encode('utf-8')
    # Use SHA256 for stability
    digest = hashlib.sha256(key_str).digest()
    # Take first 4 bytes and convert to int
    return int.from_bytes(digest[:4], 'big') % (2**31)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration with all hyperparameters."""
    
    # Instance generation
    n_nodes: int = 100
    n_instances: int = 1000
    
    # Snapshot generation
    warmup_iterations: int = 50       # iterations before taking snapshots
    snapshot_interval: int = 10       # take snapshot every N iterations
    max_iterations: int = 200         # max iterations per instance
    snapshots_per_instance: int = 10  # max snapshots per instance
    
    # Training
    batch_size: int = 128
    n_epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Continuation horizon
    # If `fixed_H` is not None, training/validation always uses that horizon
    # (no curriculum schedule).
    fixed_H: Optional[int] = 10
    H_min: int = 10                   # used only when fixed_H is None
    H_max: int = 50                   # used only when fixed_H is None
    H_warmup_epochs: int = 20         # used only when fixed_H is None
    
    # Residual policy
    gamma_residual: float = 0.2       # scale for residual logits
    
    # REINFORCE / A2C
    use_a2c: bool = True
    entropy_coeff: float = 0.003
    value_coeff: float = 0.5
    normalize_advantage: bool = True
    
    # Curriculum (snapshot sampling)
    # If False, sample uniformly from all snapshots (no easy/hard weighting).
    use_curriculum: bool = True
    stagnation_weight_easy: float = 0.2
    stagnation_weight_hard: float = 0.8
    curriculum_warmup_epochs: int = 5
    
    # FACO parameters
    n_ants: int = 32
    cand_list_size: int = 32
    backup_list_size: int = 32
    min_new_edges: int = 8
    decay: float = 0.9
    alpha: float = 1.0
    use_local_search: bool = True

    # Parallelism (C++ backend)
    # NOTE: Traced sampling (`require_prob=True`) is typically dominated by Python replay,
    # and parallelizing it can be slower due to overhead. Leave False unless profiled.
    parallel_traced: bool = False
    
    # Network
    node_feats: int = 2
    edge_feats: int = 1
    extra_feats: int = 4
    units: int = 64
    encoder_depth: int = 4
    scorer_hidden: int = 64
    
    # Baseline cache
    use_baseline_cache: bool = True
    baseline_seeds: int = 1           # Use 1 for training (paired with neural), >1 only for validation
    
    # Logging and checkpointing
    log_interval: int = 1
    save_interval: int = 1
    checkpoint_dir: str = "checkpoints"
    snapshot_dir: str = "snapshots"  # Base directory; actual dir will be config-specific
    
    def get_snapshot_dir(self) -> str:
        """Get configuration-specific snapshot directory."""
        return f"{self.snapshot_dir}/n{self.n_nodes}_k{self.cand_list_size}_ants{self.n_ants}"
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "neural-faco"
    wandb_run_name: Optional[str] = None
    
    # Validation during training
    val_interval: int = 1             # validate every N epochs
    val_instances: int = 100           # number of instances for validation
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    seed: int = 42


# =============================================================================
# Snapshot Data Structures
# =============================================================================

@dataclass
class Snapshot:
    """
    A saved FACO state sufficient to resume the algorithm.
    """
    snapshot_id: str
    instance_id: int
    
    # Instance data
    coords: np.ndarray                # (n, 2) node coordinates
    distances: np.ndarray             # (n, n) distance matrix
    
    # FACO state
    pheromone_sparse: np.ndarray      # (n, k) sparse pheromone
    source_route: np.ndarray          # (n,) current source solution
    source_cost: float
    best_route: np.ndarray            # (n,) best solution found
    best_cost: float
    
    # Candidate lists (can be rebuilt but faster to store)
    nn_list: np.ndarray               # (n, k)
    backup_list: np.ndarray           # (n, bl)
    
    # Metadata
    iteration: int                    # FACO iteration when snapshot was taken
    no_improve_count: int             # iterations without improvement
    tau_min: float
    tau_max: float
    
    # Hyperparameters (for exact restore)
    n_ants: int
    k: int                            # cand_list_size
    bl: int                           # backup_list_size
    min_new_edges: int
    rho: float                        # decay
    alpha: float
    
    # Optional: RNG state for exact reproducibility
    rng_state: Optional[Dict] = None


@dataclass  
class ContinuationResult:
    """Result of running FACO continuation from a snapshot."""
    best_cost: float                  # best cost after H iterations
    logp_sum: torch.Tensor            # sum of log-probs (scalar tensor)
    entropy_sum: torch.Tensor         # sum of entropies (tensor for gradient)
    initial_cost: float               # starting best cost
    improvement: float                # initial_cost - best_cost
    n_decisions: int = 0              # number of stochastic decisions

    # Per-ant rollout stats (used for per-ant credit assignment)
    logp_per_ant: Optional[torch.Tensor] = None        # (n_ants,) sum of log-probs per ant
    n_decisions_per_ant: Optional[torch.Tensor] = None # (n_ants,) stochastic decisions per ant
    ant_cost: Optional[torch.Tensor] = None            # (n_ants,) per-ant tour cost used for reward
    
    # Optional for A2C
    snapshot_features: Optional[torch.Tensor] = None


# =============================================================================
# Snapshot Dataset
# =============================================================================

class SnapshotDataset:
    """
    Dataset of FACO snapshots for training.
    Supports curriculum sampling based on stagnation level.
    """
    
    def __init__(self, snapshot_dir: str, device: str = "cpu"):
        self.snapshot_dir = Path(snapshot_dir)
        self.device = device
        self.snapshots: List[Snapshot] = []
        self.snapshot_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}  # O(1) lookup
        self.metadata: Dict[str, Dict] = {}
        
        # Curriculum bins
        self.easy_ids: List[str] = []    # low stagnation
        self.hard_ids: List[str] = []    # high stagnation
        
        self._load_all()
    
    def _load_all(self):
        """Load all snapshots from directory."""
        if not self.snapshot_dir.exists():
            logger.warning(f"Snapshot directory {self.snapshot_dir} does not exist")
            return
            
        for fpath in sorted(self.snapshot_dir.glob("*.pkl")):
            try:
                snapshot = self.load(fpath)
                idx = len(self.snapshots)
                self.snapshots.append(snapshot)
                self.snapshot_ids.append(snapshot.snapshot_id)
                self.id_to_idx[snapshot.snapshot_id] = idx  # O(1) lookup
                
                # Categorize by stagnation
                if snapshot.no_improve_count < 10:
                    self.easy_ids.append(snapshot.snapshot_id)
                else:
                    self.hard_ids.append(snapshot.snapshot_id)
                    
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")
        
        logger.info(f"Loaded {len(self.snapshots)} snapshots "
                   f"({len(self.easy_ids)} easy, {len(self.hard_ids)} hard)")
    
    def __len__(self) -> int:
        return len(self.snapshots)
    
    def load(self, path: Path) -> Snapshot:
        """Load a single snapshot from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return Snapshot(**data)
    
    def save(self, snapshot: Snapshot, path: Optional[Path] = None):
        """Save a snapshot to file."""
        if path is None:
            path = self.snapshot_dir / f"{snapshot.snapshot_id}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(asdict(snapshot), f)
    
    def get(self, snapshot_id: str) -> Snapshot:
        """Get snapshot by ID (O(1) lookup)."""
        idx = self.id_to_idx[snapshot_id]
        return self.snapshots[idx]
    
    def get_by_idx(self, idx: int) -> Snapshot:
        """Get snapshot by index."""
        return self.snapshots[idx]
    
    def sample_batch(self, batch_size: int, hard_weight: float = 0.5) -> List[str]:
        """
        Sample a batch of snapshot IDs with curriculum weighting.
        
        Args:
            batch_size: number of snapshots to sample
            hard_weight: probability of sampling from hard (stagnated) snapshots
        """
        if len(self.snapshots) == 0:
            return []
            
        # If no hard snapshots, sample from all
        if len(self.hard_ids) == 0:
            hard_weight = 0.0
        if len(self.easy_ids) == 0:
            hard_weight = 1.0
            
        ids = []
        for _ in range(batch_size):
            if np.random.random() < hard_weight and len(self.hard_ids) > 0:
                ids.append(np.random.choice(self.hard_ids))
            elif len(self.easy_ids) > 0:
                ids.append(np.random.choice(self.easy_ids))
            else:
                ids.append(np.random.choice(self.snapshot_ids))
        return ids


# =============================================================================
# Baseline Cache
# =============================================================================

class BaselineCache:
    """
    Cache for baseline continuation costs.
    Stores C_base(snapshot_id, H, seed) for fast lookup.
    
    Keys use actual seed values (not seed indices) for clarity and robustness.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        # Value is either:
        # - float (legacy): best_cost only
        # - dict: {"best_cost": float, "ant_best_cost": np.ndarray shape (A,)}
        self.cache: Dict[Tuple[str, int, int], Any] = {}
        self.cache_file = cache_file
        if cache_file and os.path.exists(cache_file):
            self._load()
    
    def _load(self):
        with open(self.cache_file, 'rb') as f:
            self.cache = pickle.load(f)
        logger.info(f"Loaded baseline cache with {len(self.cache)} entries")
    
    def save(self):
        if self.cache_file:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
    
    def get(self, snapshot_id: str, H: int, seed: int = 0) -> Optional[float]:
        """Get cached baseline best cost by actual seed (backward compatible)."""
        key = (snapshot_id, H, seed)
        val = self.cache.get(key)
        if val is None:
            return None
        if isinstance(val, (float, int)):
            return float(val)
        if isinstance(val, dict) and "best_cost" in val:
            return float(val["best_cost"])
        # Unknown format
        return None

    def get_antbest(self, snapshot_id: str, H: int, seed: int = 0) -> Optional[np.ndarray]:
        """Get cached per-ant best costs by actual seed."""
        key = (snapshot_id, H, seed)
        val = self.cache.get(key)
        if isinstance(val, dict) and "ant_best_cost" in val:
            arr = val["ant_best_cost"]
            if isinstance(arr, np.ndarray):
                return arr
            # tolerate lists
            return np.asarray(arr, dtype=np.float32)
        return None
    
    def set(self, snapshot_id: str, H: int, seed: int, value: float):
        """Store baseline best cost only (legacy helper)."""
        key = (snapshot_id, H, seed)
        self.cache[key] = value

    def set_full(self, snapshot_id: str, H: int, seed: int, best_cost: float, ant_best_cost: np.ndarray):
        """Store baseline best cost + per-ant best costs with actual seed as key."""
        key = (snapshot_id, H, seed)
        self.cache[key] = {
            "best_cost": float(best_cost),
            "ant_best_cost": np.asarray(ant_best_cost, dtype=np.float32),
        }
    
    def get_avg(self, snapshot_id: str, H: int, n_seeds: int = 3, 
                base_seed: Optional[int] = None) -> Optional[float]:
        """
        Get average baseline cost across seeds.
        
        Args:
            snapshot_id: snapshot identifier
            H: horizon
            n_seeds: number of seeds to average
            base_seed: if provided, looks for seeds base_seed, base_seed+1, ...
                      if None, looks for seeds 0, 1, 2, ... (legacy mode)
        """
        costs = []
        for seed_idx in range(n_seeds):
            seed = (base_seed + seed_idx) if base_seed is not None else seed_idx
            c = self.get(snapshot_id, H, seed)
            if c is not None:
                costs.append(c)
        if len(costs) == n_seeds:
            return np.mean(costs)
        return None

    def get_antbest_avg(self, snapshot_id: str, H: int, n_seeds: int = 3,
                        base_seed: Optional[int] = None) -> Optional[np.ndarray]:
        """Get average per-ant best costs across seeds (requires all entries)."""
        arrays = []
        for seed_idx in range(n_seeds):
            seed = (base_seed + seed_idx) if base_seed is not None else seed_idx
            arr = self.get_antbest(snapshot_id, H, seed)
            if arr is None:
                return None
            arrays.append(np.asarray(arr, dtype=np.float32))
        return np.mean(np.stack(arrays, axis=0), axis=0)

import json, hashlib
from dataclasses import asdict

def config_fingerprint(config: TrainConfig) -> str:
    d = asdict(config)
    # keep only fields that affect baseline continuation outcome
    keep = {
        "n_nodes","n_ants","cand_list_size","backup_list_size","min_new_edges",
        "decay","alpha","use_local_search",
    }
    d = {k: d[k] for k in keep}
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


# =============================================================================
# Value Network for A2C
# =============================================================================

class ValueNet(nn.Module):
    """
    Value network for A2C that predicts expected reward from snapshot features.
    """
    
    def __init__(self, input_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, input_dim) snapshot features
        Returns:
            values: (batch,) predicted values
        """
        return self.net(features).squeeze(-1)


def extract_snapshot_features(snapshot: Snapshot, config: TrainConfig) -> torch.Tensor:
    """
    Extract lightweight features from snapshot for value network.
    
    Features:
    - Normalized best cost
    - Stagnation counter (normalized)
    - Iteration fraction
    - Pheromone concentration (mean, std)
    - Source-best gap
    """
    n = len(snapshot.source_route)
    
    # Normalize costs by a rough estimate (sqrt(n) for random TSP)
    cost_scale = np.sqrt(n)
    norm_best = snapshot.best_cost / cost_scale
    norm_source = snapshot.source_cost / cost_scale
    
    # Stagnation
    norm_stagnation = min(snapshot.no_improve_count / 50.0, 1.0)
    
    # Iteration fraction
    iter_frac = min(snapshot.iteration / config.max_iterations, 1.0)
    
    # Pheromone stats
    phe = snapshot.pheromone_sparse
    phe_mean = np.mean(phe)
    phe_std = np.std(phe)
    phe_range = snapshot.tau_max - snapshot.tau_min
    norm_phe_mean = (phe_mean - snapshot.tau_min) / (phe_range + 1e-8)
    norm_phe_std = phe_std / (phe_range + 1e-8)
    
    # Source-best gap
    gap = (snapshot.source_cost - snapshot.best_cost) / (snapshot.best_cost + 1e-8)
    
    features = torch.tensor([
        norm_best,
        norm_source,
        norm_stagnation,
        iter_frac,
        norm_phe_mean,
        norm_phe_std,
        gap,
        float(n) / 200.0  # normalized problem size
    ], dtype=torch.float32)
    
    return features


# =============================================================================
# FACO Continuation Runner
# =============================================================================

def restore_faco_from_snapshot(snapshot: Snapshot, config: TrainConfig, 
                                device: str = "cpu") -> MFACO_TSP:
    """
    Restore a MFACO_TSP instance from a snapshot.
    
    Uses MFACO_TSP.from_snapshot() to ensure nn_list and backup_list are
    restored from snapshot for baseline cache consistency.
    """
    return MFACO_TSP.from_snapshot(
        snapshot=snapshot,
        use_local_search=config.use_local_search,
        enable_torch_sync=True,
        device=device,
    )


def run_baseline_continuation(snapshot: Snapshot, H: int, seed: int,
                               config: TrainConfig, use_traced: bool = True) -> float:
    """
    Run vanilla FACO continuation from snapshot for H iterations.
    Returns best cost after H iterations.
    
    Args:
        snapshot: FACO snapshot to continue from
        H: number of iterations
        seed: RNG seed for reproducibility
        config: training config
        use_traced: if True, use sequential traced sampler (require_prob=True) for
                   exact paired randomness with neural continuation. Slower but correct.
                   if False, use fast parallel sampler (not paired due to prange ordering).
    """
    # Set RNG seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    faco = restore_faco_from_snapshot(snapshot, config)
    
    # Disable torch sync for baseline - we never use torch tensors here
    # This significantly speeds up baseline cache generation
    faco.enable_torch_sync = False
    
    # Seed FACO's internal RNG for truly paired baseline/neural rollouts
    faco.seed_rng(seed)
    
    for _ in range(H):
        # Sample ants (no neural guidance)
        # use_traced=True ensures same RNG consumption as neural continuation
        costs, flats, _, _, _ = faco.sample(
            require_prob=use_traced,
            parallel_traced=(use_traced and config.parallel_traced),
        )
        
        # Find iteration best
        best_idx = int(np.argmin(costs))
        best_cost = costs[best_idx]
        best_flat = flats[best_idx]
        
        # Update pheromone (no torch sync since enable_torch_sync=False)
        faco._update_pheromone_from_flat(best_flat, best_cost)
    
    return faco.best_cost


def run_baseline_continuation_with_antbest(
    snapshot: Snapshot,
    H: int,
    seed: int,
    config: TrainConfig,
    use_traced: bool = True,
) -> Tuple[float, np.ndarray]:
    """Baseline continuation that also returns per-ant best cost over H iterations."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    faco = restore_faco_from_snapshot(snapshot, config)
    faco.enable_torch_sync = False
    faco.seed_rng(seed)

    A = int(faco.n_ants)
    ant_best_cost = np.full((A,), np.inf, dtype=np.float32)

    for _ in range(H):
        costs, flats, _, _, _ = faco.sample(
            require_prob=use_traced,
            parallel_traced=(use_traced and config.parallel_traced),
        )
        ant_best_cost = np.minimum(ant_best_cost, np.asarray(costs, dtype=np.float32))

        best_idx = int(np.argmin(costs))
        best_cost = float(costs[best_idx])
        best_flat = flats[best_idx]
        faco._update_pheromone_from_flat(best_flat, best_cost)

    return float(faco.best_cost), ant_best_cost


def build_pyg_from_faco(faco: MFACO_TSP, coords: np.ndarray, device: str = "cpu") -> PyGData:
    """
    Build PyG Data object from FACO's nn_list (after restore).
    
    Uses faco.nn_list instead of snapshot's stored list to ensure consistency
    between the solver's candidate edges and the graph used for neural encoding.
    """
    n = len(coords)
    k = faco.k
    
    # Node features: coordinates
    x = torch.from_numpy(coords).float()
    
    # Build edge index from faco.nn_list (kNN graph)
    src = []
    dst = []
    edge_attr = []
    
    for u in range(n):
        for j in range(k):
            v = faco.nn_list[u, j]
            if v >= 0:
                src.append(u)
                dst.append(v)
                edge_attr.append(faco.dist_np32[u, v])
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)
    
    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)


def compute_residual_logits(
    net: Net, 
    pyg: PyGData,
    faco: MFACO_TSP, 
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    Compute full (n, k) residual logits matrix from the neural network.
    
    Uses net.forward(pyg) to score all edges in one vectorized forward pass,
    avoiding Python loops and expensive per-node CPU->GPU copies.
    
    NOTE: This function does NOT depend on pheromone state - only on node
    embeddings and distances. So it can be called once per snapshot and
    reused across all H iterations.
    
    Args:
        net: neural network (encode will be called inside forward)
        pyg: PyG Data object with edges in row-major order (u, then j)
        faco: FACO instance for n, k dimensions
        gamma: residual scale factor
    
    Returns:
        residual_logits: (n, k) tensor WITH gradient
    """
    n = faco.n
    k = faco.k
    
    # Score all edges in one forward pass
    # net.forward() calls encode() internally and scores all edges
    edge_logits = net.forward(pyg)  # (E,) where E == n*k for complete kNN graph
    
    # Reshape to (n, k) - edges are built in row-major order by build_pyg_from_faco
    E = edge_logits.numel()
    assert E == n * k, f"Expected E=n*k={n*k}, got {E}"
    residual = gamma * edge_logits.view(n, k)

    
    return residual


def compute_entropy_from_probs(prob_sparse: torch.Tensor, faco: MFACO_TSP) -> torch.Tensor:
    """
    Compute entropy of the normalized probability distribution per node.
    
    NOTE: This is an approximation! The true per-decision entropy depends on
    the visited mask at each step, which differs per ant and per step. This
    computes entropy from the full probability matrix as a regularization proxy.
    Use a small entropy coefficient (e.g., 0.01) since this is not exact.
    
    Args:
        prob_sparse: (n, k) unnormalized weights
        faco: FACO instance for mask info
    
    Returns:
        total_entropy: scalar tensor (summed over all nodes)
    """
    n, k = prob_sparse.shape
    
    # Mask for valid candidates (nn_list >= 0)
    nn_valid = torch.from_numpy(faco.nn_list >= 0).to(prob_sparse.device)
    
    # Normalize per row (with masking)
    prob_masked = prob_sparse * nn_valid.float()
    row_sums = prob_masked.sum(dim=1, keepdim=True).clamp_min(1e-12)
    probs_norm = prob_masked / row_sums
    
    # Entropy: -sum(p * log(p)) per row, then sum over rows
    log_probs = torch.log(probs_norm.clamp_min(1e-12))
    entropy_per_row = -(probs_norm * log_probs * nn_valid.float()).sum(dim=1)
    
    return entropy_per_row.sum()

def run_neural_continuation(
    snapshot: Snapshot,
    net: Net,
    H: int,
    seed: int,
    config: TrainConfig,
    gamma: float = 0.1,
) -> ContinuationResult:
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = config.device

    # Restore FACO state
    faco = restore_faco_from_snapshot(snapshot, config, device)
    initial_cost = faco.best_cost

    # Paired RNG path (best-effort; depends on MFACO internals)
    faco.seed_rng(seed)

    # Build PyG and encode ONCE (node embeddings cached inside net)
    pyg = build_pyg_from_faco(faco, snapshot.coords, device)
    net.encode(pyg)

    # IMPORTANT: move pheromone to torch BEFORE using faco.pheromone_sparse
    faco.sync_pheromone_to_torch()

    n, k = faco.n, faco.k

    # Precompute static distance feature (n,k,1) on device.
    # If your heuristic is exactly h = 1/(dist+eps), then dist ≈ 1/h.
    # This is fast and avoids building full (n,n) dist on GPU.
    dist_nk = faco.h_sparse_torch.clamp_min(1e-12).reciprocal().clamp_max(1e6)  # (n,k)
    dist_feat = dist_nk.unsqueeze(-1)  # (n,k,1)

    # Static context scalars (you can make these dynamic if MFACO exposes no_improve_count)
    stagn0 = float(min(snapshot.no_improve_count / 50.0, 1.0))
    iter_frac0 = float(min(snapshot.iteration / config.max_iterations, 1.0))

    total_logp = torch.zeros((), device=device, dtype=torch.float32)
    A = int(faco.n_ants)
    total_logp_per_ant = torch.zeros((A,), device=device, dtype=torch.float32)
    total_entropy = torch.zeros((), device=device, dtype=torch.float32)
    n_stochastic_decisions = 0
    total_n_decisions_per_ant = torch.zeros((A,), device=device, dtype=torch.int32)
    ant_best_cost = np.full((A,), np.inf, dtype=np.float32)

    for iter_idx in range(H):
        # --- Build lightweight dynamic features from CURRENT pheromone ---
        tau = faco.pheromone_sparse.detach().clamp_min(1e-12)  # (n,k), detached!

        logtau = torch.log(tau)
        mu = logtau.mean(dim=1, keepdim=True)
        std = logtau.std(dim=1, keepdim=True).clamp_min(1e-6)
        logtau_z = (logtau - mu) / std  # (n,k)

        # Rank feature in [0,1] within each row (cheap since k is small)
        ranks = torch.argsort(torch.argsort(tau, dim=1), dim=1).float()
        tau_rank01 = ranks / (k - 1 + 1e-8)  # (n,k)

        # Broadcast small context features to (n,k,1)
        stagn_feat = torch.full((n, k, 1), stagn0, device=device, dtype=torch.float32)
        # simple "local progress" feature (optional but often helps)
        prog = min(1.0, iter_frac0 + (iter_idx / max(1, H)))
        prog_feat = torch.full((n, k, 1), float(prog), device=device, dtype=torch.float32)

        # Pack extra feats: (n,k,C)
        # C must match Net(extra_feats=C) if you use ResidualScorerWithContext
        extra = torch.cat([
            logtau_z.unsqueeze(-1),       # (n,k,1)
            tau_rank01.unsqueeze(-1),     # (n,k,1)
            stagn_feat,                  # (n,k,1)
            prog_feat,                   # (n,k,1)
        ], dim=-1)  # (n,k,4)

        # --- Vectorized residual logits using cached node embeddings ---
        # Build (n,k,d) embeddings
        node_emb = net._node_emb  # cached by net.encode(pyg)
        u_emb = node_emb.unsqueeze(1).expand(-1, k, -1)  # (n,k,d)
        v_emb = node_emb[faco.nn_torch]                  # (n,k,d)

        # Edge feature tensor (n,k,F)
        edge_feat = torch.cat([u_emb, v_emb, dist_feat, extra], dim=-1)

        # IMPORTANT: use the scorer MLP that exists in your Net
        # (nn.Linear works on (n,k,F) directly)
        residual_logits_torch = gamma * net.scorer.scorer(edge_feat).squeeze(-1)  # (n,k)

        # For sampling path (MFACO.sample expects numpy residual)
        residual_logits_np = residual_logits_torch.detach().cpu().numpy()

        # --- Sample with traced sampler ---
        costs, flats, touched_list, logps_nondiff, traces = faco.sample(
            require_prob=True,
            residual_logits=residual_logits_np,
            parallel_traced=config.parallel_traced,
        )

        # --- Differentiable logp via replay ---
        prob_sparse = faco.prob_sparse_torch(invtemp=1.0, residual_logits=residual_logits_torch)

        # Optional entropy proxy (now reflects current pheromone + residual)
        total_entropy = total_entropy + compute_entropy_from_probs(prob_sparse, faco)

        iter_logp_per_ant, iter_n_stochastic_per_ant = faco.replay_logp_batch(
            traces=traces,
            invtemp=1.0,
            prob_sparse=prob_sparse,
            residual_logits=None,  # already applied in prob_sparse
            per_ant=True,
        )

        total_logp_per_ant = total_logp_per_ant + iter_logp_per_ant
        total_n_decisions_per_ant = total_n_decisions_per_ant + iter_n_stochastic_per_ant

        total_logp = total_logp + iter_logp_per_ant.sum()
        n_stochastic_decisions += int(iter_n_stochastic_per_ant.sum().item())

        ant_best_cost = np.minimum(ant_best_cost, np.asarray(costs, dtype=np.float32))

        # --- Update pheromone ---
        best_idx = int(np.argmin(costs))
        best_cost = float(costs[best_idx])
        best_flat = flats[best_idx]
        faco._update_pheromone_from_flat(best_flat, best_cost)

    snapshot_features = extract_snapshot_features(snapshot, config).to(device)

    ant_cost_t = torch.from_numpy(ant_best_cost).to(device=device, dtype=torch.float32)

    return ContinuationResult(
        best_cost=float(faco.best_cost),
        logp_sum=total_logp,
        logp_per_ant=total_logp_per_ant,
        entropy_sum=total_entropy,
        initial_cost=float(initial_cost),
        improvement=float(initial_cost - faco.best_cost),
        n_decisions=n_stochastic_decisions,
        n_decisions_per_ant=total_n_decisions_per_ant,
        ant_cost=ant_cost_t,
        snapshot_features=snapshot_features,
    )


# =============================================================================
# Training Step
# =============================================================================

def compute_loss(
    results: List[ContinuationResult],
    baseline_costs: List[float],
    baseline_ant_costs: torch.Tensor,
    value_net: Optional[ValueNet],
    config: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute REINFORCE or A2C loss from batch results.
    
    Normalizes logp and entropy by n_decisions for stable gradients across
    different horizon lengths (H) and ant counts.
    
    Args:
        results: list of ContinuationResult from neural continuations
        baseline_costs: corresponding baseline costs C_base
        value_net: optional value network for A2C
        config: training config
    
    Returns:
        loss: scalar tensor
        stats: dict with metrics
    """
    device = config.device
    batch_size = len(results)

    # Aligned per-ant rewards using paired baseline per-ant costs.
    # baseline_ant_costs: (B,A) per-ant best tour cost over H for baseline.
    # results[i].ant_cost: (A,) per-ant best tour cost over H for neural.
    neural_ant_costs = torch.stack([r.ant_cost for r in results], dim=0)  # (B,A)
    rewards_per_ant = (baseline_ant_costs - neural_ant_costs) / (baseline_ant_costs + 1e-8)

    # Normalize per-ant logp by per-ant decision counts
    logp_normalized = torch.stack([
        results[i].logp_per_ant /
        results[i].n_decisions_per_ant.clamp_min(1).to(torch.float32)
        for i in range(batch_size)
    ], dim=0)  # (B,A)
    
    # Compute advantages
    if config.use_a2c and value_net is not None:
        # A2C: advantage = R - V(x)
        features = torch.stack([r.snapshot_features for r in results])
        values = value_net(features).view(-1)  # (B,)
        advantages = rewards_per_ant - values.detach().unsqueeze(1)

        # Value target: mean reward over ants (one value per snapshot)
        value_loss = F.mse_loss(values, rewards_per_ant.mean(dim=1))
    else:
        # REINFORCE: use normalized rewards as advantages
        values = None
        value_loss = torch.zeros((), device=device)
        advantages = rewards_per_ant
    
    # Normalize advantages
    if config.normalize_advantage and batch_size > 1:
        flat = advantages.view(-1)
        advantages = (advantages - flat.mean()) / (flat.std() + 1e-8)
    
    # Policy loss: -advantage * logp (using normalized logp)
    policy_loss = -(advantages * logp_normalized).mean()
    
    # Entropy bonus - normalize by n_decisions for consistent regularization
    entropy_normalized = torch.stack([
        r.entropy_sum / max(int(r.n_decisions_per_ant.sum().item()), 1) for r in results
    ])
    entropy_mean = entropy_normalized.mean()
    
    # Total loss (entropy_mean is a tensor, so gradient flows)
    loss = policy_loss + config.value_coeff * value_loss - config.entropy_coeff * entropy_mean
    
    # Stats
    # Compute improvement_diff: neural improvement% - baseline improvement%
    neural_improvements_pct = [(baseline_costs[i] - results[i].best_cost) / baseline_costs[i] * 100 for i in range(batch_size)]
    # Baseline improvement is 0 by definition (it's the reference)
    improvement_diff = float(np.mean(neural_improvements_pct))
    
    stats = {
        'reward_mean': float(rewards_per_ant.mean().detach()),
        'reward_std': float(rewards_per_ant.view(-1).std().detach()) if (batch_size > 1) else 0.0,
        'C_nn_mean': float(np.mean([r.best_cost for r in results])),
        'C_base_mean': float(np.mean(baseline_costs)),
        'improvement_diff': improvement_diff,
        'policy_loss': float(policy_loss.detach()),
        'value_loss': float(value_loss.detach()) if values is not None else 0.0,
        'entropy_mean': float(entropy_mean.detach()),
        'improvement_mean': float(np.mean([r.improvement for r in results])),
        'n_decisions_mean': float(np.mean([r.n_decisions for r in results])),
        'logp_mean': float(logp_normalized.mean().detach()),
    }
    
    return loss, stats


def train_step(
    batch_ids: List[str],
    dataset: SnapshotDataset,
    net: Net,
    value_net: Optional[ValueNet],
    baseline_cache: BaselineCache,
    H: int,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """
    Perform a single training step on a batch of snapshots.
    """
    step_start = time.time()
    
    device = config.device
    net.train()
    if value_net is not None:
        value_net.train()
    
    results: List[ContinuationResult] = []
    baseline_costs: List[float] = []
    baseline_ant_costs_list: List[torch.Tensor] = []
    
    # Timing accumulators
    baseline_time = 0.0
    neural_time = 0.0
    
    for snapshot_id in batch_ids:
        snapshot = dataset.get(snapshot_id)
        
        # Use stable hash for reproducible seeds across runs
        # Python's hash() is salted per process; hashlib is stable
        base_seed = stable_hash((snapshot_id, H))
        
        # Get baseline cost (from cache or compute)
        t0 = time.time()
        if config.use_baseline_cache:
            c_base = baseline_cache.get_avg(snapshot_id, H, config.baseline_seeds, base_seed=base_seed)
            ant_base = baseline_cache.get_antbest_avg(snapshot_id, H, config.baseline_seeds, base_seed=base_seed)
            if c_base is None or ant_base is None:
                # Compute and cache with consistent seeds
                for seed_idx in range(config.baseline_seeds):
                    actual_seed = base_seed + seed_idx
                    c, ant = run_baseline_continuation_with_antbest(snapshot, H, actual_seed, config)
                    baseline_cache.set_full(snapshot_id, H, actual_seed, c, ant)
                c_base = baseline_cache.get_avg(snapshot_id, H, config.baseline_seeds, base_seed=base_seed)
                ant_base = baseline_cache.get_antbest_avg(snapshot_id, H, config.baseline_seeds, base_seed=base_seed)
        else:
            c_base, ant_base = run_baseline_continuation_with_antbest(snapshot, H, base_seed, config)
        baseline_time += time.time() - t0
        
        baseline_costs.append(c_base)
        baseline_ant_costs_list.append(torch.from_numpy(np.asarray(ant_base, dtype=np.float32)).to(device))
        
        # Run neural continuation with SAME base seed for paired comparison
        # This reduces variance significantly
        t0 = time.time()
        result = run_neural_continuation(
            snapshot=snapshot,
            net=net,
            H=H,
            seed=base_seed,  # Paired randomness!
            config=config,
            gamma=config.gamma_residual,
        )
        neural_time += time.time() - t0
        results.append(result)
    
    # Compute loss
    t0 = time.time()
    baseline_ant_costs = torch.stack(baseline_ant_costs_list, dim=0)  # (B,A)
    loss, stats = compute_loss(results, baseline_costs, baseline_ant_costs, value_net, config)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
    if value_net is not None:
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), config.max_grad_norm)
    
    optimizer.step()
    backward_time = time.time() - t0
    
    stats['loss'] = float(loss.detach())
    
    # Add timing stats
    total_time = time.time() - step_start
    stats['time_total'] = total_time
    stats['time_baseline'] = baseline_time
    stats['time_neural'] = neural_time
    stats['time_backward'] = backward_time
    stats['throughput'] = len(batch_ids) / total_time  # snapshots/sec
    
    return stats


# =============================================================================
# Snapshot Generation
# =============================================================================

def generate_tsp_instance(n: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random TSP instance with n nodes in [0,1]^2."""
    if seed is not None:
        np.random.seed(seed)
    coords = np.random.rand(n, 2).astype(np.float32)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    return coords, distances.astype(np.float32)


def generate_snapshots(
    config: TrainConfig,
    output_dir: str,
    n_instances: int = None,
) -> None:
    """
    Generate snapshots by running vanilla FACO on random instances.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if n_instances is None:
        n_instances = config.n_instances
    
    logger.info(f"Generating snapshots for {n_instances} instances...")
    
    snapshot_count = 0
    
    for inst_idx in range(n_instances):
        # Generate instance
        coords, distances = generate_tsp_instance(config.n_nodes, seed=config.seed + inst_idx)
        distances_t = torch.from_numpy(distances).float()
        
        # Initialize FACO
        faco = MFACO_TSP(
            distances=distances_t,
            n_ants=config.n_ants,
            cand_list_size=config.cand_list_size,
            backup_list_size=config.backup_list_size,
            min_new_edges=config.min_new_edges,
            decay=config.decay,
            alpha=config.alpha,
            use_local_search=config.use_local_search,
            device="cpu",
        )
        
        no_improve_count = 0
        prev_best = faco.best_cost
        snapshots_taken = 0
        
        for iter_idx in range(config.max_iterations):
            # Sample and update
            costs, flats, _, _, _ = faco.sample(require_prob=False)
            best_idx = int(np.argmin(costs))
            best_cost = costs[best_idx]
            best_flat = flats[best_idx]
            faco._update_pheromone_from_flat(best_flat, best_cost)
            
            # Track improvement
            if faco.best_cost < prev_best - 1e-8:
                no_improve_count = 0
                prev_best = faco.best_cost
            else:
                no_improve_count += 1
            
            # Save snapshot if conditions met
            if (iter_idx >= config.warmup_iterations and 
                iter_idx % config.snapshot_interval == 0 and
                snapshots_taken < config.snapshots_per_instance):
                
                snapshot = Snapshot(
                    snapshot_id=f"inst{inst_idx:04d}_iter{iter_idx:04d}",
                    instance_id=inst_idx,
                    coords=coords,
                    distances=distances,
                    pheromone_sparse=faco.pheromone_sparse_np.copy(),
                    source_route=faco.source_route.copy(),
                    source_cost=faco.source_cost,
                    best_route=faco.best_route.copy(),
                    best_cost=faco.best_cost,
                    nn_list=faco.nn_list.copy(),
                    backup_list=faco.backup_list.copy(),
                    iteration=iter_idx,
                    no_improve_count=no_improve_count,
                    tau_min=faco.tau_min,
                    tau_max=faco.tau_max,
                    n_ants=faco.n_ants,
                    k=faco.k,
                    bl=faco.bl,
                    min_new_edges=faco.min_new_edges,
                    rho=faco.rho,
                    alpha=faco.alpha,
                )
                
                fpath = output_path / f"{snapshot.snapshot_id}.pkl"
                with open(fpath, 'wb') as f:
                    pickle.dump(asdict(snapshot), f)
                
                snapshots_taken += 1
                snapshot_count += 1
        
        if (inst_idx + 1) % 100 == 0:
            logger.info(f"  Generated {inst_idx + 1}/{n_instances} instances, "
                       f"{snapshot_count} total snapshots")
    
    logger.info(f"Done! Generated {snapshot_count} snapshots in {output_dir}")


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: TrainConfig):
    """Main training loop with wandb logging and periodic validation."""
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    device = config.device
    logger.info(f"Training on {device}")
    
    # Initialize wandb
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config),
        )
        logger.info(f"Wandb initialized: {wandb.run.name}")
    elif config.use_wandb and not WANDB_AVAILABLE:
        logger.warning("wandb requested but not installed. Install with: pip install wandb")
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Get configuration-specific snapshot directory
    snapshot_dir = config.get_snapshot_dir()
    logger.info(f"Using snapshot directory: {snapshot_dir}")
    
    # Load dataset
    dataset = SnapshotDataset(snapshot_dir, device)
    if len(dataset) == 0:
        logger.warning(f"No snapshots found in {snapshot_dir}. Generating snapshots first...")
        generate_snapshots(config, snapshot_dir)
        dataset = SnapshotDataset(snapshot_dir, device)
    
    logger.info(f"Dataset: {len(dataset)} snapshots")
    
    # Initialize networks
    net = Net(
        node_feats=config.node_feats,
        edge_feats=config.edge_feats,
        extra_feats=config.extra_feats,
        units=config.units,
        depth=config.encoder_depth,
        scorer_hidden=config.scorer_hidden,
    ).to(device)
    
    value_net = None
    if config.use_a2c:
        value_net = ValueNet(input_dim=8, hidden=64).to(device)
    
    # Optimizer
    params = list(net.parameters())
    if value_net is not None:
        params += list(value_net.parameters())
    optimizer = AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    
    # LR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.n_epochs)
    
    # Baseline cache
    fp = config_fingerprint(config)
    cache_file = os.path.join(config.checkpoint_dir, f"baseline_cache_{fp}.pkl")
    baseline_cache = BaselineCache(cache_file if config.use_baseline_cache else None)
    
    # Training loop
    global_step = 0
    best_improvement_diff = -float('inf')
    best_val_improvement_diff = -float('inf')
    
    for epoch in range(config.n_epochs):
        epoch_start = time.time()

        # Snapshot sampling curriculum
        if config.use_curriculum:
            hard_weight = config.stagnation_weight_easy + \
                (config.stagnation_weight_hard - config.stagnation_weight_easy) * \
                min(1.0, epoch / config.curriculum_warmup_epochs)
        else:
            hard_weight = 0.0

        # Continuation horizon (H)
        if config.fixed_H is not None:
            H = int(config.fixed_H)
        else:
            H = int(config.H_min + (config.H_max - config.H_min) *
                    min(1.0, epoch / config.H_warmup_epochs))
        
        epoch_stats = {
            'reward_mean': [], 'improvement_diff': [], 'loss': [],
            'C_nn_mean': [], 'C_base_mean': [], 'improvement_mean': [],
            'time_total': [], 'time_baseline': [], 'time_neural': [], 'time_backward': [],
            'throughput': [], 'policy_loss': [], 'value_loss': [], 'entropy_mean': [],
        }
        
        n_batches = max(1, len(dataset) // config.batch_size)
        
        for batch_idx in range(n_batches):
            # Sample batch
            if config.use_curriculum:
                batch_ids = dataset.sample_batch(config.batch_size, hard_weight)
            else:
                batch_ids = list(np.random.choice(dataset.snapshot_ids, size=config.batch_size, replace=True))
            
            # Training step
            stats = train_step(
                batch_ids=batch_ids,
                dataset=dataset,
                net=net,
                value_net=value_net,
                baseline_cache=baseline_cache,
                H=H,
                config=config,
                optimizer=optimizer,
            )
            
            # Accumulate stats
            for k in epoch_stats:
                if k in stats:
                    epoch_stats[k].append(stats[k])
            
            global_step += 1
            
            # Log periodically
            if global_step % config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1} | Step {global_step} | "
                    f"H={H} | "
                    f"R={stats['reward_mean']:.4f} | "
                    f"ImpDiff={stats['improvement_diff']:+.3f}% | "
                    f"Loss={stats['loss']:.4f} | "
                    f"Time={stats['time_total']:.2f}s"
                )
                
                # Log to wandb (step-level)
                if config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train/step': global_step,
                        'train/reward': stats['reward_mean'],
                        'train/improvement_diff': stats['improvement_diff'],
                        'train/loss': stats['loss'],
                        'train/policy_loss': stats['policy_loss'],
                        'train/value_loss': stats['value_loss'],
                        'train/entropy': stats['entropy_mean'],
                        'train/C_nn': stats['C_nn_mean'],
                        'train/C_base': stats['C_base_mean'],
                        'train/improvement': stats['improvement_mean'],
                        'timing/step_total': stats['time_total'],
                        'timing/baseline': stats['time_baseline'],
                        'timing/neural': stats['time_neural'],
                        'timing/backward': stats['time_backward'],
                        'timing/throughput': stats['throughput'],
                        'schedule/H': H,
                        'schedule/hard_weight': hard_weight,
                        'schedule/lr': scheduler.get_last_lr()[0],
                    }, step=global_step)
        
        # End of epoch
        scheduler.step()
        
        # Average stats
        avg_stats = {k: np.mean(v) if v else 0.0 for k, v in epoch_stats.items()}
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"\n=== Epoch {epoch+1}/{config.n_epochs} Summary ===\n"
            f"  Time: {epoch_time:.1f}s | H: {H} | Hard%: {hard_weight:.0%}\n"
            f"  Reward: {avg_stats['reward_mean']:.4f}\n"
            f"  Improvement Diff: {avg_stats['improvement_diff']:+.3f}%\n"
            f"  C_nn: {avg_stats['C_nn_mean']:.4f} | C_base: {avg_stats['C_base_mean']:.4f}\n"
            f"  Improvement: {avg_stats['improvement_mean']:.4f}\n"
            f"  Avg Step Time: {avg_stats['time_total']:.2f}s | Throughput: {avg_stats['throughput']:.1f} snap/s\n"
        )
        
        # Run validation periodically
        val_stats = None
        if (epoch + 1) % config.val_interval == 0:
            logger.info("Running validation...")
            val_start = time.time()
            val_stats = validate(net, config, n_instances=config.val_instances, H=H)
            val_time = time.time() - val_start
            
            logger.info(
                f"  Validation ({config.val_instances} instances, {val_time:.1f}s):\n"
                f"    Improvement Diff: {val_stats['improvement_diff']:+.3f}%\n"
                f"    Neural Cost: {val_stats['neural_cost_mean']:.4f}\n"
                f"    Baseline Cost: {val_stats['baseline_cost_mean']:.4f}\n"
                f"    Improvement: {val_stats['improvement_mean']:.4f}\n"
            )
            
            if val_stats['improvement_diff'] > best_val_improvement_diff:
                best_val_improvement_diff = val_stats['improvement_diff']
        
        # Log epoch-level metrics to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            epoch_log = {
                'epoch': epoch + 1,
                'epoch/reward_mean': avg_stats['reward_mean'],
                'epoch/improvement_diff': avg_stats['improvement_diff'],
                'epoch/loss': avg_stats['loss'],
                'epoch/C_nn': avg_stats['C_nn_mean'],
                'epoch/C_base': avg_stats['C_base_mean'],
                'epoch/improvement': avg_stats['improvement_mean'],
                'epoch/time': epoch_time,
                'epoch/throughput': avg_stats['throughput'],
            }
            if val_stats is not None:
                epoch_log.update({
                    'val/improvement_diff': val_stats['improvement_diff'],
                    'val/neural_cost': val_stats['neural_cost_mean'],
                    'val/baseline_cost': val_stats['baseline_cost_mean'],
                    'val/improvement': val_stats['improvement_mean'],
                    'val/best_improvement_diff': best_val_improvement_diff,
                })
            wandb.log(epoch_log, step=global_step)
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0 or avg_stats['improvement_diff'] > best_improvement_diff:
            if avg_stats['improvement_diff'] > best_improvement_diff:
                best_improvement_diff = avg_stats['improvement_diff']
                ckpt_name = f"best_model.pt"
            else:
                ckpt_name = f"epoch_{epoch+1}.pt"
            
            ckpt_path = os.path.join(config.checkpoint_dir, ckpt_name)
            torch.save({
                'epoch': epoch + 1,
                'net_state_dict': net.state_dict(),
                'value_net_state_dict': value_net.state_dict() if value_net else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'config': asdict(config),
                'best_improvement_diff': best_improvement_diff,
                'best_val_improvement_diff': best_val_improvement_diff,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
        
        # Save baseline cache periodically
        if config.use_baseline_cache and (epoch + 1) % 10 == 0:
            baseline_cache.save()
    
    # Final save
    if config.use_baseline_cache:
        baseline_cache.save()
    
    # Finish wandb run
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    logger.info("Training complete!")


# =============================================================================
# Validation
# =============================================================================

def validate(
    net: Net,
    config: TrainConfig,
    n_instances: int = 20,
    H: int = 50,
) -> Dict[str, float]:
    """
    Validate neural policy against baseline on fresh instances.
    """
    device = config.device
    net.eval()
    
    neural_costs = []
    baseline_costs = []
    improvements = []
    
    for i in range(n_instances):
        # Generate instance
        coords, distances = generate_tsp_instance(config.n_nodes, seed=10000 + i)
        distances_t = torch.from_numpy(distances).float()
        
        # Run baseline FACO to get a snapshot
        faco = MFACO_TSP(
            distances=distances_t,
            n_ants=config.n_ants,
            cand_list_size=config.cand_list_size,
            backup_list_size=config.backup_list_size,
            min_new_edges=config.min_new_edges,
            decay=config.decay,
            alpha=config.alpha,
            use_local_search=config.use_local_search,
            device="cpu",
        )
        
        # Warmup
        for _ in range(config.warmup_iterations):
            costs, flats, _, _, _ = faco.sample(require_prob=False)
            best_idx = int(np.argmin(costs))
            faco._update_pheromone_from_flat(flats[best_idx], costs[best_idx])
        
        # Create snapshot
        snapshot = Snapshot(
            snapshot_id=f"val_{i}",
            instance_id=i,
            coords=coords,
            distances=distances,
            pheromone_sparse=faco.pheromone_sparse_np.copy(),
            source_route=faco.source_route.copy(),
            source_cost=faco.source_cost,
            best_route=faco.best_route.copy(),
            best_cost=faco.best_cost,
            nn_list=faco.nn_list.copy(),
            backup_list=faco.backup_list.copy(),
            iteration=config.warmup_iterations,
            no_improve_count=0,
            tau_min=faco.tau_min,
            tau_max=faco.tau_max,
            n_ants=faco.n_ants,
            k=faco.k,
            bl=faco.bl,
            min_new_edges=faco.min_new_edges,
            rho=faco.rho,
            alpha=faco.alpha,
        )
        
        # Run baseline continuation
        c_base = run_baseline_continuation(snapshot, H, seed=i, config=config)
        
        # Run neural continuation (with no_grad since this is inference)
        # Temporarily enable no_grad mode for the whole forward pass
        net.eval()
        with torch.no_grad():
            # For inference, we need a modified version that doesn't track gradients
            result = run_neural_continuation(snapshot, net, H, seed=i, config=config,
                                            gamma=config.gamma_residual)
        
        neural_costs.append(result.best_cost)
        baseline_costs.append(c_base)
        improvements.append(c_base - result.best_cost)
    
    # Compute improvement_diff: neural improvement% - baseline improvement%
    # Neural improvement % relative to baseline continuation cost
    neural_improvements_pct = [(baseline_costs[i] - neural_costs[i]) / baseline_costs[i] * 100 for i in range(n_instances)]
    improvement_diff = np.mean(neural_improvements_pct)
    
    return {
        'neural_cost_mean': np.mean(neural_costs),
        'baseline_cost_mean': np.mean(baseline_costs),
        'improvement_mean': np.mean(improvements),
        'improvement_diff': improvement_diff,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Neural Residual FACO")
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'generate', 'validate'],
                       help='Mode: train, generate snapshots, or validate')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for resume/validate')
    
    # Override config options
    parser.add_argument('--n_nodes', type=int, default=None)
    parser.add_argument('--n_instances', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--snapshot_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)

    # Curriculum / horizon overrides
    parser.add_argument('--no_curriculum', action='store_true',
                       help='Disable curriculum snapshot sampling (uniform over all snapshots)')
    parser.add_argument('--use_curriculum', action='store_true', default=None,
                       help='Enable curriculum snapshot sampling')
    parser.add_argument('--H', type=int, default=None,
                       help='Fix continuation horizon H (disables H schedule)')
    
    # Wandb options
    parser.add_argument('--use_wandb', action='store_true', default=None,
                       help='Enable wandb logging')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    
    # Validation options
    parser.add_argument('--val_interval', type=int, default=None,
                       help='Validation interval (epochs)')
    parser.add_argument('--val_instances', type=int, default=None,
                       help='Number of validation instances')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TrainConfig(**config_dict)
    else:
        config = TrainConfig()
    
    # Override with CLI args
    for key in ['n_nodes', 'n_instances', 'batch_size', 'n_epochs', 'lr', 
                'device', 'snapshot_dir', 'checkpoint_dir', 'wandb_project',
                'wandb_run_name', 'val_interval', 'val_instances']:
        val = getattr(args, key)
        if val is not None:
            setattr(config, key, val)
    
    # Handle --use_wandb / --no_wandb flags
    if args.no_wandb:
        config.use_wandb = False
    elif args.use_wandb:
        config.use_wandb = True

    # Handle curriculum flags
    if args.no_curriculum:
        config.use_curriculum = False
    elif args.use_curriculum:
        config.use_curriculum = True

    if args.H is not None:
        config.fixed_H = int(args.H)
    
    if args.mode == 'generate':
        snapshot_dir = config.get_snapshot_dir()
        logger.info(f"Generating snapshots in: {snapshot_dir}")
        generate_snapshots(config, snapshot_dir)
    
    elif args.mode == 'train':
        train(config)
    
    elif args.mode == 'validate':
        if args.checkpoint is None:
            raise ValueError("Must provide --checkpoint for validation")
        
        # Load checkpoint
        ckpt = torch.load(args.checkpoint, map_location=config.device)
        
        net = Net(
            node_feats=config.node_feats,
            edge_feats=config.edge_feats,
            extra_feats=config.extra_feats,
            units=config.units,
            depth=config.encoder_depth,
            scorer_hidden=config.scorer_hidden,
        ).to(config.device)
        net.load_state_dict(ckpt['net_state_dict'])
        
        stats = validate(net, config)
        logger.info(f"\nValidation Results:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

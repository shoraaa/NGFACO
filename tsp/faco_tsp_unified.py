"""
MFACO TSP Solver - Unified interface with C++ backend and Python fallback.

This module provides a unified MFACO_TSP class that uses the fast C++ backend
when available, or falls back to the pure Python implementation.

Usage:
    from faco_tsp_unified import MFACO_TSP, MFACOTrace
    
    # Create solver (automatically uses C++ if available)
    faco = MFACO_TSP(distances, n_ants=20, ...)
    
    # Create from snapshot for training
    faco = MFACO_TSP.from_snapshot(snapshot, config)
    
    # Sample solutions
    costs, flats, touched, logps, traces = faco.sample(require_prob=True)
    
    # Update pheromone
    faco._update_pheromone_from_flat(best_flat, best_cost)
"""

from __future__ import annotations
import warnings
from typing import Optional, Tuple, List, Any
import numpy as np
import torch

# Try to import C++ backend
try:
    from faco_cpp import MFACO_TSP as MFACO_TSP_CPP, MFACOTrace as MFACOTrace_CPP
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False
    warnings.warn(
        "faco_cpp not found, using pure Python backend. "
        "Build C++ extension with: cd tsp/faco && pip install -e ."
    )

# Import Python fallback
from faco_tsp import (
    MFACO_TSP as MFACO_TSP_PY, 
    MFACOTrace as MFACOTrace_PY,
    build_nn_and_backup_lists,
    tsp_cost_from_cycle,
)

# Re-export utility functions
__all__ = [
    'MFACO_TSP',
    'MFACOTrace', 
    'has_cpp_backend',
    'set_faco_cpp_threads',
    'build_nn_and_backup_lists',
    'tsp_cost_from_cycle',
]


def set_faco_cpp_threads(n_threads: int) -> None:
    """Set OpenMP thread count for the C++ backend (process-global).

    If you built `faco_cpp` with OpenMP, this controls how many threads the
    *parallel* sampling path uses. The traced path (`require_prob=True`) is
    intentionally single-threaded for determinism.
    """
    if not HAS_CPP_BACKEND:
        raise RuntimeError("faco_cpp backend not available; cannot set threads")
    import faco_cpp
    faco_cpp.set_num_threads(int(n_threads))


class MFACOTrace:
    """
    Unified trace class that wraps either C++ or Python trace.
    
    For Python compatibility, provides the same interface as faco_tsp.MFACOTrace.
    """
    
    def __init__(self, trace):
        """
        Args:
            trace: Either MFACOTrace_CPP batch trace or MFACOTrace_PY
        """
        self._trace = trace
        self._is_cpp = HAS_CPP_BACKEND and isinstance(trace, MFACOTrace_CPP) if HAS_CPP_BACKEND else False
    
    @property
    def start_node(self) -> int:
        if self._is_cpp:
            # For CPP batch trace, this should be accessed per-ant
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.start_node
    
    @property
    def curr_nodes(self) -> List[int]:
        if self._is_cpp:
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.curr_nodes
    
    @property
    def chosen_nodes(self) -> List[int]:
        if self._is_cpp:
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.chosen_nodes
    
    @property
    def is_stochastic(self) -> List[bool]:
        if self._is_cpp:
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.is_stochastic
    
    @property
    def used_uniform_fallback(self) -> List[bool]:
        if self._is_cpp:
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.used_uniform_fallback


class MFACO_TSP:
    """
    Unified MFACO TSP solver with C++ backend and Python fallback.
    
    Provides the same API as faco_tsp.MFACO_TSP but uses the fast C++
    implementation when available.
    """
    
    def __init__(
        self,
        distances: torch.Tensor,
        n_ants: int,
        cand_list_size: int = 32,
        backup_list_size: int = 32,
        min_new_edges: int = 8,
        decay: float = 0.9,
        alpha: float = 1.0,
        p_best: float = 0.05,
        gbest_as_source_prob: float = 1.0,
        use_local_search: bool = True,
        random_mode: bool = False,
        enable_torch_sync: bool = True,
        device: str = "cuda",
        use_cpp: bool = True,  # New option to force Python backend
        disable_heuristic: bool = False,
    ):
        """
        Initialize MFACO_TSP solver.
        
        Args:
            distances: (n, n) distance matrix
            n_ants: number of ants
            cand_list_size: candidate list size (k)
            backup_list_size: backup list size
            min_new_edges: minimum new edges before copying from source
            decay: pheromone decay rate (rho)
            alpha: pheromone exponent
            p_best: probability best for tau limits
            gbest_as_source_prob: probability to use global best as source
            use_local_search: whether to apply 2-opt local search
            random_mode: if True, use uniform random instead of roulette
            enable_torch_sync: if False, skip torch pheromone sync (faster baseline)
            device: torch device (only used by Python backend)
            use_cpp: if False, force Python backend even if C++ available
        """
        self.device = device
        self.disable_heuristic = bool(disable_heuristic)

        # C++ backend currently bakes heuristic into sampling (eta=1/dist).
        # If heuristic is disabled, force Python backend to ensure correctness.
        if self.disable_heuristic and use_cpp and HAS_CPP_BACKEND:
            warnings.warn(
                "disable_heuristic=True forces Python backend; C++ backend does not support disabling heuristic yet."
            )
            use_cpp = False

        self._use_cpp = use_cpp and HAS_CPP_BACKEND
        
        # Convert distances to numpy for C++ backend
        if isinstance(distances, torch.Tensor):
            dist_np = distances.cpu().numpy().astype(np.float32)
        else:
            dist_np = np.asarray(distances, dtype=np.float32)
        
        self._dist_np = dist_np
        
        if self._use_cpp:
            # Use C++ backend
            self._cpp = MFACO_TSP_CPP(
                dist_np,
                n_ants,
                cand_list_size,
                backup_list_size,
                min_new_edges,
                decay,
                alpha,
                p_best,
                use_local_search,
                random_mode,
                "cpu"
            )
            self._py = None
            
            # For compatibility, keep torch tensors for replay_logp (use private names)
            self._distances = distances.to(device) if isinstance(distances, torch.Tensor) else torch.from_numpy(dist_np).to(device)
            self._pheromone_sparse = torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(device)
            self._h_sparse_torch = torch.from_numpy(self._cpp.heuristic_sparse_np.copy()).to(device)
            self._nn_torch = torch.from_numpy(self._cpp.nn_list.copy()).to(device).long()
            self._enable_torch_sync = enable_torch_sync
        else:
            # Use Python backend
            self._cpp = None
            self._py = MFACO_TSP_PY(
                distances if isinstance(distances, torch.Tensor) else torch.from_numpy(dist_np),
                n_ants,
                cand_list_size,
                backup_list_size,
                min_new_edges,
                decay,
                alpha,
                p_best,
                gbest_as_source_prob,
                use_local_search,
                random_mode,
                enable_torch_sync,
                disable_heuristic,
                device
            )
    
    @classmethod
    def from_snapshot(
        cls,
        snapshot: Any,
        use_local_search: bool = True,
        enable_torch_sync: bool = True,
        device: str = "cpu",
        use_cpp: bool = True,
        disable_heuristic: bool = False,
    ) -> 'MFACO_TSP':
        """
        Create an MFACO_TSP instance restored from a Snapshot object.
        
        This is the preferred way to restore FACO state for training, as it
        ensures consistent baseline cache behavior by using the exact nn_list
        and backup_list stored in the snapshot.
        
        Args:
            snapshot: A Snapshot dataclass with fields:
                - distances: (n, n) distance matrix
                - n_ants: number of ants
                - k: candidate list size
                - bl: backup list size  
                - min_new_edges: min new edges parameter
                - rho: decay rate
                - alpha: pheromone exponent
                - nn_list: (n, k) nearest neighbor list
                - backup_list: (n, bl) backup neighbor list
                - pheromone_sparse: (n, k) pheromone values
                - source_route: (n,) current source solution
                - source_cost: float
                - best_route: (n,) best solution found
                - best_cost: float
                - tau_min, tau_max: pheromone bounds
            use_local_search: whether to apply 2-opt local search
            enable_torch_sync: if False, skip torch pheromone sync
            device: torch device
            use_cpp: if False, force Python backend
        
        Returns:
            MFACO_TSP instance with state restored from snapshot
        """
        distances = torch.from_numpy(snapshot.distances).float()
        
        # Create instance with snapshot parameters
        instance = cls(
            distances=distances,
            n_ants=snapshot.n_ants,
            cand_list_size=snapshot.k,
            backup_list_size=snapshot.bl,
            min_new_edges=snapshot.min_new_edges,
            decay=snapshot.rho,
            alpha=snapshot.alpha,
            use_local_search=use_local_search,
            random_mode=False,
            enable_torch_sync=enable_torch_sync,
            device=device,
            use_cpp=use_cpp,
            disable_heuristic=disable_heuristic,
        )
        
        # Load snapshot state using the load_snapshot method
        snapshot_dict = {
            'pheromone_sparse_np': snapshot.pheromone_sparse,
            'source_route': snapshot.source_route,
            'source_cost': snapshot.source_cost,
            'best_route': snapshot.best_route,
            'best_cost': snapshot.best_cost,
            'tau_min': snapshot.tau_min,
            'tau_max': snapshot.tau_max,
            'nn_list': snapshot.nn_list,
            'backup_list': snapshot.backup_list,
        }
        instance.load_snapshot(snapshot_dict, validate=False)
        
        # Update source_positions cache
        for i, v in enumerate(instance.source_route):
            instance._set_source_position(int(v), i)
        
        return instance
    
    def _set_source_position(self, node: int, position: int) -> None:
        """Set a single source position. Used during snapshot restore."""
        if self._use_cpp:
            # C++ backend: source_positions is a numpy view
            self._cpp.source_positions[node] = position
        else:
            # Python backend: direct array access
            self._py.source_positions[node] = position
    
    # ========================================================================
    # Properties (delegated to backend)
    # ========================================================================
    
    @property
    def n(self) -> int:
        return self._cpp.n if self._use_cpp else self._py.n
    
    @property
    def n_ants(self) -> int:
        return self._cpp.n_ants if self._use_cpp else self._py.n_ants
    
    @property
    def k(self) -> int:
        return self._cpp.k if self._use_cpp else self._py.k
    
    @property
    def bl(self) -> int:
        return self._cpp.bl if self._use_cpp else self._py.bl
    
    @property
    def min_new_edges(self) -> int:
        return self._cpp.min_new_edges if self._use_cpp else self._py.min_new_edges
    
    @property
    def rho(self) -> float:
        return self._cpp.rho if self._use_cpp else self._py.rho
    
    @property
    def alpha(self) -> float:
        return self._cpp.alpha if self._use_cpp else self._py.alpha
    
    @property
    def p_best(self) -> float:
        return self._cpp.p_best if self._use_cpp else self._py.p_best
    
    @property
    def use_local_search(self) -> bool:
        return self._cpp.use_local_search if self._use_cpp else self._py.use_local_search
    
    @property
    def random_mode(self) -> bool:
        return self._cpp.random_mode if self._use_cpp else self._py.random_mode
    
    @property
    def source_cost(self) -> float:
        return self._cpp.source_cost if self._use_cpp else self._py.source_cost
    
    @property
    def best_cost(self) -> float:
        return self._cpp.best_cost if self._use_cpp else self._py.best_cost
    
    @property
    def tau_min(self) -> float:
        return self._cpp.tau_min if self._use_cpp else self._py.tau_min
    
    @property
    def tau_max(self) -> float:
        return self._cpp.tau_max if self._use_cpp else self._py.tau_max
    
    @property
    def source_route(self) -> np.ndarray:
        return np.asarray(self._cpp.source_route) if self._use_cpp else self._py.source_route
    
    @property
    def best_route(self) -> np.ndarray:
        return np.asarray(self._cpp.best_route) if self._use_cpp else self._py.best_route
    
    @property
    def pheromone_sparse_np(self) -> np.ndarray:
        return np.asarray(self._cpp.pheromone_sparse_np) if self._use_cpp else self._py.pheromone_sparse_np

    @pheromone_sparse_np.setter
    def pheromone_sparse_np(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        if self._use_cpp:
            # C++ binding exposes `set_pheromone(pheromone)`.
            self._cpp.set_pheromone(arr)
            # Keep torch cache consistent (used by prob_sparse_torch/replay).
            if hasattr(self, "_pheromone_sparse") and isinstance(self._pheromone_sparse, torch.Tensor):
                self._pheromone_sparse.copy_(torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(self.device))
        else:
            self._py.pheromone_sparse_np = arr
    
    @property
    def heuristic_sparse_np(self) -> np.ndarray:
        return np.asarray(self._cpp.heuristic_sparse_np) if self._use_cpp else self._py.heuristic_sparse_np
    
    @property
    def nn_list(self) -> np.ndarray:
        return np.asarray(self._cpp.nn_list) if self._use_cpp else self._py.nn_list
    
    @property
    def backup_list(self) -> np.ndarray:
        return np.asarray(self._cpp.backup_list) if self._use_cpp else self._py.backup_list
    
    @property
    def nn_pos(self) -> np.ndarray:
        return np.asarray(self._cpp.nn_pos) if self._use_cpp else self._py.nn_pos
    
    @property
    def source_positions(self) -> np.ndarray:
        return np.asarray(self._cpp.source_positions) if self._use_cpp else self._py.source_positions
    
    @property
    def dist_np32(self) -> np.ndarray:
        """Return (n, n) distance matrix as float32 numpy array."""
        if self._use_cpp:
            return self._dist_np
        else:
            return self._py.dist_np32
    
    @property 
    def h_sparse_torch(self) -> torch.Tensor:
        """Return (n, k) heuristic sparse tensor."""
        if self._use_cpp:
            return self._h_sparse_torch
        else:
            return self._py.h_sparse_torch
    
    @property
    def nn_torch(self) -> torch.Tensor:
        """Return (n, k) nearest neighbor indices as torch tensor."""
        if self._use_cpp:
            return self._nn_torch
        else:
            return self._py.nn_torch
    
    @property
    def pheromone_sparse(self) -> torch.Tensor:
        """Return (n, k) pheromone sparse tensor."""
        if self._use_cpp:
            return self._pheromone_sparse
        else:
            return self._py.pheromone_sparse
    
    @property
    def distances(self) -> torch.Tensor:
        """Return (n, n) distances as torch tensor."""
        if self._use_cpp:
            return self._distances
        else:
            return self._py.distances
    
    @property
    def enable_torch_sync(self) -> bool:
        """Return whether torch sync is enabled."""
        if self._use_cpp:
            return self._enable_torch_sync
        else:
            return self._py.enable_torch_sync
    
    @enable_torch_sync.setter
    def enable_torch_sync(self, value: bool) -> None:
        """Set whether torch sync is enabled."""
        if self._use_cpp:
            self._enable_torch_sync = value
        else:
            self._py.enable_torch_sync = value
    
    # ========================================================================
    # API Methods
    # ========================================================================
    
    def seed_rng(self, seed: int) -> None:
        """Seed the RNG for reproducibility."""
        if self._use_cpp:
            self._cpp.seed_rng(seed)
            np.random.seed(seed)  # Also seed numpy for any Python-side randomness
        else:
            self._py.seed_rng(seed)
    
    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
        residual_logits: np.ndarray = None,
        parallel_traced: bool = False,
    ):
        """
        Sample solutions from all ants.
        
        Returns:
            costs: (n_ants,) array of costs
            flats: list of (n+1,) arrays, each with last == first
            touched_list: list of touched node arrays
            logps_nondiff: None (placeholder)
            traces: list of MFACOTrace or MFACOTrace batch if require_prob else None
        """
        if self._use_cpp:
            if require_prob:
                # Keep traced sampling deterministic.
                parallel_traced = False
            costs, flats, touched, logps, traces_raw = self._cpp.sample(
                invtemp, require_prob, residual_logits, parallel_traced
            )

            # Keep the C++ batch trace object (much faster than converting to Python lists)
            traces = traces_raw
            
            # Sync pheromone to torch if needed
            if require_prob and self.enable_torch_sync:
                self.sync_pheromone_to_torch()
            
            return costs, flats, touched, logps, traces
        else:
            # Python backend ignores parallel_traced
            return self._py.sample(invtemp, require_prob, residual_logits)
    
    def _update_pheromone_from_flat(self, best_flat: np.ndarray, best_cost: float) -> None:
        """
        Update pheromone: evaporate + deposit on best route.
        """
        if self._use_cpp:
            self._cpp._update_pheromone_from_flat(
                best_flat.astype(np.int32), 
                float(best_cost)
            )
            if self.enable_torch_sync:
                self.sync_pheromone_to_torch()
        else:
            self._py._update_pheromone_from_flat(best_flat, best_cost)
    
    def prob_sparse_torch(
        self,
        invtemp: float = 1.0,
        residual_logits: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Returns unnormalized weights for candidate edges: shape (n, k).
        
        NOTE: For C++ backend, this recomputes from numpy pheromone.
        """
        if self._use_cpp:
            tau = self.pheromone_sparse.clamp_min(1e-10) ** self.alpha

            if self.disable_heuristic:
                w = tau
            else:
                h = self.h_sparse_torch
                if invtemp != 1.0:
                    h = h ** float(invtemp)
                w = tau * h
            w = w.clamp_min(1e-12)
            
            if residual_logits is not None:
                residual_clamped = residual_logits.clamp(-10.0, 10.0)
                w = w * torch.exp(residual_clamped)
            
            return w
        else:
            return self._py.prob_sparse_torch(invtemp, residual_logits)
    
    def replay_logp_batch(
        self,
        traces: List,
        invtemp: float = 1.0,
        prob_sparse: Optional[torch.Tensor] = None,
        residual_logits: Optional[torch.Tensor] = None,
        per_ant: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch replay of log-probabilities.

        Returns per-ant log-prob sums and per-ant stochastic decision counts.
        
        NOTE: This is always done in Python/PyTorch for gradient support.
        """
        if self._use_cpp:
            # Replay logic with C++ state (needs to stay differentiable in torch)
            if prob_sparse is not None and residual_logits is not None:
                raise ValueError("Pass either prob_sparse or residual_logits, not both")
            if prob_sparse is None:
                prob_sparse = self.prob_sparse_torch(invtemp=invtemp, residual_logits=residual_logits)


            # Fast path: C++ batch trace includes pick_j + valid_mask
            if (
                HAS_CPP_BACKEND
                and isinstance(traces, MFACOTrace_CPP)
                and hasattr(traces, "pick_j")
                and hasattr(traces, "valid_mask")
            ):
                logp_pa, ndec_pa = self._replay_logp_batch_cpp(traces, prob_sparse)
                if per_ant:
                    return logp_pa, ndec_pa
                return logp_pa.sum(), ndec_pa.sum()

            # Fallback: older list-of-traces format
            logp_pa, ndec_pa = self._replay_logp_batch_impl(traces, prob_sparse)
            if per_ant:
                return logp_pa, ndec_pa
            return logp_pa.sum(), ndec_pa.sum()
        else:
            # Python backend: compute per-ant replay from trace list
            if prob_sparse is not None and residual_logits is not None:
                raise ValueError("Pass either prob_sparse or residual_logits, not both")
            if prob_sparse is None:
                prob_sparse = self.prob_sparse_torch(invtemp=invtemp, residual_logits=residual_logits)
            logp_pa, ndec_pa = self._replay_logp_batch_impl(traces, prob_sparse)
            if per_ant:
                return logp_pa, ndec_pa
            return logp_pa.sum(), ndec_pa.sum()

    def _replay_logp_batch_cpp(
        self,
        traces: 'MFACOTrace_CPP',
        prob_sparse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast vectorized replay for C++ MFACOTrace batches (per-ant).

        Uses per-decision `pick_j` (index in nn_list row) and `valid_mask` (bitmask
        of valid candidate slots at that decision time).
        """
        device = prob_sparse.device
        k = int(self.k)

        curr_np = np.asarray(traces.curr_nodes, dtype=np.int64)
        is_stoch_np = np.asarray(traces.is_stochastic, dtype=np.uint8)
        used_unif_np = np.asarray(traces.used_uniform, dtype=np.uint8)
        pick_j_np = np.asarray(traces.pick_j, dtype=np.int64)
        valid_mask_np = np.asarray(traces.valid_mask, dtype=np.uint64)

        starts_np = np.asarray(traces.starts, dtype=np.int64)
        n_ants = int(traces.n_ants)

        # Build decision -> ant index mapping using starts offsets
        counts_np = (starts_np[1:] - starts_np[:-1]).astype(np.int64)
        ant_idx_all = torch.repeat_interleave(
            torch.arange(n_ants, device=device, dtype=torch.int64),
            torch.as_tensor(counts_np, device=device, dtype=torch.int64),
        )

        is_stoch = torch.as_tensor(is_stoch_np, device=device).bool()

        # Per-ant stochastic decision counts
        if is_stoch.any():
            n_decisions_per_ant = torch.bincount(
                ant_idx_all[is_stoch], minlength=n_ants
            ).to(torch.int32)
        else:
            n_decisions_per_ant = torch.zeros((n_ants,), device=device, dtype=torch.int32)

        logp_per_ant = torch.zeros((n_ants,), device=device, dtype=torch.float32)

        # Roulette decisions
        roulette_mask = (is_stoch_np != 0) & (used_unif_np == 0) & (pick_j_np >= 0)
        if roulette_mask.any():
            curr = torch.as_tensor(curr_np[roulette_mask], device=device)
            pick = torch.as_tensor(pick_j_np[roulette_mask], device=device)
            vm = torch.as_tensor(valid_mask_np[roulette_mask].astype(np.int64), device=device)

            ant_idx = ant_idx_all[torch.as_tensor(roulette_mask, device=device)]

            w_rows = prob_sparse[curr]  # (D,k)
            bitpos = torch.arange(k, device=device, dtype=torch.int64)
            valid = ((vm.unsqueeze(1) >> bitpos) & 1).to(w_rows.dtype)
            sums = (w_rows * valid).sum(dim=1).clamp_min(1e-12)
            picked = w_rows.gather(1, pick.unsqueeze(1)).squeeze(1)
            probs = (picked / sums).clamp_min(1e-12)
            decision_logp = torch.log(probs)
            logp_per_ant.scatter_add_(0, ant_idx, decision_logp)

        # Uniform fallback decisions: log(1/m)
        uniform_mask = (is_stoch_np != 0) & (used_unif_np != 0)
        if uniform_mask.any():
            # Portable bit-count across NumPy versions.
            m = np.fromiter(
                (int(x).bit_count() for x in valid_mask_np[uniform_mask]),
                dtype=np.float32,
            )
            m_t = torch.as_tensor(m, device=device)
            decision_logp = -torch.log(m_t.clamp_min(1.0))
            ant_idx = ant_idx_all[torch.as_tensor(uniform_mask, device=device)]
            logp_per_ant.scatter_add_(0, ant_idx, decision_logp)

        return logp_per_ant, n_decisions_per_ant
    
    def _replay_logp_batch_impl(
        self,
        traces: List[MFACOTrace_PY],
        prob_sparse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implementation of replay_logp_batch using current prob_sparse.
        
        This is a Python reimplementation for C++ backend compatibility.
        """
        n = self.n
        k = self.k
        nn_list = self.nn_list
        
        n_ants = len(traces)
        logp_per_ant = torch.zeros((n_ants,), device=self.device, dtype=torch.float32)
        n_decisions_per_ant = torch.zeros((n_ants,), device=self.device, dtype=torch.int32)

        # Collect all roulette stochastic decisions (vectorized gather)
        all_curr = []
        all_pick_j = []
        all_valid_masks = []
        all_ant_idx = []

        # Uniform fallback handled per-ant (cheap)
        uniform_logp_per_ant = torch.zeros((n_ants,), device=self.device, dtype=torch.float32)
        
        for a, trace in enumerate(traces):
            visited = np.zeros(n, dtype=np.bool_)
            visited[trace.start_node] = True
            
            for i, (curr, chosen) in enumerate(zip(trace.curr_nodes, trace.chosen_nodes)):
                if trace.is_stochastic[i]:
                    n_decisions_per_ant[a] += 1

                if trace.is_stochastic[i] and trace.used_uniform_fallback[i]:
                    nn = nn_list[curr]
                    valid = [(v >= 0) and (not visited[int(v)]) for v in nn]
                    m = int(sum(valid))
                    if m > 0:
                        uniform_logp_per_ant[a] = uniform_logp_per_ant[a] + (-torch.log(torch.tensor(float(m), device=self.device)))

                if trace.is_stochastic[i] and not trace.used_uniform_fallback[i]:
                    nn = nn_list[curr]
                    valid = np.array([(v >= 0) and (not visited[int(v)]) for v in nn], dtype=np.bool_)
                    
                    pick = -1
                    for j, v in enumerate(nn):
                        if int(v) == int(chosen):
                            pick = j
                            break
                    
                    if pick >= 0 and valid[pick]:
                        all_curr.append(curr)
                        all_pick_j.append(pick)
                        all_valid_masks.append(valid)
                        all_ant_idx.append(a)
                
                visited[int(chosen)] = True
        
        n_roulette = len(all_curr)
        if n_roulette == 0:
            return uniform_logp_per_ant, n_decisions_per_ant
        
        # Convert to tensors
        curr_t = torch.tensor(all_curr, device=self.device, dtype=torch.long)
        pick_t = torch.tensor(all_pick_j, device=self.device, dtype=torch.long)
        valid_t = torch.tensor(np.array(all_valid_masks), device=self.device, dtype=torch.bool)
        
        # Gather weights
        w_rows = prob_sparse[curr_t]
        w_masked = w_rows * valid_t.float()
        sums = w_masked.sum(dim=1).clamp_min(1e-12)
        w_picked = w_rows[torch.arange(n_roulette, device=self.device), pick_t]
        
        probs = (w_picked / sums).clamp_min(1e-12)
        log_probs = torch.log(probs)

        ant_idx_t = torch.tensor(all_ant_idx, device=self.device, dtype=torch.long)
        logp_per_ant.scatter_add_(0, ant_idx_t, log_probs)
        logp_per_ant = logp_per_ant + uniform_logp_per_ant

        return logp_per_ant, n_decisions_per_ant
    
    def sync_pheromone_to_torch(self) -> None:
        """Sync numpy pheromone to torch tensor."""
        if self._use_cpp:
            phe_np = np.asarray(self._cpp.pheromone_sparse_np)
            phe_cpu = torch.as_tensor(phe_np)
            self.pheromone_sparse.copy_(phe_cpu.to(self.device))
        else:
            self._py.sync_pheromone_to_torch()
    
    def load_snapshot(self, snapshot: dict, validate: bool = True) -> None:
        """Load state from snapshot."""
        if self._use_cpp:
            self._cpp.load_snapshot(
                snapshot['pheromone_sparse_np'].astype(np.float32),
                snapshot['source_route'].astype(np.int32),
                float(snapshot['source_cost']),
                snapshot['best_route'].astype(np.int32),
                float(snapshot['best_cost']),
                float(snapshot['tau_min']),
                float(snapshot['tau_max']),
                snapshot.get('nn_list', self.nn_list).astype(np.int32),
                snapshot.get('backup_list', self.backup_list).astype(np.int32)
            )
            # Refresh cached tensors that depend on nn_list/heuristic.
            self._h_sparse_torch = torch.as_tensor(np.asarray(self._cpp.heuristic_sparse_np)).to(self.device)
            self._nn_torch = torch.as_tensor(np.asarray(self._cpp.nn_list), dtype=torch.long).to(self.device)
            if self.enable_torch_sync:
                self.sync_pheromone_to_torch()
        else:
            self._py.load_snapshot(snapshot, validate)
    
    def save_snapshot(self) -> dict:
        """Save current state for later restoration."""
        if self._use_cpp:
            return {
                'source_route': np.array(self._cpp.source_route).copy(),
                'source_cost': self._cpp.source_cost,
                'best_route': np.array(self._cpp.best_route).copy(),
                'best_cost': self._cpp.best_cost,
                'pheromone_sparse_np': np.array(self._cpp.pheromone_sparse_np).copy(),
                'tau_min': self._cpp.tau_min,
                'tau_max': self._cpp.tau_max,
                'n': self._cpp.n,
                'k': self._cpp.k,
                'bl': self._cpp.bl,
                'rho': self._cpp.rho,
                'alpha': self._cpp.alpha,
                'p_best': self._cpp.p_best,
                'min_new_edges': self._cpp.min_new_edges,
                'random_mode': self._cpp.random_mode,
                'use_local_search': self._cpp.use_local_search,
            }
        else:
            return self._py.save_snapshot()


# Convenience function to check if C++ backend is available
def has_cpp_backend() -> bool:
    """Returns True if the C++ backend is available."""
    return HAS_CPP_BACKEND

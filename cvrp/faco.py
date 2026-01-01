# faco.py
# ------------------------------------------------------------
# Focused FACO-style ACO for CVRP (Depot = 0)
#
# PATCHED (fixes + training support):
# 1) Fixed undefined attrs: self.k_sparse / simple_heuristic usage removed.
# 2) Heuristic init now correctly builds sparse heuristic on kNN edges: 1 / dist.
# 3) Added training-mode sampling:
#    - sample(require_prob=True) returns (costs, flats, touched, logps, traces)
#    - logps are Torch tensors (can carry grad through learned heuristic)
#    - deterministic "restart" rule (min unvisited) for trace replay stability
# 4) Added replay_logp(...) to recompute logπ under new heuristic for PPO.
#
# Notes:
# - Environment (route edits) is non-differentiable (Python lists), which is fine.
#   Grad flows through log-probs computed from (pheromone, heuristic).
# - Pheromone is updated outside the gradient (standard ACO / RL style).
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
import numba as nb
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict, Any

from torch.distributions import Categorical

EPS = 1e-10


# ============================================================
# kNN utilities
# ============================================================

def build_nearest_neighbor_lists(distances: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build nearest neighbors and backup list (fixed 64).
    distances: (n,n) float32
    returns:
      nn_list: (n,k)
      backup_nn_list: (n,64)
    """
    n = distances.shape[0]
    k = min(k, n - 1)
    nn_list = np.full((n, k), -1, dtype=np.int64)
    backup = np.full((n, 64), -1, dtype=np.int64)

    for i in range(n):
        order = np.argsort(distances[i])
        order = order[order != i]
        nn = order[:k]
        nn_list[i, :len(nn)] = nn

        rest = order[k:k + 64]
        backup[i, :len(rest)] = rest

    return nn_list, backup


# ============================================================
# CVRP representation helpers
# ============================================================

def flatten_routes(routes: List[List[int]]) -> np.ndarray:
    out = [0]
    for r in routes:
        if len(r) == 0:
            continue
        out.extend(r)
        out.append(0)
    if out[-1] != 0:
        out.append(0)
    return np.array(out, dtype=np.int64)


def split_flat_to_routes(flat: np.ndarray) -> List[List[int]]:
    routes: List[List[int]] = []
    cur: List[int] = []
    for x in flat:
        if x == 0:
            if len(cur) > 0:
                routes.append(cur)
                cur = []
        else:
            cur.append(int(x))
    if len(cur) > 0:
        routes.append(cur)
    return routes


def route_cost_flat(flat: np.ndarray, dist: np.ndarray) -> float:
    return float(dist[flat[:-1], flat[1:]].sum())


def compute_route_load(route: List[int], demand: np.ndarray) -> int:
    if len(route) == 0:
        return 0
    return int(demand[np.array(route, dtype=np.int64)].sum())


# ============================================================
# Focused intra-route 2-opt (Numba)
# Route cycle representation: tour = [0, c1, c2, ..., cm]
# implicit wrap-around back to 0.
# ============================================================

@nb.njit(nb.int32[:](nb.uint16[:], nb.int32), nogil=True)
def _build_positions_cycle(tour: np.ndarray, n_nodes: int) -> np.ndarray:
    pos = np.full(n_nodes, -1, dtype=np.int32)
    for i in range(tour.shape[0]):
        pos[tour[i]] = i
    return pos


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.int32[:], nb.int32), nogil=True)
def _two_opt_once_focused_cycle(distmat: np.ndarray, tour: np.ndarray,
                               fixed_positions: np.ndarray, fixed_len: int) -> np.float32:
    n = tour.shape[0]
    best_delta = 0.0
    best_p = 0
    best_q = 0

    for t in range(fixed_len):
        i = fixed_positions[t]
        if i <= 0 or i >= n - 1:
            continue

        node_i = tour[i]
        node_prev = tour[i - 1]

        for j in range(i + 1, n):
            node_j = tour[j]
            node_next = tour[(j + 1) % n]

            if node_prev == node_j or node_next == node_i:
                continue

            change = (distmat[node_prev, node_j] + distmat[node_i, node_next]
                      - distmat[node_prev, node_i] - distmat[node_j, node_next])
            if change < best_delta:
                best_delta = change
                best_p = i
                best_q = j

    if best_delta < -1e-6:
        tour[best_p:best_q + 1] = np.flip(tour[best_p:best_q + 1])
        return np.float32(best_delta)

    return np.float32(0.0)


@nb.njit(nb.uint16[:](nb.float32[:, :], nb.uint16[:], nb.int32[:], nb.int32, nb.int64), nogil=True)
def two_opt_focused_cycle(distmat: np.ndarray, tour: np.ndarray,
                          fixed_positions: np.ndarray, fixed_len: int,
                          max_iterations: int = 2000) -> np.ndarray:
    it = 0
    delta = np.float32(-1.0)
    while delta < -1e-6 and it < max_iterations:
        delta = _two_opt_once_focused_cycle(distmat, tour, fixed_positions, fixed_len)
        it += 1
    return tour


# ============================================================
# Trace + Focus result
# ============================================================

@dataclass
class FACOTrace:
    start_customer: int
    curr_seq: torch.Tensor      # (T,)
    choice_seq: torch.Tensor    # (T,)
    min_new_edges: int
    k: int


@dataclass
class FocusResult:
    routes: List[List[int]]
    touched_customers: np.ndarray
    new_edges_count: int
    logp: Optional[torch.Tensor] = None
    trace: Optional[FACOTrace] = None


# ============================================================
# FACO (Focused)
# ============================================================

class FACO:
    def __init__(
        self,
        distances: torch.Tensor,        # (n,n)
        demand: torch.Tensor,           # (n,)
        n_ants: int = 20,
        k_nearest: int = 20,
        capacity: int = 50,
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        pheromone: Optional[torch.Tensor] = None,  # either (n,n) or (n,k)
        heuristic: Optional[torch.Tensor] = None,  # either (n,n) or (n,k)
        min_new_edges: int = 8,
        sample_two_opt: bool = True,
        ls_max_iterations: int = 2000,
        use_full_graph: bool = False,   # If True, use all edges (k = n-1) for ablation
        device: str = "cpu",
    ):
        self.device = device
        self.dist_t = distances.to(device)
        self.dem_t = demand.to(device)

        self.dist = self.dist_t.detach().cpu().numpy().astype(np.float32)
        self.demand = self.dem_t.detach().cpu().numpy().astype(np.int32)

        self.n = self.dist.shape[0]
        assert self.demand.shape[0] == self.n
        self.capacity = int(capacity)

        self.n_ants = int(n_ants)
        self.use_full_graph = bool(use_full_graph)
        
        # In full graph mode, use all neighbors (k = n-1)
        if self.use_full_graph:
            self.k = self.n - 1
        else:
            self.k = min(int(k_nearest), self.n - 1)
        
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.MIN_NEW_EDGES = int(min_new_edges)
        self.sample_two_opt = bool(sample_two_opt)
        self.ls_max_iterations = int(ls_max_iterations)

        # kNN structures
        self.nn_list, self.backup_nn_list = build_nearest_neighbor_lists(self.dist, self.k)
        self.nn_index_map = np.full((self.n, self.n), -1, dtype=np.int32)
        for i in range(self.n):
            for j in range(self.k):
                v = self.nn_list[i, j]
                if v >= 0:
                    self.nn_index_map[i, v] = j

        # pheromone_sparse: (n,k) - will be initialized after tau_max is computed
        self._pheromone_init = pheromone  # store for later initialization

        # heuristic_sparse: (n,k)
        if heuristic is None:
            # default heuristic on kNN edges: 1 / dist
            self.heuristic_sparse = self._default_heuristic_sparse().to(device)
        else:
            heuristic = heuristic.to(device)
            if heuristic.shape == (self.n, self.k):
                self.heuristic_sparse = heuristic
            elif heuristic.shape == (self.n, self.n):
                self.heuristic_sparse = self._compress_full_to_sparse(heuristic)
            else:
                raise ValueError(f"heuristic must be (n,n) or (n,k), got {heuristic.shape}")

        # init best solution
        self.best_flat = self._init_greedy_solution_flat()
        self.best_cost = route_cost_flat(self.best_flat, self.dist)
        
        # MMAS parameters
        self.p_best = 0.05  # MMAS typical default
        self.tau_min, self.tau_max = self._calc_trail_limits(self.best_cost)
        
        # Now initialize pheromone with tau_max (MMAS standard)
        if self._pheromone_init is None:
            self.pheromone_sparse = torch.full((self.n, self.k), self.tau_max, device=device, dtype=torch.float32)
        else:
            pheromone = self._pheromone_init.to(device)
            if pheromone.shape == (self.n, self.k):
                self.pheromone_sparse = pheromone
            elif pheromone.shape == (self.n, self.n):
                self.pheromone_sparse = self._compress_full_to_sparse(pheromone)
            else:
                raise ValueError(f"pheromone must be (n,n) or (n,k), got {pheromone.shape}")
            # Clamp initial pheromone to MMAS bounds
            self.pheromone_sparse.clamp_(min=self.tau_min, max=self.tau_max)
        del self._pheromone_init

    # ------------------------------------------------------------
    # Helpers for sparse conversion
    # ------------------------------------------------------------

    def _compress_full_to_sparse(self, full_matrix: torch.Tensor) -> torch.Tensor:
        """Extract sparse (n, k) from full (n, n) matrix, preserving gradients."""
        nn = torch.from_numpy(self.nn_list).to(full_matrix.device)
        rows = torch.arange(self.n, device=full_matrix.device).unsqueeze(1).expand(self.n, self.k)
        cols = nn.clamp_min(0)  # Clamp -1 padding to 0 (will extract garbage but won't be used)
        # Use direct advanced indexing - this preserves gradient!
        sparse = full_matrix[rows, cols].clamp_min(1e-10)
        return sparse

    def _default_heuristic_sparse(self) -> torch.Tensor:
        # heuristic(u, v) = 1/dist(u,v) on kNN edges, else 0 (not stored)
        nn = self.nn_list
        out = np.full((self.n, self.k), 1e-10, dtype=np.float32)
        for i in range(self.n):
            for j in range(self.k):
                v = nn[i, j]
                if v >= 0:
                    d = self.dist[i, v]
                    out[i, j] = 1.0 / max(d, 1e-10)
        return torch.from_numpy(out)

    def set_heuristic(self, heuristic: torch.Tensor):
        """
        Allow updating heuristic during training/inference.
        heuristic: (n,n) or (n,k)
        """
        heuristic = heuristic.to(self.device)
        if heuristic.shape == (self.n, self.k):
            self.heuristic_sparse = heuristic
        elif heuristic.shape == (self.n, self.n):
            self.heuristic_sparse = self._compress_full_to_sparse(heuristic)
        else:
            raise ValueError(f"heuristic must be (n,n) or (n,k), got {heuristic.shape}")

    # ------------------------------------------------------------
    # Initialization: simple greedy feasible route building
    # ------------------------------------------------------------

    def _init_greedy_solution_flat(self) -> np.ndarray:
        unserved = set(range(1, self.n))
        routes: List[List[int]] = []

        while unserved:
            load = 0
            cur = 0
            route: List[int] = []
            while True:
                best = None
                best_d = 1e30
                for v in list(unserved):
                    if load + int(self.demand[v]) > self.capacity:
                        continue
                    d = float(self.dist[cur, v])
                    if d < best_d:
                        best_d = d
                        best = v
                if best is None:
                    break
                route.append(best)
                unserved.remove(best)
                load += int(self.demand[best])
                cur = best
            routes.append(route)

        return flatten_routes(routes)

    # ------------------------------------------------------------
    # Probabilities (torch for training / numpy for fast eval)
    # ------------------------------------------------------------

    def prob_sparse_torch(self, invtemp: float) -> torch.Tensor:
        # (n,k) torch tensor, grad can flow through heuristic_sparse
        tau = self.pheromone_sparse.clamp_min(EPS)
        heu = self.heuristic_sparse.clamp_min(EPS)
        eff_beta = self.beta * float(invtemp)
        return (tau ** self.alpha) * (heu ** eff_beta)

    def prob_sparse_numpy(self, invtemp: float) -> np.ndarray:
        # (n,k) numpy float32
        prob = self.prob_sparse_torch(invtemp).detach().cpu().numpy().astype(np.float32)
        prob = np.maximum(prob, EPS).astype(np.float32)
        return prob

    # ------------------------------------------------------------
    # Public sampling API
    # ------------------------------------------------------------

    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
    ):
        """
        If require_prob=False:
            returns: costs (n_ants,), flats (list), touched (list)
        If require_prob=True:
            returns: costs, flats, touched, logps, traces
              - logps: torch.Tensor (n_ants,) on self.device
              - traces: list[FACOTrace] length n_ants
        """
        ref_flat = self.best_flat
        ref_edges = set(zip(ref_flat[:-1].tolist(), ref_flat[1:].tolist()))

        costs = np.zeros(self.n_ants, dtype=np.float32)
        flats: List[np.ndarray] = []
        touched_list: List[np.ndarray] = []

        if require_prob:
            prob_sparse = self.prob_sparse_torch(invtemp)  # torch
            logps: List[torch.Tensor] = []
            traces: List[FACOTrace] = []
        else:
            prob_sparse = self.prob_sparse_numpy(invtemp)  # numpy
            logps = []
            traces = []

        for a in range(self.n_ants):
            res = self._focused_modify(
                ref_flat=ref_flat,
                ref_edges=ref_edges,
                prob_sparse=prob_sparse,
                target_new_edges=self.MIN_NEW_EDGES,
                require_prob=require_prob,
            )

            if self.sample_two_opt and res.touched_customers.size > 0:
                res.routes = self._focused_two_opt_intra(res.routes, res.touched_customers)

            flat = flatten_routes(res.routes)
            flats.append(flat)
            touched_list.append(res.touched_customers)
            costs[a] = route_cost_flat(flat, self.dist)

            if require_prob:
                assert res.logp is not None and res.trace is not None
                logps.append(res.logp)
                traces.append(res.trace)

                # prob_sparse is the same tensor used to sample
                lp_re = self.replay_logp(res.trace, invtemp=invtemp, ref_flat=ref_flat, prob_sparse=prob_sparse)
                err = (lp_re.detach() - res.logp.detach()).abs().item()
                if err > 1e-4:
                    print("[REPLAY MISMATCH INSIDE SAMPLE]", err)

        if require_prob:
            logps_t = torch.stack(logps) if len(logps) else torch.zeros((self.n_ants,), device=self.device)
            return costs, flats, touched_list, logps_t, traces

        return costs, flats, touched_list

    # ------------------------------------------------------------
    # Focused modify: relocate edits until enough new edges appear
    # ------------------------------------------------------------

    def _focused_modify(
        self,
        ref_flat: np.ndarray,
        ref_edges: Set[Tuple[int, int]],
        prob_sparse,                  # torch (n,k) if require_prob else numpy (n,k)
        target_new_edges: int,
        require_prob: bool,
    ) -> FocusResult:
        routes = split_flat_to_routes(ref_flat)
        route_loads = [compute_route_load(r, self.demand) for r in routes]

        route_id = np.full(self.n, -1, dtype=np.int32)
        pos_in_route = np.full(self.n, -1, dtype=np.int32)

        def rebuild_maps():
            route_id.fill(-1)
            pos_in_route.fill(-1)
            for ridx, r in enumerate(routes):
                for p, v in enumerate(r):
                    route_id[v] = ridx
                    pos_in_route[v] = p

        rebuild_maps()

        def pred_succ_in_route(r: List[int], p: int) -> Tuple[int, int]:
            pred = 0 if p == 0 else r[p - 1]
            succ = 0 if p == (len(r) - 1) else r[p + 1]
            return pred, succ

        counted_new_edges: Set[Tuple[int, int]] = set()
        touched: Set[int] = set()
        new_edges = 0

        # --- start customer: random but stored in trace for replay ---
        customers = np.arange(1, self.n, dtype=np.int64)
        if customers.size == 0:
            return FocusResult(routes, np.array([], dtype=np.int64), 0)

        start_customer = int(np.random.randint(1, self.n))
        current = start_customer

        visited = np.zeros(self.n, dtype=np.uint8)
        visited[current] = 1

        def feasible_insert(curr: int, v: int) -> bool:
            if v <= 0 or curr <= 0:
                return False
            rt = int(route_id[curr])
            rs = int(route_id[v])
            if rt < 0 or rs < 0:
                return False
            if rt == rs:
                return True
            return (route_loads[rt] + int(self.demand[v]) <= self.capacity)

        def relocate_after(curr: int, v: int) -> Tuple[List[Tuple[int, int]], List[int]]:
            """
            Move v to be successor of curr.
            Returns:
              added_edges: local new directed edges (for counting)
              touched_nodes: nodes around modified edges
            """
            nonlocal routes, route_loads

            rt = int(route_id[curr])
            rs = int(route_id[v])
            if rt < 0 or rs < 0:
                return [], []
            
            # Early exit if v already follows curr (no-op move)
            pt = int(pos_in_route[curr])
            tgt = routes[rt]
            succ_curr = 0 if pt == (len(tgt) - 1) else tgt[pt + 1]
            if succ_curr == v:
                return [], [curr, v]

            # source info
            ps = int(pos_in_route[v])
            src = routes[rs]
            pred_v, succ_v = pred_succ_in_route(src, ps)

            # target info
            pt = int(pos_in_route[curr])
            tgt = routes[rt]
            succ_curr = 0 if pt == (len(tgt) - 1) else tgt[pt + 1]

            # remove v from source
            src.pop(ps)
            if rs != rt:
                route_loads[rs] -= int(self.demand[v])

            # if source becomes empty, remove route
            if len(src) == 0:
                routes.pop(rs)
                route_loads.pop(rs)
                rebuild_maps()
            else:
                rebuild_maps()

            # recompute target after potential shifts
            rt = int(route_id[curr])
            tgt = routes[rt]
            pt = int(pos_in_route[curr])
            succ_curr = 0 if pt == (len(tgt) - 1) else tgt[pt + 1]

            # insert v after curr
            tgt.insert(pt + 1, v)
            if rs != rt:
                route_loads[rt] += int(self.demand[v])

            rebuild_maps()

            added_edges = [(pred_v, succ_v), (curr, v), (v, succ_curr)]
            touched_nodes = [curr, v, pred_v, succ_v, succ_curr]
            return added_edges, touched_nodes

        # log-prob + trace (torch)
        logps: List[torch.Tensor] = []
        curr_seq: List[int] = []
        choice_seq: List[int] = []

        max_steps = (self.n - 1) * 3
        steps = 0

        def pick_next(current: int):
            """Return (v, lp_tensor_or_None). lp is 0 for deterministic fallbacks."""
            # ---------- (1) candidate list roulette ----------
            cand = self.nn_list[current]  # (k,)
            feasible_vs: List[int] = []
            feasible_w = []

            if require_prob:
                w_row: torch.Tensor = prob_sparse[current]  # (k,)
                for j in range(self.k):
                    v = int(cand[j])
                    if v <= 0 or visited[v] == 1 or (not feasible_insert(current, v)):
                        continue
                    feasible_vs.append(v)
                    feasible_w.append(w_row[j].clamp_min(EPS))
            else:
                w_row = prob_sparse[current].astype(np.float64)
                for j in range(self.k):
                    v = int(cand[j])
                    if v <= 0 or visited[v] == 1 or (not feasible_insert(current, v)):
                        continue
                    feasible_vs.append(v)
                    feasible_w.append(float(w_row[j]))

            if len(feasible_vs) > 0:
                if require_prob:
                    cand_nodes_t = torch.tensor(feasible_vs, dtype=torch.long, device=self.device)
                    w = torch.stack(feasible_w)              # (m,)
                    p = w / (w.sum() + EPS)
                    dist = Categorical(probs=p)
                    idx = dist.sample()
                    v = int(cand_nodes_t[idx].item())
                    lp = dist.log_prob(idx)
                    return v, lp
                else:
                    w = np.array(feasible_w, dtype=np.float64) + 1e-12
                    w = w / w.sum()
                    v = int(np.random.choice(np.array(feasible_vs, dtype=np.int64), p=w))
                    return v, None

            # ---------- (2) backup list first feasible ----------
            for vv in self.backup_nn_list[current]:
                v = int(vv)
                if v < 0:
                    break
                if v <= 0 or visited[v] == 1:
                    continue
                if not feasible_insert(current, v):
                    continue
                if require_prob:
                    return v, torch.tensor(0.0, device=self.device)
                return v, None

            # ---------- (3) global fallback: nearest feasible ----------
            best_v = -1
            best_d = 1e30
            # only customers
            for v in range(1, self.n):
                if visited[v] == 1:
                    continue
                if not feasible_insert(current, v):
                    continue
                d = float(self.dist[current, v])
                if d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and v < best_v):
                    best_d = d
                    best_v = v

            if best_v >= 0:
                if require_prob:
                    return best_v, torch.tensor(0.0, device=self.device)
                return best_v, None

            # nothing feasible under CVRP constraints
            return -1, None

        while new_edges < target_new_edges and steps < max_steps:
            steps += 1

            v, lp = pick_next(current)

            if v < 0:
                # ---------- (4) CVRP-only: restart current ----------
                unv = np.where(visited[1:] == 0)[0] + 1
                if unv.size == 0:
                    break
                new_current = int(unv.min())
                
                # Record restart in trace: use negative choice to signal restart
                if require_prob:
                    curr_seq.append(int(current))
                    choice_seq.append(-new_current)  # negative signals restart to this node
                    logps.append(torch.tensor(0.0, device=self.device))  # deterministic
                
                current = new_current
                visited[current] = 1
                continue

            # IMPORTANT: record every action (even deterministic fallbacks)
            if require_prob:
                curr_seq.append(int(current))
                choice_seq.append(int(v))
                logps.append(lp if lp is not None else torch.tensor(0.0, device=self.device))

            added_edges, touched_nodes = relocate_after(current, v)

            for e in added_edges:
                if e not in ref_edges and e not in counted_new_edges:
                    counted_new_edges.add(e)
                    new_edges += 1
                    if new_edges >= target_new_edges:
                        break

            for x in touched_nodes:
                if x > 0:
                    touched.add(int(x))

            current = v
            visited[v] = 1

        touched_arr = np.array(sorted(touched), dtype=np.int64)

        if require_prob:
            logp = torch.stack(logps).sum() if len(logps) else torch.tensor(0.0, device=self.device)
            trace = FACOTrace(
                start_customer=start_customer,
                curr_seq=torch.tensor(curr_seq, dtype=torch.long, device=self.device),
                choice_seq=torch.tensor(choice_seq, dtype=torch.long, device=self.device),
                min_new_edges=int(target_new_edges),
                k=int(self.k),
            )
            return FocusResult(routes=routes, touched_customers=touched_arr, new_edges_count=int(new_edges), logp=logp, trace=trace)

        return FocusResult(routes=routes, touched_customers=touched_arr, new_edges_count=int(new_edges))

    # ------------------------------------------------------------
    # Focused intra-route 2-opt
    # ------------------------------------------------------------

    def _focused_two_opt_intra(self, routes: List[List[int]], touched_customers: np.ndarray) -> List[List[int]]:
        if touched_customers.size == 0:
            return routes

        touched_set = set(int(x) for x in touched_customers.tolist())

        cust_route = np.full(self.n, -1, dtype=np.int32)
        for ridx, r in enumerate(routes):
            for v in r:
                cust_route[v] = ridx

        affected_routes = set()
        for v in touched_set:
            ridx = int(cust_route[v])
            if ridx >= 0:
                affected_routes.add(ridx)

        distmat = self.dist

        for ridx in list(affected_routes):
            r = routes[ridx]
            if len(r) < 4:
                continue

            tour = np.empty(len(r) + 1, dtype=np.uint16)
            tour[0] = 0
            tour[1:] = np.array(r, dtype=np.uint16)

            pos = _build_positions_cycle(tour, self.n)

            fixed: List[int] = []
            L = tour.shape[0]
            for u in r:
                if u not in touched_set:
                    continue
                pu = int(pos[u])
                if pu <= 0:
                    continue
                fixed.append(pu)
                if pu - 1 > 0:
                    fixed.append(pu - 1)
                if pu + 1 < L:
                    fixed.append(pu + 1)

            if len(fixed) == 0:
                continue

            fixed = np.array(sorted(set(fixed)), dtype=np.int32)
            tour2 = two_opt_focused_cycle(distmat, tour, fixed, fixed.shape[0], max_iterations=self.ls_max_iterations)
            routes[ridx] = tour2[1:].astype(np.int64).tolist()

        return routes

    # ------------------------------------------------------------
    # PPO helper: replay log-prob under NEW prob_sparse (torch)
    # ------------------------------------------------------------

    def replay_logp(
        self,
        trace: FACOTrace,
        invtemp: float = 1.0,
        ref_flat: Optional[np.ndarray] = None,
        prob_sparse: Optional[torch.Tensor] = None,
        return_entropy: bool = False,
        return_n_terms: bool = False,
    ) -> torch.Tensor:
        """
        Recompute logπ for the stored trace under the *current* heuristic/pheromone.

        - Uses the same deterministic restart rule as sampling (min unvisited).
        - If an action in the trace becomes infeasible, returns -inf.

        Args:
          trace: FACOTrace from sample(require_prob=True)
          invtemp: used if prob_sparse is None
          ref_flat: if None, uses self.best_flat snapshot at call time (usually you want the original state's best_flat)
          prob_sparse: optional precomputed (n,k) torch probs for that state
          return_entropy: if True, returns (logp, entropy) tuple where entropy is from feasible distribution
          return_n_terms: if True, also returns number of stochastic terms accumulated
        """
        if ref_flat is None:
            ref_flat = self.best_flat

        if prob_sparse is None:
            prob_sparse = self.prob_sparse_torch(invtemp)

        routes = split_flat_to_routes(ref_flat)
        route_loads = [compute_route_load(r, self.demand) for r in routes]

        route_id = np.full(self.n, -1, dtype=np.int32)
        pos_in_route = np.full(self.n, -1, dtype=np.int32)

        def rebuild_maps():
            route_id.fill(-1)
            pos_in_route.fill(-1)
            for ridx, r in enumerate(routes):
                for p, v in enumerate(r):
                    route_id[v] = ridx
                    pos_in_route[v] = p

        rebuild_maps()

        def pred_succ_in_route(r: List[int], p: int) -> Tuple[int, int]:
            pred = 0 if p == 0 else r[p - 1]
            succ = 0 if p == (len(r) - 1) else r[p + 1]
            return pred, succ

        def feasible_insert(curr: int, v: int) -> bool:
            if v <= 0 or curr <= 0:
                return False
            rt = int(route_id[curr])
            rs = int(route_id[v])
            if rt < 0 or rs < 0:
                return False
            if rt == rs:
                return True
            return (route_loads[rt] + int(self.demand[v]) <= self.capacity)

        def relocate_after(curr: int, v: int):
            nonlocal routes, route_loads

            rt = int(route_id[curr])
            rs = int(route_id[v])
            if rt < 0 or rs < 0:
                return

            ps = int(pos_in_route[v])
            src = routes[rs]
            pred_v, succ_v = pred_succ_in_route(src, ps)

            # remove v
            src.pop(ps)
            if rs != rt:
                route_loads[rs] -= int(self.demand[v])

            if len(src) == 0:
                routes.pop(rs)
                route_loads.pop(rs)
                rebuild_maps()
            else:
                rebuild_maps()

            # recompute target
            rt = int(route_id[curr])
            tgt = routes[rt]
            pt = int(pos_in_route[curr])

            tgt.insert(pt + 1, v)
            if rs != rt:
                route_loads[rt] += int(self.demand[v])

            rebuild_maps()

        visited = np.zeros(self.n, dtype=np.uint8)
        current = int(trace.start_customer)
        visited[current] = 1

        logps: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []  # Track entropy from feasible distribution
        n_stochastic_terms = 0  # Count only stochastic (candidate-list) terms
        curr_seq = trace.curr_seq.detach().cpu().numpy()
        choice_seq = trace.choice_seq.detach().cpu().numpy()

        for t in range(choice_seq.shape[0]):
            curr_t = int(curr_seq[t])
            v_t = int(choice_seq[t])

            current = curr_t

            # Handle restart signal (negative v_t means restart to abs(v_t))
            if v_t < 0:
                new_current = -v_t
                # Verify deterministic restart matches trace
                unv = np.where(visited[1:] == 0)[0] + 1
                if unv.size == 0 or int(unv.min()) != new_current:
                    # Restart mismatch
                    if return_entropy:
                        if return_n_terms:
                            return torch.tensor(float("-inf"), device=self.device), torch.tensor(0.0, device=self.device), n_stochastic_terms
                        return torch.tensor(float("-inf"), device=self.device), torch.tensor(0.0, device=self.device)
                    if return_n_terms:
                        return torch.tensor(float("-inf"), device=self.device), n_stochastic_terms
                    return torch.tensor(float("-inf"), device=self.device)
                
                # Restart is deterministic => no log-prob contribution, no stochastic count
                logps.append(torch.tensor(0.0, device=self.device))
                entropies.append(torch.tensor(0.0, device=self.device))
                current = new_current
                visited[current] = 1
                continue

            # build feasible candidate list as in sampler
            cand = self.nn_list[current]
            w_row = prob_sparse[current]  # (k,) torch

            feasible_vs: List[int] = []
            feasible_w: List[torch.Tensor] = []
            for j in range(self.k):
                v = int(cand[j])
                if v <= 0:
                    continue
                if visited[v] == 1:
                    continue
                if not feasible_insert(current, v):
                    continue
                feasible_vs.append(v)
                feasible_w.append(w_row[j].clamp_min(EPS))

            if len(feasible_vs) == 0:
                # ---------- (2) backup list first feasible ----------
                det_v = -1
                for vv in self.backup_nn_list[current]:
                    v = int(vv)
                    if v < 0:
                        break
                    if v <= 0 or visited[v] == 1:
                        continue
                    if not feasible_insert(current, v):
                        continue
                    det_v = v
                    break

                # ---------- (3) global nearest fallback ----------
                if det_v < 0:
                    best_v = -1
                    best_d = 1e30
                    for v in range(1, self.n):
                        if visited[v] == 1:
                            continue
                        if not feasible_insert(current, v):
                            continue
                        d = float(self.dist[current, v])
                        if d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and v < best_v):
                            best_d = d
                            best_v = v
                    det_v = best_v

                if det_v < 0:
                    # CVRP-only: restart mismatch → -inf
                    if return_entropy:
                        if return_n_terms:
                            return torch.tensor(float("-inf"), device=self.device), torch.tensor(0.0, device=self.device), n_stochastic_terms
                        return torch.tensor(float("-inf"), device=self.device), torch.tensor(0.0, device=self.device)
                    if return_n_terms:
                        return torch.tensor(float("-inf"), device=self.device), n_stochastic_terms
                    return torch.tensor(float("-inf"), device=self.device)

                # trace must match deterministic choice
                if v_t != det_v:
                    if return_entropy:
                        if return_n_terms:
                            return torch.tensor(float("-inf"), device=self.device), torch.tensor(0.0, device=self.device), n_stochastic_terms
                        return torch.tensor(float("-inf"), device=self.device), torch.tensor(0.0, device=self.device)
                    if return_n_terms:
                        return torch.tensor(float("-inf"), device=self.device), n_stochastic_terms
                    return torch.tensor(float("-inf"), device=self.device)

                # deterministic action => logp += 0, entropy += 0, no stochastic count
                logps.append(torch.tensor(0.0, device=self.device))
                entropies.append(torch.tensor(0.0, device=self.device))
                relocate_after(current, v_t)
                visited[v_t] = 1
                current = v_t
                continue

            # ---------- candidate list action (stochastic) ----------
            cand_nodes_t = torch.tensor(feasible_vs, dtype=torch.long, device=self.device)
            w = torch.stack(feasible_w)
            p = w / (w.sum() + EPS)
            dist = Categorical(probs=p)

            idx = (cand_nodes_t == v_t).nonzero(as_tuple=False)
            if idx.numel() == 0:
                if return_entropy:
                    if return_n_terms:
                        return torch.tensor(float("-inf"), device=self.device), torch.tensor(0.0, device=self.device), n_stochastic_terms
                    return torch.tensor(float("-inf"), device=self.device), torch.tensor(0.0, device=self.device)
                if return_n_terms:
                    return torch.tensor(float("-inf"), device=self.device), n_stochastic_terms
                return torch.tensor(float("-inf"), device=self.device)
            idx = idx[0, 0]

            logps.append(dist.log_prob(idx))
            entropies.append(dist.entropy())  # Entropy from feasible distribution
            n_stochastic_terms += 1  # This is a stochastic choice

            relocate_after(current, v_t)
            visited[v_t] = 1
            current = v_t

        logp_sum = torch.stack(logps).sum() if len(logps) else torch.tensor(0.0, device=self.device)
        if return_entropy:
            ent_mean = torch.stack(entropies).mean() if len(entropies) else torch.tensor(0.0, device=self.device)
            if return_n_terms:
                return logp_sum, ent_mean, n_stochastic_terms
            return logp_sum, ent_mean
        if return_n_terms:
            return logp_sum, n_stochastic_terms
        return logp_sum

    # ------------------------------------------------------------
    # MMAS trail limits computation
    # ------------------------------------------------------------

    def _calc_trail_limits(self, best_cost: float) -> tuple:
        """
        Compute MMAS trail limits [tau_min, tau_max].
        
        Based on candidate-list variant used in FACO/MFACO:
        - tau_max = 1 / (best_cost * (1 - rho))
        - tau_min uses candidate list size (k) as avg
        """
        # Map decay to MMAS rho (evaporation rate)
        rho = 1.0 - float(self.decay)
        rho = min(max(rho, 1e-6), 1.0 - 1e-6)
        
        tau_max = 1.0 / (float(best_cost) * (1.0 - rho) + 1e-12)
        
        # Candidate-list version: use k as avg
        avg = float(max(self.k, 2))
        p = float(self.p_best) ** (1.0 / avg)
        tau_min = min(tau_max, tau_max * (1.0 - p) / ((avg - 1.0) * p + 1e-12))
        
        return float(tau_min), float(tau_max)

    # ------------------------------------------------------------
    # Pheromone update (elitist best)
    # ------------------------------------------------------------

    def _update_pheromone_from_flat(self, flat: np.ndarray, cost: float):
        """
        MMAS-style pheromone update with trail limits.
        
        Evaporate, clamp to [tau_min, tau_max], deposit, and clamp again.
        Note: tau_min/tau_max are updated in run() when best improves.
        """
        # Evaporation and clamp to bounds
        self.pheromone_sparse.mul_(self.decay)
        self.pheromone_sparse.clamp_(min=self.tau_min, max=self.tau_max)
        
        # Deposit pheromone on best solution edges
        delta = 1.0 / (float(cost) + 1e-12)
        
        u = flat[:-1].astype(np.int64)
        v = flat[1:].astype(np.int64)
        for a, b in zip(u, v):
            j = self.nn_index_map[a, b]
            if j >= 0:
                # Add delta and clamp to tau_max
                new_val = float(self.pheromone_sparse[a, j]) + delta
                self.pheromone_sparse[a, j] = min(new_val, self.tau_max)

    # ------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------

    @torch.no_grad()
    def run(self, n_iterations: int, invtemp: float = 1.0) -> float:
        for _ in range(n_iterations):
            costs, flats, _touched = self.sample(invtemp=invtemp, require_prob=False)

            best_idx = int(costs.argmin())
            best_cost = float(costs[best_idx])
            best_flat = flats[best_idx]

            if best_cost < self.best_cost:
                self.best_cost = best_cost
                self.best_flat = best_flat
                # Recompute MMAS trail limits based on new best cost
                self.tau_min, self.tau_max = self._calc_trail_limits(self.best_cost)

            self._update_pheromone_from_flat(best_flat, best_cost)

        return float(self.best_cost)


# ============================================================
# Minimal test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n = 51  # 0 is depot
    coords = torch.rand(n, 2)
    dist = torch.cdist(coords, coords, p=2)
    demand = torch.zeros(n, dtype=torch.int64)
    demand[1:] = torch.randint(low=1, high=10, size=(n - 1,))

    solver = FACO(
        distances=dist,
        demand=demand,
        n_ants=10,
        k_nearest=20,
        capacity=50,
        min_new_edges=8,
        sample_two_opt=True,
        device="cpu",
    )

    print("Init best:", solver.best_cost)
    best = solver.run(n_iterations=50, invtemp=1.0)
    print("After run:", best)
    print("Best flat:", solver.best_flat)

    # training-mode smoke test (logp + trace)
    costs, flats, touched, logps, traces = solver.sample(invtemp=1.0, require_prob=True)
    print("Train sample costs[0], logp[0]:", costs[0], float(logps[0].detach().cpu()))
    # replay check
    lp_re = solver.replay_logp(traces[0], invtemp=1.0, ref_flat=solver.best_flat)
    print("Replay logp[0]:", float(lp_re.detach().cpu()))

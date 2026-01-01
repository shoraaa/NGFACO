from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import numba as nb
import torch


# =============================================================================
# Numba JIT-compiled helper functions (ported from faco.py for performance)
# =============================================================================

@nb.jit(nb.int64(nb.int64, nb.uint16[:], nb.int64[:]), nopython=True, nogil=True)
def _get_succ(node: int, route: np.ndarray, positions: np.ndarray) -> int:
    """Get the successor of a given node in the route."""
    pos = positions[node]
    n = len(route)
    if pos + 1 == n:
        return route[0]
    return route[pos + 1]


@nb.jit(nb.int64(nb.int64, nb.uint16[:], nb.int64[:]), nopython=True, nogil=True)
def _get_pred(node: int, route: np.ndarray, positions: np.ndarray) -> int:
    """Get the predecessor of a given node in the route."""
    pos = positions[node]
    n = len(route)
    if pos == 0:
        return route[n - 1]
    return route[pos - 1]


@nb.jit(nb.boolean(nb.int64, nb.int64, nb.uint16[:], nb.int64[:]), nopython=True, nogil=True)
def _contains_edge(a: int, b: int, route: np.ndarray, positions: np.ndarray) -> bool:
    """Check if there is an edge (undirected) between nodes a and b in the route."""
    return b == _get_succ(a, route, positions) or b == _get_pred(a, route, positions)


@nb.jit(nb.float64(nb.int64, nb.int64, nb.uint16[:], nb.int64[:], nb.float64, nb.float32[:,:]), nopython=True, nogil=True)
def _relocate_node(target: int, node: int, route: np.ndarray, positions: np.ndarray, 
                   current_cost: float, distances: np.ndarray) -> float:
    """Relocate a node to be the successor of the target node in the route."""
    if node == target or _get_succ(target, route, positions) == node:
        return current_cost
    
    node_pos = positions[node]
    target_pos = positions[target]
    
    node_pred = _get_pred(node, route, positions)
    node_succ = _get_succ(node, route, positions)
    target_succ = _get_succ(target, route, positions)
    
    # Calculate cost delta
    cost_delta = (- distances[node_pred, node]
                  - distances[node, node_succ]
                  - distances[target, target_succ]
                  + distances[node_pred, node_succ]
                  + distances[target, node]
                  + distances[node, target_succ])
    
    if target_pos < node_pos:
        # Case 1: target is before node
        node_value = route[node_pos]
        for i in range(node_pos, target_pos + 1, -1):
            route[i] = route[i - 1]
        route[target_pos + 1] = node_value
        for i in range(target_pos + 1, node_pos + 1):
            positions[route[i]] = i
    else:
        # Case 2: target is after node
        node_value = route[node_pos]
        for i in range(node_pos, target_pos):
            route[i] = route[i + 1]
        route[target_pos] = node_value
        for i in range(node_pos, target_pos + 1):
            positions[route[i]] = i
    
    return current_cost + cost_delta


@nb.jit(nb.float64(nb.uint16[:], nb.float32[:,:]), nopython=True, nogil=True)
def _get_route_cost(route: np.ndarray, distances: np.ndarray) -> float:
    """Calculate the total cost of a route."""
    n = len(route)
    cost = 0.0
    for i in range(n - 1):
        cost += distances[route[i], route[i + 1]]
    cost += distances[route[n - 1], route[0]]
    return cost


@nb.jit(nb.boolean(nb.int64[:], nb.int64), nopython=True, nogil=True)
def _checklist_contains(checklist: np.ndarray, node: int) -> bool:
    """Check if a node exists in the checklist. checklist[0] stores length."""
    length = checklist[0]
    for i in range(1, length + 1):
        if checklist[i] == node:
            return True
    return False


@nb.jit(nb.void(nb.int64[:], nb.int64), nopython=True, nogil=True)
def _checklist_push(checklist: np.ndarray, node: int) -> None:
    """Add a node to the checklist. checklist[0] stores length."""
    length = checklist[0]
    checklist[length + 1] = node
    checklist[0] = length + 1


@nb.jit(nb.int32(nb.int32, nb.int32, nb.uint16[:], nb.int64[:]), nopython=True, nogil=True)
def _flip_route_section(start_node: int, end_node: int, route: np.ndarray, positions: np.ndarray) -> int:
    """Flip a section of the route between start_node and end_node."""
    first = positions[start_node]
    last = positions[end_node]
    
    if first > last:
        temp = first
        first = last
        last = temp
    
    n = len(route)
    segment_length = last - first
    remaining_length = n - segment_length
    
    if segment_length <= remaining_length:
        left = first
        right = last - 1
        while left < right:
            temp = route[left]
            route[left] = route[right]
            route[right] = temp
            left += 1
            right -= 1
        for k in range(first, last):
            positions[route[k]] = k
        return first
    else:
        first_adj = (first - 1) if first > 0 else n - 1
        last_adj = last % n
        temp = first_adj
        first_adj = last_adj
        last_adj = temp
        
        l = first_adj
        r = last_adj
        i = 0
        j = n - first_adj + last_adj + 1
        
        while i < j:
            temp = route[l]
            route[l] = route[r]
            route[r] = temp
            positions[route[l]] = l
            positions[route[r]] = r
            l = (l + 1) % n
            r = (r - 1) if r > 0 else n - 1
            i += 1
            j -= 1
        return 0


@nb.jit(nb.float64(nb.uint16[:], nb.int64[:], nb.float32[:,:], nb.int64[:,:], nb.int64[:], nb.int64, nb.int64), 
        nopython=True, nogil=True)
def _two_opt_nn(route: np.ndarray, positions: np.ndarray, distances: np.ndarray, 
                nn_list: np.ndarray, checklist: np.ndarray, nn_list_size: int, max_changes: int) -> float:
    """2-opt local search using nearest neighbors optimization."""
    n = len(route)
    changes_count = 0
    cost_change = 0.0
    checklist_pos = 1  # Start from index 1 (index 0 is length)
    
    while checklist_pos <= checklist[0] and changes_count < max_changes:
        a = checklist[checklist_pos]
        checklist_pos += 1
        
        if a < 0 or a >= n:
            continue
            
        a_next = _get_succ(a, route, positions)
        a_prev = _get_pred(a, route, positions)
        
        dist_a_to_next = distances[a, a_next]
        dist_a_to_prev = distances[a_prev, a]
        
        max_diff = -1.0
        best_move = np.array([-1, -1, -1, -1], dtype=np.int64)
        
        # Check moves with a -> a_next edge
        for j in range(min(nn_list_size, nn_list.shape[1])):
            b = nn_list[a, j]
            if b < 0 or b >= n:
                break
            dist_ab = distances[a, b]
            if dist_a_to_next > dist_ab:
                b_next = _get_succ(b, route, positions)
                diff = (dist_a_to_next + distances[b, b_next] 
                       - dist_ab - distances[a_next, b_next])
                if diff > max_diff:
                    best_move[0] = a_next
                    best_move[1] = b_next
                    best_move[2] = a
                    best_move[3] = b
                    max_diff = diff
            else:
                break
        
        # Check moves with a_prev -> a edge
        for j in range(min(nn_list_size, nn_list.shape[1])):
            b = nn_list[a, j]
            if b < 0 or b >= n:
                break
            dist_ab = distances[a, b]
            if dist_a_to_prev > dist_ab:
                b_prev = _get_pred(b, route, positions)
                diff = (dist_a_to_prev + distances[b_prev, b]
                       - dist_ab - distances[a_prev, b_prev])
                if diff > max_diff:
                    best_move[0] = a
                    best_move[1] = b
                    best_move[2] = a_prev
                    best_move[3] = b_prev
                    max_diff = diff
            else:
                break
        
        if max_diff > 0:
            _flip_route_section(best_move[0], best_move[1], route, positions)
            changes_count += 1
            cost_change -= max_diff
    
    return cost_change


@nb.jit(nb.int64(nb.int64, nb.uint8[:], nb.float32[:,:], nb.int64[:,:], nb.int64[:,:], nb.int64, nb.int64, nb.float32[:,:]), 
        nopython=True, nogil=True)
def _select_next_node(current_node: int, visited_mask: np.ndarray, probmat_sparse: np.ndarray,
                      nn_list: np.ndarray, backup_nn_list: np.ndarray, k_nearest: int, n: int, 
                      distances: np.ndarray) -> int:
    """Select the next node using sparse probability matrix and nearest neighbor optimization."""
    cl = np.zeros(k_nearest, dtype=np.int64)
    cl_products = np.zeros(k_nearest, dtype=np.float64)
    cl_size = 0
    cl_products_sum = 0.0
    max_product = 0.0
    max_node = current_node
    
    for j in range(k_nearest):
        neighbor = nn_list[current_node, j]
        if neighbor >= 0 and visited_mask[neighbor] == 0:
            product = probmat_sparse[current_node, j]
            cl[cl_size] = neighbor
            cl_products[cl_size] = product
            cl_products_sum += product
            cl_size += 1
            if product > max_product:
                max_product = product
                max_node = neighbor
    
    chosen_node = max_node
    
    if cl_size > 1:
        r = np.random.random() * cl_products_sum
        cumsum = 0.0
        for i in range(cl_size):
            cumsum += cl_products[i]
            if r <= cumsum:
                chosen_node = cl[i]
                break
    elif cl_size == 0:
        chosen_node = current_node
        for i in range(backup_nn_list.shape[1]):
            neighbor = backup_nn_list[current_node, i]
            if neighbor >= 0 and visited_mask[neighbor] == 0:
                chosen_node = neighbor
                break
        if chosen_node == current_node:
            min_distance = np.inf
            for node in range(n):
                if visited_mask[node] == 0:
                    distance = distances[current_node, node]
                    if distance < min_distance:
                        min_distance = distance
                        chosen_node = node
    
    return chosen_node


@nb.jit(nb.int64(nb.int64, nb.uint8[:], nb.int64[:,:], nb.int64[:,:], nb.int64, nb.int64, nb.float32[:,:]), 
        nopython=True, nogil=True)
def _select_next_node_random(current_node: int, visited_mask: np.ndarray,
                              nn_list: np.ndarray, backup_nn_list: np.ndarray, 
                              k_nearest: int, n: int, distances: np.ndarray) -> int:
    """Select the next node uniformly at random from valid candidates."""
    cl = np.zeros(k_nearest, dtype=np.int64)
    cl_size = 0
    
    for j in range(k_nearest):
        neighbor = nn_list[current_node, j]
        if neighbor >= 0 and visited_mask[neighbor] == 0:
            cl[cl_size] = neighbor
            cl_size += 1
    
    if cl_size > 1:
        idx = np.random.randint(0, cl_size)
        return cl[idx]
    elif cl_size == 1:
        return cl[0]
    else:
        # Backup list
        for i in range(backup_nn_list.shape[1]):
            neighbor = backup_nn_list[current_node, i]
            if neighbor >= 0 and visited_mask[neighbor] == 0:
                return neighbor
        # Global fallback
        min_distance = np.inf
        chosen = current_node
        for node in range(n):
            if visited_mask[node] == 0:
                distance = distances[current_node, node]
                if distance < min_distance:
                    min_distance = distance
                    chosen = node
        return chosen


@nb.jit(nb.uint16[:](nb.int64, nb.int64[:,:], nb.float32[:,:], nb.int64), nopython=True, nogil=True)
def _build_nn_tour(start_node: int, nn_list: np.ndarray, distances: np.ndarray, n: int) -> np.ndarray:
    """Build a tour using nearest neighbors with fallback to closest unvisited node."""
    tour = np.zeros(n, dtype=np.uint16)
    visited = np.zeros(n, dtype=np.uint8)
    visited[start_node] = 1
    tour[0] = start_node
    
    for i in range(1, n):
        prev = tour[i - 1]
        next_node = prev
        
        for j in range(nn_list.shape[1]):
            neighbor = nn_list[prev, j]
            if neighbor >= 0 and neighbor < n and visited[neighbor] == 0:
                next_node = neighbor
                break
        
        if next_node == prev:
            min_cost = np.inf
            for node in range(n):
                if visited[node] == 0 and distances[prev, node] < min_cost:
                    min_cost = distances[prev, node]
                    next_node = node
        
        visited[next_node] = 1
        tour[i] = next_node
    
    return tour


@nb.jit(nopython=True, nogil=True)
def _sample_single_ant(probmat_sparse: np.ndarray, nn_list: np.ndarray, backup_nn_list: np.ndarray,
                       start_node: int, distances: np.ndarray, source_route: np.ndarray,
                       source_positions: np.ndarray, min_new_edges: int, random_mode: bool,
                       use_local_search: bool, ls_nn_size: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample a single ant's solution using FACO approach."""
    n = len(source_route)
    k_nearest = nn_list.shape[1]
    
    # Start with a copy of the source route
    route = source_route.copy()
    positions = np.zeros(n, dtype=np.int64)
    for i in range(n):
        positions[route[i]] = i
    
    cost = _get_route_cost(route, distances)
    visited_mask = np.zeros(n, dtype=np.uint8)
    visited_mask[start_node] = 1
    visited_count = 1  # Track count instead of np.sum() for O(1)
    current_node = start_node
    
    # Checklist for local search: index 0 stores length
    checklist = np.full(n + 1, -1, dtype=np.int64)
    checklist[0] = 1  # Length starts at 1
    checklist[1] = start_node
    
    new_edges = 0
    
    while new_edges < min_new_edges and visited_count < n:
        curr = current_node
        
        if random_mode:
            chosen_node = _select_next_node_random(
                current_node, visited_mask, nn_list, backup_nn_list, k_nearest, n, distances
            )
        else:
            chosen_node = _select_next_node(
                current_node, visited_mask, probmat_sparse, nn_list, backup_nn_list, k_nearest, n, distances
            )
        
        visited_mask[chosen_node] = 1
        visited_count += 1
        chos_pred = _get_pred(chosen_node, route, positions)
        
        # Relocate node
        cost = _relocate_node(curr, chosen_node, route, positions, cost, distances)
        
        # Check if this creates a new edge
        if not _contains_edge(curr, chosen_node, source_route, source_positions):
            if not _checklist_contains(checklist, curr):
                _checklist_push(checklist, curr)
            if not _checklist_contains(checklist, chosen_node):
                _checklist_push(checklist, chosen_node)
            if not _checklist_contains(checklist, chos_pred):
                _checklist_push(checklist, chos_pred)
            new_edges += 1
        
        current_node = chosen_node
    
    if use_local_search:
        _two_opt_nn(route, positions, distances, nn_list, checklist, ls_nn_size, n)
    
    # Compute final cost
    final_cost = _get_route_cost(route, distances)
    
    return route, checklist, final_cost


@nb.jit(nopython=True, nogil=True, parallel=True)
def _sample_all_ants_parallel(probmat_sparse: np.ndarray, nn_list: np.ndarray, backup_nn_list: np.ndarray,
                              start_nodes: np.ndarray, distances: np.ndarray, source_route: np.ndarray,
                              source_positions: np.ndarray, min_new_edges: int, random_mode: bool,
                              use_local_search: bool, ls_nn_size: int,
                              routes_out: np.ndarray, costs_out: np.ndarray) -> None:
    """Sample all ants in parallel."""
    n_ants = len(start_nodes)
    n = len(source_route)
    
    for a in nb.prange(n_ants):
        route, checklist, cost = _sample_single_ant(
            probmat_sparse, nn_list, backup_nn_list, start_nodes[a], distances,
            source_route, source_positions, min_new_edges, random_mode,
            use_local_search, ls_nn_size
        )
        routes_out[a, :] = route
        costs_out[a] = cost


@nb.jit(nopython=True, nogil=True)
def _sample_single_ant_traced(probmat_sparse: np.ndarray, nn_list: np.ndarray, backup_nn_list: np.ndarray,
                               start_node: int, distances: np.ndarray, source_route: np.ndarray,
                               source_positions: np.ndarray, min_new_edges: int, random_mode: bool,
                               use_local_search: bool, ls_nn_size: int,
                               curr_nodes_out: np.ndarray, chosen_nodes_out: np.ndarray,
                               is_stochastic_out: np.ndarray, used_uniform_out: np.ndarray,
                               logp_contributions: np.ndarray,
                               checklist_out: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    """
    Sample a single ant's solution with trace recording for training.
    Records ALL relocation steps (not just stochastic ones) for correct replay.
    
    Returns:
        route: final route
        cost: final cost
        n_decisions: number of decisions made (length of trace)
        checklist_len: number of touched nodes in checklist
    """
    n = len(source_route)
    k_nearest = nn_list.shape[1]
    
    # Start with a copy of the source route
    route = source_route.copy()
    positions = np.zeros(n, dtype=np.int64)
    for i in range(n):
        positions[route[i]] = i
    
    cost = _get_route_cost(route, distances)
    visited_mask = np.zeros(n, dtype=np.uint8)
    visited_mask[start_node] = 1
    visited_count = 1  # Track count instead of np.sum() for O(1)
    current_node = start_node
    
    # Checklist for local search
    checklist = np.full(n + 1, -1, dtype=np.int64)
    checklist[0] = 1
    checklist[1] = start_node
    
    new_edges = 0
    n_decisions = 0
    
    while new_edges < min_new_edges and visited_count < n:
        curr = current_node
        
        # Build candidate list (use float64 to prevent underflow)
        cl = np.zeros(k_nearest, dtype=np.int64)
        cl_products = np.zeros(k_nearest, dtype=np.float64)
        cl_size = 0
        cl_products_sum = 0.0
        
        for j in range(k_nearest):
            neighbor = nn_list[current_node, j]
            if neighbor >= 0 and visited_mask[neighbor] == 0:
                # Cast to float64 and clamp to prevent underflow
                # Use 1e-12 to match replay_logp clamp for consistency
                product = max(float(probmat_sparse[current_node, j]), 1e-12)
                cl[cl_size] = neighbor
                cl_products[cl_size] = product
                cl_products_sum += product
                cl_size += 1
        
        chosen_node = current_node
        logp_contrib = 0.0
        is_stochastic = 0  # Use int for Numba compatibility (0=False, 1=True)
        used_uniform = 0   # Track if we used uniform fallback
        
        if cl_size > 1:
            is_stochastic = 1
            if random_mode:
                # Uniform random selection
                idx = np.random.randint(0, cl_size)
                chosen_node = cl[idx]
                logp_contrib = -np.log(float(cl_size))
                used_uniform = 1
            else:
                # Guard against NaN or non-positive sum
                if (not np.isfinite(cl_products_sum)) or cl_products_sum <= 0.0:
                    # Fallback to uniform and record it
                    idx = np.random.randint(0, cl_size)
                    chosen_node = cl[idx]
                    logp_contrib = -np.log(float(cl_size))
                    used_uniform = 1
                else:
                    # Roulette wheel selection
                    r = np.random.random() * cl_products_sum
                    cumsum = 0.0
                    chosen_node = cl[cl_size - 1]  # Default to last in case of rounding
                    for i in range(cl_size):
                        cumsum += cl_products[i]
                        if r <= cumsum:
                            chosen_node = cl[i]
                            break
                    # Compute log prob (use the actual product, clamped)
                    chosen_idx = -1
                    for i in range(cl_size):
                        if cl[i] == chosen_node:
                            chosen_idx = i
                            break
                    prob = cl_products[chosen_idx] / cl_products_sum
                    prob = max(prob, 1e-12)  # Clamp to match replay_logp
                    logp_contrib = np.log(prob)
        elif cl_size == 1:
            chosen_node = cl[0]
            # Deterministic: logp_contrib stays 0
        else:
            # Backup list
            for i in range(backup_nn_list.shape[1]):
                neighbor = backup_nn_list[current_node, i]
                if neighbor >= 0 and visited_mask[neighbor] == 0:
                    chosen_node = neighbor
                    break
            if chosen_node == current_node:
                # Global fallback
                min_distance = np.inf
                for node in range(n):
                    if visited_mask[node] == 0:
                        distance = distances[current_node, node]
                        if distance < min_distance:
                            min_distance = distance
                            chosen_node = node
        
        # Record ALL steps for correct replay (not just stochastic ones)
        curr_nodes_out[n_decisions] = curr
        chosen_nodes_out[n_decisions] = chosen_node
        is_stochastic_out[n_decisions] = is_stochastic
        used_uniform_out[n_decisions] = used_uniform
        logp_contributions[n_decisions] = logp_contrib
        n_decisions += 1
        
        visited_mask[chosen_node] = 1
        visited_count += 1
        chos_pred = _get_pred(chosen_node, route, positions)
        
        # Relocate node using Numba function
        cost = _relocate_node(curr, chosen_node, route, positions, cost, distances)
        
        # Check if this creates a new edge
        if not _contains_edge(curr, chosen_node, source_route, source_positions):
            if not _checklist_contains(checklist, curr):
                _checklist_push(checklist, curr)
            if not _checklist_contains(checklist, chosen_node):
                _checklist_push(checklist, chosen_node)
            if not _checklist_contains(checklist, chos_pred):
                _checklist_push(checklist, chos_pred)
            new_edges += 1
        
        current_node = chosen_node
    
    if use_local_search:
        _two_opt_nn(route, positions, distances, nn_list, checklist, ls_nn_size, n)
    
    final_cost = _get_route_cost(route, distances)
    
    # Copy checklist to output
    checklist_len = checklist[0]
    for i in range(checklist_len):
        checklist_out[i] = checklist[i + 1]
    
    return route, final_cost, n_decisions, checklist_len


# =============================================================================
# Dataclass and pure-Python functions for training compatibility
# =============================================================================


@dataclass
class MFACOTrace:
    """Stores all relocation decisions to replay log-prob correctly."""
    start_node: int
    curr_nodes: List[int]
    chosen_nodes: List[int]
    is_stochastic: List[bool]  # True if decision had multiple valid candidates
    used_uniform_fallback: List[bool]  # True if underflow caused uniform fallback (for replay matching)


def tsp_cost_from_cycle(cycle: np.ndarray, dist: np.ndarray) -> float:
    """cycle: shape (n+1,), last equals first."""
    return float(dist[cycle[:-1], cycle[1:]].sum())


def build_nn_and_backup_lists(dist: np.ndarray, cl_size: int, bl_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      nn_list: (n, cl_size) nearest neighbors per node (excluding self)
      backup_list: (n, bl_size) next nearest after candidate list
    """
    n = dist.shape[0]
    nn_list = np.empty((n, cl_size), dtype=np.int64)
    backup_list = np.empty((n, bl_size), dtype=np.int64) if bl_size > 0 else np.empty((n, 0), dtype=np.int64)

    for u in range(n):
        order = np.argsort(dist[u], kind="stable")
        order = order[order != u]
        need = cl_size + bl_size
        picks = order[:need]
        nn_list[u] = picks[:cl_size]
        if bl_size > 0:
            backup_list[u] = picks[cl_size:cl_size + bl_size]
    return nn_list, backup_list


def edges_undirected_set(route: np.ndarray) -> set[tuple[int, int]]:
    """
    route: (n,) cycle order without repetition.
    Returns undirected edges set with canonical ordering (min, max).
    """
    n = route.shape[0]
    s = set()
    prev = int(route[-1])
    for i in range(n):
        cur = int(route[i])
        a, b = (prev, cur) if prev < cur else (cur, prev)
        s.add((a, b))
        prev = cur
    return s


class MFACO_TSP:
    """
    MFACO for TSP, designed to integrate with REINFORCE/A2C training:
      - candidate-list pheromone (sparse)
      - relocate-based focused editing from a source solution
      - checklist-guided deterministic local 2-opt (optional)
      - replayable log-prob from stored traces
    
    Training usage note:
      replay_logp() uses current pheromone state, so compute logp BEFORE
      calling _update_pheromone_from_flat() to ensure consistency.
    """

    def __init__(
        self,
        distances: torch.Tensor,             # (n,n)
        n_ants: int,
        cand_list_size: int = 32,
        backup_list_size: int = 32,
        min_new_edges: int = 8,
        decay: float = 0.9,                  # rho in paper
        alpha: float = 1.0,                  # pheromone exponent
        p_best: float = 0.05,                # for tau limits
        gbest_as_source_prob: float = 1.0,   # always use best as update source by default
        use_local_search: bool = True,
        random_mode: bool = False,           # if True, use uniform random selection instead of roulette
        enable_torch_sync: bool = True,      # if False, skip sync_pheromone_to_torch (faster baseline)
        device: str = "cuda",
    ):
        self.device = device
        self.distances = distances.to(device)
        self.n = int(distances.size(0))
        self.n_ants = int(n_ants)

        self.k = int(min(cand_list_size, self.n - 1))
        self.bl = int(min(backup_list_size, max(0, self.n - 1 - self.k)))
        self.min_new_edges = int(min_new_edges)

        self.rho = float(decay)
        self.alpha = float(alpha)
        self.p_best = float(p_best)
        self.gbest_as_source_prob = float(gbest_as_source_prob)

        self.use_local_search = bool(use_local_search)
        self.random_mode = bool(random_mode)
        self.enable_torch_sync = bool(enable_torch_sync)

        # Precompute NN and backup lists on CPU (stable, deterministic)
        dist_np = self.distances.detach().cpu().numpy()
        self.nn_list, self.backup_list = build_nn_and_backup_lists(dist_np, self.k, self.bl)

        # Mapping for fast pheromone updates: nn_pos[u, v] -> index in nn_list[u] or -1
        self.nn_pos = -np.ones((self.n, self.n), dtype=np.int32)
        for u in range(self.n):
            for j, v in enumerate(self.nn_list[u]):
                self.nn_pos[u, int(v)] = j

        # Initialize a source solution using a greedy NN tour (multiple starts, pick best)
        best_route = None
        best_cost = float("inf")
        for s in range(min(8, self.n)):
            start = s
            route = self._build_nn_tour(start, dist_np)
            cyc = np.concatenate([route, route[:1]])
            c = tsp_cost_from_cycle(cyc, dist_np)
            if c < best_cost:
                best_cost = c
                best_route = route

        assert best_route is not None
        self.source_route = best_route.copy()                 # (n,)
        self.source_cost = float(best_cost)
        self.best_route = best_route.copy()
        self.best_cost = float(best_cost)

        # Cache numpy distance matrix (avoid repeated GPU->CPU copies)
        self.dist_np32 = dist_np.astype(np.float32)
        
        # Cache source positions array
        self.source_positions = np.zeros(self.n, dtype=np.int64)
        for i, v in enumerate(self.source_route):
            self.source_positions[int(v)] = i

        # Trail limits and pheromone init (CandListModel with all trails set to tau_max)
        self.tau_min, self.tau_max = self._calc_trail_limits_cl(self.source_cost)
        # Primary storage in numpy for fast inference path
        self.pheromone_sparse_np = np.full((self.n, self.k), self.tau_max, dtype=np.float32)
        # Build sparse heuristic from distances (1/dist for candidate edges)
        self.heuristic_sparse_np = np.zeros((self.n, self.k), dtype=np.float32)
        for u in range(self.n):
            for j, v in enumerate(self.nn_list[u]):
                self.heuristic_sparse_np[u, j] = 1.0 / (self.dist_np32[u, int(v)] + 1e-10)
        # Keep torch tensors for training path (backward compat)
        self.pheromone_sparse = torch.full((self.n, self.k), self.tau_max, device=self.device, dtype=torch.float32)
        
        # Cache torch tensors for replay_logp (avoid recomputation)
        self.nn_torch = torch.as_tensor(self.nn_list, device=self.device, dtype=torch.long)
        self.h_sparse_torch = torch.as_tensor(self.heuristic_sparse_np, device=self.device, dtype=torch.float32)
        
        # Pre-allocate trace buffers for _sample_with_trace (reused across calls)
        self._trace_curr_nodes = np.zeros((self.n_ants, self.n), dtype=np.int64)
        self._trace_chosen_nodes = np.zeros((self.n_ants, self.n), dtype=np.int64)
        self._trace_is_stochastic = np.zeros((self.n_ants, self.n), dtype=np.int64)
        self._trace_used_uniform = np.zeros((self.n_ants, self.n), dtype=np.int64)
        self._trace_logp_contribs = np.zeros((self.n_ants, self.n), dtype=np.float64)
        self._trace_checklist = np.zeros((self.n_ants, self.n), dtype=np.int64)

        # ants memory (mfaco starts ants from best in C++; we keep costs/routes)
        self.ant_routes = [self.best_route.copy() for _ in range(self.n_ants)]
        self.ant_costs = np.array([self.best_cost] * self.n_ants, dtype=np.float32)

        # Note: Use seed_rng() to control global np.random for reproducibility
        # Numba uses np.random internally, so we control it via np.random.seed()

    # ---------- init helpers ----------

    def _build_nn_tour(self, start: int, dist: np.ndarray) -> np.ndarray:
        n = self.n
        visited = np.zeros(n, dtype=np.bool_)
        route = np.empty(n, dtype=np.int64)
        cur = int(start)
        route[0] = cur
        visited[cur] = True
        for i in range(1, n):
            # nearest unvisited (deterministic tie-break by stable argsort)
            order = np.argsort(dist[cur], kind="stable")
            for v in order:
                if not visited[v]:
                    cur = int(v)
                    route[i] = cur
                    visited[cur] = True
                    break
        return route

    def _calc_trail_limits_cl(self, solution_cost: float) -> Tuple[float, float]:
        # calc_trail_limits_cl from C++ (avg=cand_list_size)
        tau_max = 1.0 / (solution_cost * (1.0 - self.rho) + 1e-12)
        avg = float(max(2, self.k))
        p = float(self.p_best ** (1.0 / avg))
        tau_min = min(tau_max, tau_max * (1.0 - p) / ((avg - 1.0) * p + 1e-12))
        return float(tau_min), float(tau_max)

    # ---------- API used by training ----------

    def prob_sparse_torch(self, invtemp: float = 1.0, residual_logits: torch.Tensor = None) -> torch.Tensor:
        """
        Returns unnormalized weights for candidate edges: shape (n,k).
        weight[u, j] corresponds to edge (u -> nn_list[u, j]).
        
        Matches numpy sampling: w = tau^alpha * h^invtemp * exp(residual)
        
        Args:
            invtemp: inverse temperature (beta) applied to heuristic only
            residual_logits: optional (n, k) tensor of neural residual logits
        """
        tau = self.pheromone_sparse.clamp_min(1e-10) ** self.alpha

        # Use cached heuristic (matches numpy heuristic_sparse_np exactly)
        h = self.h_sparse_torch

        # Apply invtemp to heuristic only (matches numpy sampling)
        if invtemp != 1.0:
            h = h ** float(invtemp)
        
        w = tau * h
        w = w.clamp_min(1e-12)
        
        # Apply neural residual if provided (additive in log-space)
        if residual_logits is not None:
            # Clamp to prevent overflow
            residual_clamped = residual_logits.clamp(-10.0, 10.0)
            w = w * torch.exp(residual_clamped)
        
        return w

    def seed_rng(self, seed: int) -> None:
        """
        Seed the global numpy RNG used by Numba.
        Call this before sampling for reproducibility or paired BvB rollouts.
        
        NOTE: For _sample_fast() (parallel via prange), RNG call ordering may differ
        across runs even with the same seed due to thread scheduling. For perfect
        determinism in BvB training, use require_prob=True (sequential traced sampling)
        for both baseline and neural rollouts.
        """
        np.random.seed(seed)

    def compute_probmat_sparse(self, invtemp: float = 1.0, residual_logits: np.ndarray = None) -> np.ndarray:
        """
        Compute the sparse probability matrix for candidate edges.
        
        Args:
            invtemp: inverse temperature (beta) for heuristic
            residual_logits: optional (n, k) array of neural residual logits to add
                            before exponentiation. Use this to inject learned biases.
        
        Returns:
            probmat_sparse: (n, k) unnormalized weights for candidate edges
        """
        # Base: tau^alpha * eta^beta
        probmat_sparse = (self.pheromone_sparse_np ** self.alpha) * (self.heuristic_sparse_np ** invtemp)
        
        # Apply neural residual if provided
        if residual_logits is not None:
            # Clamp to prevent overflow (exp(10) ≈ 22000, exp(-10) ≈ 0.00005)
            residual_clamped = np.clip(residual_logits.astype(np.float32), -10.0, 10.0)
            probmat_sparse = probmat_sparse * np.exp(residual_clamped)
        probmat_sparse = np.maximum(probmat_sparse, 1e-12)
        return probmat_sparse

    def sample(self, invtemp: float = 1.0, require_prob: bool = False, 
               residual_logits: np.ndarray = None):
        """
        Sample solutions from all ants.
        
        If require_prob=False, uses fast Numba-parallel implementation.
        If require_prob=True, uses slower Python implementation that tracks traces.
        
        Args:
            invtemp: inverse temperature for heuristic
            require_prob: if True, return log-probs and traces for training
            residual_logits: optional (n, k) neural residual logits for learned biases
        
        Returns:
          costs_np: (n_ants,)
          flats: list[np.ndarray] each shape (n+1,) cycle, last==first
          touched_list: list[np.ndarray] touched nodes for LS guidance (checklist)
          logps: torch.Tensor (n_ants,) if require_prob else None
              NOTE: These are NOT differentiable (float values from Numba).
              For training, use replay_logp(trace, residual_logits=...) to get gradients.
          traces: list[MFACOTrace] if require_prob else None
        """
        probmat_sparse = self.compute_probmat_sparse(invtemp, residual_logits)
        
        if require_prob:
            # Use slow path with tracing for training
            return self._sample_with_trace(invtemp=invtemp, probmat_sparse=probmat_sparse)
        else:
            # Use fast Numba-parallel path for inference
            return self._sample_fast(invtemp=invtemp, probmat_sparse=probmat_sparse)

    def _sample_fast(self, invtemp: float = 1.0, probmat_sparse: np.ndarray = None):
        """Fast Numba-parallel sampling without probability tracking."""
        # Compute probability matrix if not provided
        if probmat_sparse is None:
            probmat_sparse = self.compute_probmat_sparse(invtemp)
        
        source_route = self.source_route.astype(np.uint16)
        
        # Generate random start nodes using global numpy RNG (controlled by seed_rng)
        start_nodes = np.random.randint(0, self.n, size=self.n_ants, dtype=np.int64)
        
        # Output arrays
        routes_out = np.zeros((self.n_ants, self.n), dtype=np.uint16)
        costs_out = np.zeros(self.n_ants, dtype=np.float32)
        
        # Call parallel numba function
        _sample_all_ants_parallel(
            probmat_sparse, self.nn_list, self.backup_list,
            start_nodes, self.dist_np32, source_route, self.source_positions,
            self.min_new_edges, self.random_mode, self.use_local_search, self.k,
            routes_out, costs_out
        )
        
        # Convert to expected output format
        flats = []
        touched_list = []
        for a in range(self.n_ants):
            route = routes_out[a].astype(np.int64)
            flat = np.concatenate([route, route[:1]])
            flats.append(flat)
            touched_list.append(np.array([], dtype=np.int64))  # Not tracked in fast mode
        
        return costs_out, flats, touched_list, None, None

    def _sample_with_trace(self, invtemp: float = 1.0, probmat_sparse: np.ndarray = None):
        """Fast Numba sampling with trace tracking for training (require_prob=True)."""
        # Compute probability matrix if not provided
        if probmat_sparse is None:
            probmat_sparse = self.compute_probmat_sparse(invtemp)
        
        source_route = self.source_route.astype(np.uint16)

        flats: List[np.ndarray] = []
        touched_list: List[np.ndarray] = []
        traces: List[MFACOTrace] = []
        logps = torch.zeros(self.n_ants, device=self.device, dtype=torch.float32)
        costs = np.empty(self.n_ants, dtype=np.float32)
        
        for a in range(self.n_ants):
            # Generate random start using global numpy RNG (controlled by seed_rng)
            start = int(np.random.randint(0, self.n))
            
            # Use pre-allocated buffers (slices for this ant)
            curr_nodes_out = self._trace_curr_nodes[a]
            chosen_nodes_out = self._trace_chosen_nodes[a]
            is_stochastic_out = self._trace_is_stochastic[a]
            used_uniform_out = self._trace_used_uniform[a]
            logp_contributions = self._trace_logp_contribs[a]
            checklist_out = self._trace_checklist[a]
            
            # Call Numba traced sampling function
            route, cost, n_decisions, checklist_len = _sample_single_ant_traced(
                probmat_sparse, self.nn_list, self.backup_list,
                start, self.dist_np32, source_route, self.source_positions,
                self.min_new_edges, self.random_mode, self.use_local_search, self.k,
                curr_nodes_out, chosen_nodes_out, is_stochastic_out, used_uniform_out,
                logp_contributions, checklist_out
            )
            
            # Finalize outputs
            flat = np.concatenate([route.astype(np.int64), route[:1].astype(np.int64)])
            flats.append(flat)
            touched_list.append(checklist_out[:checklist_len].copy())
            costs[a] = float(cost)
            
            # Sum up log probabilities (only stochastic decisions contribute)
            total_logp = float(np.sum(logp_contributions[:n_decisions]))
            logps[a] = total_logp
            
            # Build trace for replay (ALL steps recorded for correct visited reconstruction)
            traces.append(MFACOTrace(
                start_node=start,
                curr_nodes=curr_nodes_out[:n_decisions].tolist(),
                chosen_nodes=chosen_nodes_out[:n_decisions].tolist(),
                is_stochastic=[bool(is_stochastic_out[i]) for i in range(n_decisions)],
                used_uniform_fallback=[bool(used_uniform_out[i]) for i in range(n_decisions)]
            ))

        return costs, flats, touched_list, logps, traces

    def replay_logp(
        self,
        trace: MFACOTrace,
        invtemp: float,
        ref_flat: Optional[np.ndarray] = None,
        prob_sparse: Optional[torch.Tensor] = None,
        residual_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Recompute log-prob for the stored trace using current prob_sparse (weights).
        
        Uses is_stochastic flags to determine which steps contribute to logp.
        All steps are replayed to correctly reconstruct visited state.
        
        IMPORTANT: Call this BEFORE _update_pheromone_from_flat() to ensure
        pheromone state matches what was used during sampling.
        
        Args:
            trace: MFACOTrace with recorded decisions
            invtemp: inverse temperature (beta) for heuristic
            ref_flat: (unused) reference solution for compatibility
            prob_sparse: optional pre-computed (n, k) weights tensor
            residual_logits: optional (n, k) neural residual logits (carries grad for training)
        
        Returns:
            logp: scalar tensor (differentiable if residual_logits has grad)
        """
        if prob_sparse is None:
            prob_sparse = self.prob_sparse_torch(invtemp=invtemp, residual_logits=residual_logits)
        elif residual_logits is not None:
            # Apply residual to provided prob_sparse
            residual_clamped = residual_logits.clamp(-10.0, 10.0)
            prob_sparse = prob_sparse * torch.exp(residual_clamped)

        visited = np.zeros(self.n, dtype=np.bool_)
        visited[trace.start_node] = True

        logp = torch.zeros((), device=self.device, dtype=torch.float32)
        
        for i, (curr, chosen) in enumerate(zip(trace.curr_nodes, trace.chosen_nodes)):
            # Only compute logp for stochastic decisions
            if trace.is_stochastic[i]:
                nn = self.nn_list[curr]
                w_row = prob_sparse[curr]  # (k,)

                # Defensive: also check v >= 0 in case nn_list ever has padding
                valid = np.array([(v >= 0) and (not visited[int(v)]) for v in nn], dtype=np.bool_)
                valid_idx = np.nonzero(valid)[0]
                m = int(valid_idx.size)

                # Strict check: if trace says stochastic but we see m <= 1, trace is inconsistent
                if m <= 1:
                    raise ValueError(
                        f"Replay mismatch at step {i}: trace marked stochastic but only {m} valid candidates. "
                        f"curr={curr}, chosen={chosen}. This indicates trace/pheromone state mismatch."
                    )

                # find chosen position in nn list
                pick = -1
                for j, v in enumerate(nn):
                    if int(v) == int(chosen):
                        pick = j
                        break
                if pick < 0 or not valid[pick]:
                    raise ValueError(f"Replay mismatch: chosen {chosen} not valid from curr {curr}")

                # Check if sampling used uniform fallback (recorded in trace)
                if self.random_mode or trace.used_uniform_fallback[i]:
                    # Uniform probability: 1/m
                    logp = logp + torch.log(torch.tensor(1.0 / m, device=self.device, dtype=torch.float32))
                else:
                    # Roulette probability from weights
                    w_valid = w_row[torch.as_tensor(valid_idx, device=self.device)]
                    s = w_valid.sum().clamp_min(1e-12)
                    p = (w_row[pick] / s).clamp_min(1e-12)
                    logp = logp + torch.log(p)

            # Always update visited for correct state reconstruction
            visited[int(chosen)] = True

        return logp

    def replay_logp_batch(
        self,
        traces: List[MFACOTrace],
        invtemp: float = 1.0,
        prob_sparse: Optional[torch.Tensor] = None,
        residual_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Vectorized batch replay of log-probabilities for multiple traces.
        
        Much faster than calling replay_logp() in a loop - batches all stochastic
        decisions across all traces into single tensor operations.
        
        Args:
            traces: list of MFACOTrace objects from sampling
            invtemp: inverse temperature (beta) for heuristic
            prob_sparse: optional pre-computed (n, k) weights tensor
            residual_logits: optional (n, k) neural residual logits (carries grad)
        
        Returns:
            total_logp: scalar tensor (sum of all log-probs, differentiable)
            n_stochastic: total count of stochastic decisions
        """
        if prob_sparse is None:
            prob_sparse = self.prob_sparse_torch(invtemp=invtemp, residual_logits=residual_logits)
        elif residual_logits is not None:
            residual_clamped = residual_logits.clamp(-10.0, 10.0)
            prob_sparse = prob_sparse * torch.exp(residual_clamped)
        
        # Collect all stochastic decisions across all traces
        # We need: curr_node, chosen_node, valid_mask at decision time
        all_curr = []
        all_pick_j = []  # index in nn_list
        all_valid_masks = []  # (k,) boolean mask per decision
        
        for trace in traces:
            visited = np.zeros(self.n, dtype=np.bool_)
            visited[trace.start_node] = True
            
            for i, (curr, chosen) in enumerate(zip(trace.curr_nodes, trace.chosen_nodes)):
                if trace.is_stochastic[i] and not trace.used_uniform_fallback[i]:
                    nn = self.nn_list[curr]
                    valid = np.array([(v >= 0) and (not visited[int(v)]) for v in nn], dtype=np.bool_)
                    
                    # Find chosen position in nn list
                    pick = -1
                    for j, v in enumerate(nn):
                        if int(v) == int(chosen):
                            pick = j
                            break
                    
                    if pick >= 0 and valid[pick]:
                        all_curr.append(curr)
                        all_pick_j.append(pick)
                        all_valid_masks.append(valid)
                
                visited[int(chosen)] = True
        
        n_stochastic = len(all_curr)
        
        if n_stochastic == 0:
            return torch.zeros((), device=self.device, dtype=torch.float32), 0
        
        # Convert to tensors
        curr_t = torch.tensor(all_curr, device=self.device, dtype=torch.long)
        pick_t = torch.tensor(all_pick_j, device=self.device, dtype=torch.long)
        valid_t = torch.tensor(np.array(all_valid_masks), device=self.device, dtype=torch.bool)
        
        # Gather weights for all decisions: (n_stochastic, k)
        w_rows = prob_sparse[curr_t]  # (n_stochastic, k)
        
        # Mask invalid candidates
        w_masked = w_rows * valid_t.float()  # (n_stochastic, k)
        
        # Compute normalization sums
        sums = w_masked.sum(dim=1).clamp_min(1e-12)  # (n_stochastic,)
        
        # Gather picked weights
        w_picked = w_rows[torch.arange(n_stochastic, device=self.device), pick_t]  # (n_stochastic,)
        
        # Compute log probabilities
        probs = (w_picked / sums).clamp_min(1e-12)
        log_probs = torch.log(probs)
        
        # Count uniform fallbacks for proper total
        n_uniform = sum(
            sum(trace.used_uniform_fallback[i] for i, s in enumerate(trace.is_stochastic) if s)
            for trace in traces
        )
        total_stochastic = n_stochastic + n_uniform
        
        # Handle uniform fallback decisions (constant log-prob, no gradient)
        # These don't contribute to gradient, just to the count
        uniform_logp = torch.zeros((), device=self.device, dtype=torch.float32)
        for trace in traces:
            visited = np.zeros(self.n, dtype=np.bool_)
            visited[trace.start_node] = True
            for i, (curr, chosen) in enumerate(zip(trace.curr_nodes, trace.chosen_nodes)):
                if trace.is_stochastic[i] and trace.used_uniform_fallback[i]:
                    nn = self.nn_list[curr]
                    valid = [(v >= 0) and (not visited[int(v)]) for v in nn]
                    m = sum(valid)
                    if m > 0:
                        uniform_logp = uniform_logp + np.log(1.0 / m)
                visited[int(chosen)] = True
        
        total_logp = log_probs.sum() + uniform_logp
        
        return total_logp, total_stochastic

    def _update_pheromone_from_flat(self, best_flat: np.ndarray, best_cost: float) -> None:
        """
        Environment update: evaporate + deposit on candidate edges + update source solution + update tau limits.
        best_flat: (n+1,) last==first
        """
        # Update global best & source
        if best_cost < self.best_cost:
            self.best_cost = float(best_cost)
            self.best_route = best_flat[:-1].copy()

        # Update trail limits based on GLOBAL best (MMAS-style), not iteration best
        self.tau_min, self.tau_max = self._calc_trail_limits_cl(self.best_cost)

        # Evaporate (numpy for fast path)
        self.pheromone_sparse_np *= (1.0 - self.rho)
        np.clip(self.pheromone_sparse_np, self.tau_min, self.tau_max, out=self.pheromone_sparse_np)

        # Deposit (like model.deposit_pheromone in C++)
        dep = 1.0 / (float(best_cost) + 1e-12)
        route = best_flat[:-1]
        prev = int(route[-1])
        for cur in route:
            cur = int(cur)
            j = int(self.nn_pos[prev, cur])
            if j >= 0:
                self.pheromone_sparse_np[prev, j] = min(self.pheromone_sparse_np[prev, j] + dep, self.tau_max)
            # symmetric: also update reverse if present
            jr = int(self.nn_pos[cur, prev])
            if jr >= 0:
                self.pheromone_sparse_np[cur, jr] = min(self.pheromone_sparse_np[cur, jr] + dep, self.tau_max)
            prev = cur

        # Update source solution (mfaco does this every iteration)
        self.source_route = route.copy()
        self.source_cost = float(best_cost)
        # Update cached source_positions
        for i, v in enumerate(self.source_route):
            self.source_positions[int(v)] = i
        
        # Sync torch pheromone for replay_logp / prob_sparse_torch compatibility
        # Skip if enable_torch_sync=False (e.g., during baseline cache generation)
        if self.enable_torch_sync:
            self.sync_pheromone_to_torch()

    # ---------- Snapshot save/load for BvB training ----------

    def sync_pheromone_to_torch(self) -> None:
        """Sync numpy pheromone to torch tensor (for replay_logp / prob_sparse_torch)."""
        # Use as_tensor to handle both CPU and CUDA devices correctly
        self.pheromone_sparse.copy_(torch.as_tensor(self.pheromone_sparse_np, device=self.device))

    def reset_pheromone(self) -> None:
        """Reset pheromone trails to tau_max (useful after loading snapshot)."""
        self.pheromone_sparse_np.fill(self.tau_max)
        self.sync_pheromone_to_torch()

    def save_snapshot(self) -> dict:
        """
        Save current state for later restoration (e.g., for BvB training).
        
        Returns:
            snapshot: dict with all state needed to restore the solver.
        """
        return {
            # Core state
            'source_route': self.source_route.copy(),
            'source_cost': self.source_cost,
            'best_route': self.best_route.copy(),
            'best_cost': self.best_cost,
            'pheromone_sparse_np': self.pheromone_sparse_np.copy(),
            'tau_min': self.tau_min,
            'tau_max': self.tau_max,
            # Config (for validation on load)
            'n': self.n,
            'k': self.k,
            'bl': self.bl,
            'rho': self.rho,
            'alpha': self.alpha,
            'p_best': self.p_best,
            'min_new_edges': self.min_new_edges,
            'random_mode': self.random_mode,
            'use_local_search': self.use_local_search,
        }

    def load_snapshot(self, snapshot: dict, validate: bool = True) -> None:
        """
        Restore state from a saved snapshot.
        
        Args:
            snapshot: dict from save_snapshot()
            validate: if True, raise error on config mismatch
        """
        # Validate config if requested
        if validate:
            mismatches = []
            # Integer/bool keys: exact match
            for key in ['n', 'k', 'bl', 'min_new_edges', 'random_mode', 'use_local_search']:
                if key in snapshot and getattr(self, key) != snapshot[key]:
                    mismatches.append(f"{key}: {getattr(self, key)} != {snapshot[key]}")
            # Float keys: use tolerance
            for key in ['rho', 'alpha', 'p_best']:
                if key in snapshot and not np.isclose(getattr(self, key), snapshot[key], rtol=1e-6):
                    mismatches.append(f"{key}: {getattr(self, key)} != {snapshot[key]}")
            if mismatches:
                raise ValueError(f"Snapshot config mismatch: {', '.join(mismatches)}")
        
        # Restore core state
        self.source_route = snapshot['source_route'].copy()
        self.source_cost = snapshot['source_cost']
        self.best_route = snapshot['best_route'].copy()
        self.best_cost = snapshot['best_cost']
        self.pheromone_sparse_np = snapshot['pheromone_sparse_np'].copy()
        self.tau_min = snapshot['tau_min']
        self.tau_max = snapshot['tau_max']
        
        # Rebuild cached positions
        for i, v in enumerate(self.source_route):
            self.source_positions[int(v)] = i
        
        # Sync torch pheromone
        self.sync_pheromone_to_torch()

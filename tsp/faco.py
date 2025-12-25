import concurrent.futures
import os
import random
import time
from datetime import datetime
from functools import cached_property
from itertools import combinations

import numpy as np
import numba as nb
import torch
from torch.distributions import Categorical

from two_opt import batched_two_opt_python

MIN_NEW_EDGES = 8
SOURCE_SOL_LOCAL_UPDATE = True
KEEP_BETTER_ANT_SOL = True
SAMPLE_TWO_OPT = True

class FACO():
    def __init__(
        self, 
        distances: torch.Tensor,
        n_ants=20,
        heuristic: torch.Tensor | None = None,
        k_sparse=None,
        k_nearest=20,  # Added for enhanced optimization
        pheromone: torch.Tensor | None = None,
        ants_route: torch.Tensor | None = None,
        ants_cost: torch.Tensor | None = None,
        best_ant_route: torch.Tensor | None = None,
        best_ant_cost: float | None = None,
        sample_two_opt: bool | None = SAMPLE_TWO_OPT,
        decay=0.9,
        alpha=1,
        beta=1,
        p_best = 0.1, 
        # AS variants
        elitist=False,
        maxmin=False,
        rank_based=False,
        n_elites=None,
        smoothing=False,
        smoothing_thres=5,
        smoothing_delta=0.5,
        shift_cost=True,
        local_search_type: str | None = None,
        trail_limit_factor=5.0,
        device='cpu',
    ):
        self.problem_size = len(distances)
        self.distances = distances.to(device)
        self.n_ants = n_ants
        self.k_nearest = min(k_nearest, self.problem_size - 1)  # Added for optimization
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.p_best = p_best  # New parameter for maxmin variant
        self.elitist = elitist or maxmin  # maxmin uses elitist
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites or n_ants // 10  # only for rank-based
        self.sample_two_opt = sample_two_opt
        self.smoothing = smoothing
        self.smoothing_cnt = 0
        self.smoothing_thres = smoothing_thres
        self.smoothing_delta = smoothing_delta
        self.shift_cost = shift_cost
        assert local_search_type in [None, "2opt", "nls"]
        self.local_search_type = local_search_type
        self.trail_limit_factor = trail_limit_factor
        self.device = device
        
        if k_sparse is None:
            self.k_sparse = self.k_nearest
        else:
            self.k_nearest = k_sparse
            self.k_sparse = k_sparse

        # Build nearest neighbor structures first for sparse matrices
        self._build_neighbor_structures()

        if pheromone is None:
            # Initialize sparse pheromone matrix (n x k_nearest)
            self.pheromone_sparse = torch.ones((self.problem_size, self.k_nearest), device=device)
        else:
            if pheromone.shape == (self.problem_size, self.k_nearest):
                self.pheromone_sparse = pheromone.to(device)
            else:
                # Convert full matrix to sparse if needed
                self.pheromone_sparse = self._compress_full_to_sparse(pheromone.to(device))

        if heuristic is None:
            assert self.k_sparse is not None
            # Use legacy sparse heuristic for backward compatibility
            full_heuristic = self.simple_heuristic(distances, self.k_sparse)
            self.heuristic = full_heuristic.to(device)
            self.heuristic_sparse = self._compress_full_to_sparse(full_heuristic)
        else:
            self.heuristic = heuristic.to(device)
            if heuristic.shape == (self.problem_size, self.k_nearest):
                self.heuristic_sparse = heuristic.to(device)
            else:
                self.heuristic_sparse = self._compress_full_to_sparse(heuristic.to(device))

        initial_routes = build_multiple_nn_tours_parallel(
            os.cpu_count() or 4, self.nn_list, self.distances_numpy, self.problem_size
        )
        initial_costs = self.gen_path_costs_init(initial_routes)
        initial_routes = torch.from_numpy(initial_routes.astype(np.int64)).to(device)

        if self.local_search_type == "nls":
            initial_routes = self.nls_init(initial_routes, inference=True)
        # Find the best initial route
        best_initial_idx = torch.argmin(initial_costs)
        best_initial_route = initial_routes[best_initial_idx]
        # Convert best initial route to torch tensor for compatibility
        self.shortest_path = best_initial_route.clone()
        self.lowest_cost = initial_costs[best_initial_idx]
        # print(f"Initial cost: {self.lowest_cost:.2f}")

        if ants_route is not None and ants_cost is not None:
            assert ants_route.shape == (self.problem_size, n_ants)
            assert ants_cost.shape == (n_ants,)
            self.ants_route = ants_route.to(device)
            self.ants_cost = ants_cost.to(device)
        else:
            # Initialize ants' routes and costs
            self.ants_route = torch.zeros((self.problem_size, n_ants), dtype=torch.int64, device=device)
            self.ants_cost = torch.zeros(n_ants, dtype=torch.float32, device=device)

            # Fill ants' routes with initial low-cost random paths
            for i in range(n_ants):
                self.ants_route[:, i] = self.shortest_path.clone()
                self.ants_cost[i] = float(self.lowest_cost)
        
        if best_ant_route is not None and best_ant_cost is not None:
            assert best_ant_route.shape == (self.problem_size,)
            self.best_ant_route = best_ant_route.to(device)
            self.best_ant_cost = best_ant_cost.to(device)
        else:
            # Initialize best ant with the shortest path
            self.best_ant_route = self.shortest_path.clone()
            self.best_ant_cost = float(self.lowest_cost)
    
    @torch.no_grad()
    def simple_heuristic(self, distances, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        _, topk_indices = torch.topk(distances, k=k_sparse, dim=1, largest=False)
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(distances), device=self.device), repeats=k_sparse
        )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = distances[edge_index_u, edge_index_v]
        heuristic = 1 / sparse_distances
        return heuristic

    def sample(self, invtemp=1.0, inference=False, start_node=None):
        """Sample paths using optimized sparse matrices and nearest neighbor selection."""
        if inference:
            # Use sparse matrices for optimized inference
            effective_beta = self.beta * invtemp
            
            # Compute sparse probability matrix (n x k_nearest)
            probmat_sparse_numpy = (self.pheromone_sparse_numpy ** self.alpha) * (self.heuristic_sparse_numpy ** effective_beta)

            # # Convert to numpy for numba optimization
            # probmat_sparse_numpy = probmat_sparse.detach().cpu().numpy().astype(np.float32)
            
            # Convert shortest_path to numpy for FACO
            local_source_route_numpy = self.shortest_path.detach().cpu().numpy().astype(np.uint16)
            
            # Use optimized sampling with sparse matrices
            paths = numba_sample_sparse_optimized_faco(
                probmat_sparse_numpy,
                self.nn_list,
                self.backup_nn_list,
                self.distances_numpy,
                local_source_route_numpy,
                count=self.n_ants,
                start_node=start_node,
            )
            paths = torch.from_numpy(paths.T.astype(np.int64)).to(self.device)
            # paths = self.gen_path(require_prob=False, start_node=start_node)
            log_probs = None
        else:
            # Fallback to full matrix for training (backward compatibility)
            paths, log_probs = self.gen_path(invtemp=invtemp, require_prob=True, start_node=start_node)
        
        costs = self.gen_path_costs(paths)
        return costs, log_probs, paths

    def local_search(self, paths, inference=False):
        if self.local_search_type == "2opt":
            paths = self.two_opt(paths, inference)
        elif self.local_search_type == "nls":
            paths = self.nls(paths, inference)
        return paths

    @torch.no_grad()
    def run(self, n_iterations, start_node=None):
        assert n_iterations > 0

        start_time = time.time()
        for iteration in range(n_iterations):
            # Use enhanced sampling with optimized algorithm
            # print("Iteration:", iteration + 1)
            costs, _, paths = self.sample(inference=True, start_node=start_node)
            _paths = paths.clone()

            # Update ants' routes and costs efficiently
            if KEEP_BETTER_ANT_SOL:
                # Vectorized comparison to find improved solutions
                improved_mask = costs < self.ants_cost
                improved_indices = torch.nonzero(improved_mask, as_tuple=True)[0]
                
                if len(improved_indices) > 0:
                    # Batch update only improved solutions
                    self.ants_route[:, improved_indices] = paths[:, improved_indices]
                    self.ants_cost[improved_indices] = costs[improved_indices]
            else:
                # Update all ants' routes and costs at once
                self.ants_route = paths.clone()
                self.ants_cost = costs.clone()

            iteration_best_idx = self.ants_cost.argmin()
            if costs[iteration_best_idx] < self.best_ant_cost:
                self.best_ant_route = paths[:, iteration_best_idx].clone()
                self.best_ant_cost = costs[iteration_best_idx].item()

            # Probabilistic selection: 1% iteration best, 99% global best
            if torch.rand(1).item() < 0.01:
                # Use iteration best ant (1% probability)
                upd_ant_route = paths[:, iteration_best_idx]
                upd_ant_cost = costs[iteration_best_idx]
            else:
                # Use global best ant (99% probability)
                upd_ant_route = self.best_ant_route
                upd_ant_cost = torch.tensor(self.best_ant_cost, device=self.device)

            if upd_ant_cost < self.lowest_cost:
                self.shortest_path = upd_ant_route.clone()
                self.lowest_cost = upd_ant_cost.item() if isinstance(upd_ant_cost, torch.Tensor) else upd_ant_cost
                elapsed_time = time.time() - start_time
                # print(f"  [Runtime] {elapsed_time:.2f}s - New best cost: {self.lowest_cost:.2f}")
            
            # Update pheromones using probabilistic ant selection
            self.update_pheromone(upd_ant_route.unsqueeze(1), upd_ant_cost.unsqueeze(0))
                
        end_time = time.time()

        # Pairwise Jaccard similarity between paths
        # edge_sets = []
        # _paths = _paths.T.cpu().numpy()  # type: ignore
        # for _p in _paths:
        #     edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))

        # # Diversity
        # jaccard_sum = 0
        # for i, j in combinations(range(len(edge_sets)), 2):
        #     jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        # diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        diversity = 0

        # Ensure lowest_cost is a Python float for compatibility with torch tensors
        return float(self.lowest_cost), diversity, end_time - start_time

    @torch.no_grad()
    def update_pheromone(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        deltas = 1.0 / costs

        # Apply decay to sparse pheromone matrix
        self.pheromone_sparse = self.pheromone_sparse * self.decay

        # Elitist
        best_delta, best_idx = deltas.max(dim=0)
        best_tour = paths[:, best_idx]
        self._update_sparse_pheromone_path(best_tour, best_delta)
       
        # trail limit factor
        _max = 1 / ((1 - self.decay) * self.lowest_cost)
        avg = self.k_nearest if hasattr(self, 'k_nearest') else self.problem_size
        p = self.p_best ** (1. / avg)
        _min = min(_max, _max * (1 - p) / ((avg - 1) * p))
        self.pheromone_sparse = torch.clamp(self.pheromone_sparse, min=_min, max=_max)

        # Invalidate cache only when needed
        if hasattr(self, '_pheromone_sparse_numpy'):
            delattr(self, '_pheromone_sparse_numpy')


    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        # assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_path_costs_init(self, paths):
        '''
        Args:
            paths: torch tensor with shape (n_ants, problem_size)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape[1] == self.problem_size
        u = torch.from_numpy(paths.astype(np.int64)).to(self.device)  # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        # assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_numpy_path_costs(self, paths):
        '''
        Args:
            paths: numpy array with shape (n_ants, problem_size)
        Returns:
                Lengths of paths: numpy array with shape (n_ants,)
        '''
        assert paths.shape[1] == self.problem_size
        u = paths  # shape: (n_ants, problem_size)
        v = np.roll(u, shift=1, axis=1)  # shape: (n_ants, problem_size)
        # assert (self.distances[u, v] > 0).all()
        return np.sum(self.distances_numpy[u, v], axis=1)

    def gen_path(self, invtemp=1.0, require_prob=False, paths=None, start_node=None):
        '''
        Tour contruction for all ants using sparse matrices
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        # Use sparse probability matrix computation
        prob_mat_sparse = (self.pheromone_sparse ** self.alpha) * (self.heuristic_sparse ** self.beta)
        
        if paths is None:
            if start_node is None:
                start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
            else:
                start = torch.ones((self.n_ants,), dtype = torch.long, device=self.device) * start_node
        else:
            start = paths[0]
        paths_list = [start] # paths_list[i] is the ith move (tensor) for all ants

        initial_prob = torch.ones(self.n_ants, device=self.device) * (1 / self.problem_size)
        log_probs_list = [initial_prob.log()]  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions

        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        index = torch.arange(self.n_ants, device=self.device)
        mask[index, start] = 0

        prev = start
        for i in range(self.problem_size - 1):
            # Use sparse matrix calculation for better performance
            dist = self._get_sparse_probabilities(prev, mask, invtemp, prob_mat_sparse)
            dist = Categorical(probs=dist)
            actions = paths[i + 1] if paths is not None else dist.sample() # shape: (n_ants,)
            paths_list.append(actions)
            if require_prob:
                log_probs = dist.log_prob(actions) # shape: (n_ants,)
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[index, actions] = 0
        
        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)

    @cached_property
    def distances_numpy(self):
        return self.distances.detach().cpu().numpy().astype(np.float32)

    @cached_property
    def heuristic_sparse_numpy(self):
        return self.heuristic_sparse.detach().cpu().numpy().astype(np.float32)

    @cached_property
    def pheromone_sparse_numpy(self):
        return self.pheromone_sparse.detach().cpu().numpy().astype(np.float32)

    @cached_property
    def heuristic_numpy(self):
        return self.heuristic.detach().cpu().numpy().astype(np.float32)  # type: ignore

    @cached_property
    def heuristic_dist(self):
        return 1 / (self.heuristic_numpy / self.heuristic_numpy.max(-1, keepdims=True) + 1e-5)
    
    def _expand_sparse_to_full_for_2opt_torch(self, sparse_matrix):
        """Convert sparse n×k matrix to full n×n format for 2-opt operations only."""
        full_matrix = np.full((self.problem_size, self.problem_size), 1e-10, dtype=np.float32)
        
        for i in range(self.problem_size):
            for j in range(self.k_nearest):
                neighbor = self.nn_list[i, j]
                if neighbor >= 0:  # Valid neighbor
                    full_matrix[i, neighbor] = sparse_matrix[i, j]
                    full_matrix[neighbor, i] = sparse_matrix[i, j]  # Symmetric
        
        return full_matrix

    def two_opt(self, paths, inference = False):
        maxt = 100 if inference else self.problem_size // 4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)
        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)

        return best_paths

    def nls(self, paths, inference=False, T_nls=5, T_p=20):
        maxt = 100 if inference else self.problem_size // 4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)

        best_costs = self.gen_numpy_path_costs(best_paths)
        new_paths = best_paths

        for _ in range(T_nls):
            perturbed_paths = batched_two_opt_python(self.heuristic_dist, new_paths, max_iterations=T_p)
            new_paths = batched_two_opt_python(self.distances_numpy, perturbed_paths, max_iterations=maxt)
            new_costs = self.gen_numpy_path_costs(new_paths)

            improved_indices = new_costs < best_costs
            best_paths[improved_indices] = new_paths[improved_indices]
            best_costs[improved_indices] = new_costs[improved_indices]

        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)

        return best_paths
    
    def nls_init(self, paths, inference=False, T_nls=5, T_p=20):
        """
        Perform NLS initialization for paths.
        
        Args:
            paths: Initial paths to optimize
            inference: Whether in inference mode
            T_nls: Number of NLS iterations
            T_p: Number of perturbation iterations
            
        Returns:
            Optimized paths after NLS
        """
        maxt = 100 if inference else self.problem_size // 4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.cpu().numpy(), max_iterations=maxt)

        best_costs = self.gen_path_costs_init(best_paths)
        new_paths = best_paths

        
        for _ in range(T_nls):
            perturbed_paths = batched_two_opt_python(self.heuristic_dist, new_paths, max_iterations=T_p)
            new_paths = batched_two_opt_python(self.distances_numpy, perturbed_paths, max_iterations=maxt)
            new_costs = self.gen_path_costs_init(new_paths)
            
            improved_indices = new_costs < best_costs
            best_paths[improved_indices] = new_paths[improved_indices]
            best_costs[improved_indices] = new_costs[improved_indices]

        best_paths = torch.from_numpy(best_paths.astype(np.int64)).to(self.device)

        return best_paths
    
    def _build_neighbor_structures(self):
        """Build nearest neighbor and backup neighbor lists."""
        distances_np = self.distances.detach().cpu().numpy().astype(np.float32)
        self.nn_list, self.backup_nn_list = build_nearest_neighbor_lists(distances_np, self.k_nearest)
        
        # Build neighbor index array for fast lookup during pheromone updates
        # nn_index_map[i, j] = k means neighbor j is at position k in node i's neighbor list
        # Use -1 to indicate that neighbor j is not in node i's neighbor list
        self.nn_index_map = np.full((self.problem_size, self.problem_size), -1, dtype=np.int32)
        for i in range(self.problem_size):
            for j in range(self.k_nearest):
                neighbor = self.nn_list[i, j]
                if neighbor >= 0:  # Valid neighbor
                    self.nn_index_map[i, neighbor] = j
    
    def _get_sparse_probabilities(self, current_nodes, mask, invtemp, prob_mat_sparse):
        """
        Calculate probabilities using sparse matrix representation.
        
        Args:
            current_nodes: Current positions of all ants
            mask: Mask for visited nodes
            invtemp: Inverse temperature
            prob_mat_sparse: Sparse probability matrix
            
        Returns:
            Probability distribution for next node selection
        """
        # Convert sparse probabilities to full format for current nodes
        batch_size = current_nodes.shape[0]
        full_probs = torch.zeros((batch_size, self.problem_size), device=self.device)
        
        # Get nearest neighbors for current nodes
        nn_indices = torch.from_numpy(self.nn_list).to(self.device)
        
        for i in range(batch_size):
            current_node = current_nodes[i]
            # Get probabilities for nearest neighbors
            for j in range(self.k_nearest):
                neighbor = nn_indices[current_node, j]
                if neighbor >= 0:  # Valid neighbor
                    full_probs[i, neighbor] = prob_mat_sparse[current_node, j]
         
        # Apply mask and temperature
        dist = (full_probs ** invtemp) * mask
        # Add small epsilon to avoid numerical issues
        dist = dist + 1e-10
        dist = dist / dist.sum(dim=1, keepdim=True)
        
        return dist
    
    def _check_sparse_convergence(self, threshold):
        """
        Check convergence using sparse matrix representation.
        
        Args:
            threshold: Convergence threshold
            
        Returns:
            True if converged, False otherwise
        """
        # Check if all edges in the shortest path have pheromone >= threshold
        nn_indices = torch.from_numpy(self.nn_list).to(self.device)
        
        for i in range(self.problem_size):
            current_node = self.shortest_path[i]
            next_node = self.shortest_path[(i + 1) % self.problem_size]
            
            # Find next_node in current_node's nearest neighbors
            found = False
            for j in range(self.k_nearest):
                neighbor = nn_indices[current_node, j]
                if neighbor == next_node:
                    if self.pheromone_sparse[current_node, j] < threshold:
                        return False
                    found = True
                    break
            
            # If next_node is not in nearest neighbors, it's not converged
            if not found:
                return False
        
        return True

    def _build_sparse_heuristic_matrix(self):
        """Build sparse heuristic matrix (n x k_nearest) using nearest neighbors."""
        heuristic_sparse = torch.zeros((self.problem_size, self.k_nearest), device=self.device)
        nn_indices = torch.from_numpy(self.nn_list).to(self.device)
        
        for i in range(self.problem_size):
            for j in range(self.k_nearest):
                neighbor = nn_indices[i, j]
                if neighbor >= 0:  # Valid neighbor
                    heuristic_sparse[i, j] = 1.0 / self.distances[i, neighbor]
                else:
                    heuristic_sparse[i, j] = 1e-10  # Small value for invalid neighbors
        
        return heuristic_sparse
    
    def _compress_full_to_sparse(self, full_matrix):
        """Convert full n×n matrix to sparse n×k format using nearest neighbors."""
        sparse_matrix = torch.zeros((self.problem_size, self.k_nearest), device=self.device)
        nn_indices = torch.from_numpy(self.nn_list).to(self.device)
        for i in range(self.problem_size):
            for j in range(self.k_nearest):
                neighbor = nn_indices[i, j]
                if neighbor >= 0:  # Valid neighbor
                    sparse_matrix[i, j] = full_matrix[i, neighbor]
                else:
                    sparse_matrix[i, j] = 1e-10
        
        return sparse_matrix
    
    def _update_sparse_pheromone_path(self, path, delta):
        """Update pheromone for a single path using sparse matrix representation."""
        path_cpu = path.cpu().numpy()
        
        for i in range(len(path_cpu)):
            from_node = path_cpu[i]
            to_node = path_cpu[(i + 1) % len(path_cpu)]  # Wrap around for TSP
            
            # Update forward direction using direct array lookup
            j = self.nn_index_map[from_node, to_node]
            if j >= 0:  # Valid neighbor
                self.pheromone_sparse[from_node, j] += delta
            

class FACO_NP(): 
    """
    ACO class for numpy implementation
    """
    def __init__(
        self, 
        distances: np.ndarray,
        n_ants=20,
        heuristic: np.ndarray | None = None,
        k_sparse=None,
        k_nearest=20,  # Added for enhanced optimization
        pheromone: np.ndarray | None = None,
        ants_route: np.ndarray | None = None,
        ants_cost: np.ndarray | None = None,
        best_ant_route: np.ndarray | None = None,
        best_ant_cost: float | None = None,
        decay=0.9,
        alpha=1,
        beta=1,
        p_best=0.1,  # Added for maxmin variant
        # AS variants
        elitist=False,
        maxmin=False,
        rank_based=False,
        n_elites=None,
        smoothing=False,
        smoothing_thres=5,
        smoothing_delta=0.5,
        shift_cost=True,
        local_search_type: str | None = None,
        trail_limit_factor=5.0,
    ):
        self.problem_size = len(distances)
        self.distances = distances
        self.n_ants = n_ants
        self.k_nearest = min(k_nearest, self.problem_size - 1)  # Added for optimization
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.p_best = p_best  # Added parameter for maxmin variant
        self.elitist = elitist or maxmin  # maxmin uses elitist
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites or n_ants // 10  # only for rank-based
        self.smoothing = smoothing
        self.smoothing_cnt = 0
        self.smoothing_thres = smoothing_thres
        self.smoothing_delta = smoothing_delta
        self.shift_cost = shift_cost
        assert local_search_type in [None, "2opt", "nls"]
        self.local_search_type = local_search_type
        self.trail_limit_factor = trail_limit_factor

        if k_sparse is None:
            self.k_sparse = self.k_nearest
        else:
            self.k_nearest = k_sparse
            self.k_sparse = k_sparse

        # Build nearest neighbor structures first for sparse matrices
        self._build_neighbor_structures()

        if pheromone is None:
            # Initialize sparse pheromone matrix (n x k_nearest)
            self.pheromone_sparse = np.ones((self.problem_size, self.k_nearest), dtype=np.float32)
        else:
            if pheromone.shape == (self.problem_size, self.k_nearest):
                self.pheromone_sparse = pheromone.astype(np.float32)
            else:
                # Convert full matrix to sparse if needed
                self.pheromone_sparse = self._compress_full_to_sparse(pheromone.astype(np.float32))

        if heuristic is None:
            assert self.k_sparse is not None
            # Use legacy sparse heuristic for backward compatibility
            full_heuristic = self.simple_heuristic(distances, self.k_sparse)
            self.heuristic = full_heuristic.astype(np.float32)
            self.heuristic_sparse = self._compress_full_to_sparse(full_heuristic)
        else:
            self.heuristic = heuristic.astype(np.float32)
            if heuristic.shape == (self.problem_size, self.k_nearest):
                self.heuristic_sparse = heuristic.astype(np.float32)
            else:
                self.heuristic_sparse = self._compress_full_to_sparse(heuristic.astype(np.float32))

        initial_routes = build_multiple_nn_tours_parallel(
            os.cpu_count() or 4, self.nn_list, self.distances, self.problem_size
        )
        initial_routes = self.nls(initial_routes, inference=True)
        initial_costs = self.gen_path_costs(initial_routes)
        # Find the best initial route
        best_initial_idx = np.argmin(initial_costs)
        best_initial_route = initial_routes[best_initial_idx]
        # Set the best initial route as shortest path
        self.shortest_path = best_initial_route.copy()
        self.lowest_cost = initial_costs[best_initial_idx]
        # print(f"Initial cost: {self.lowest_cost:.2f}")

        if ants_route is not None and ants_cost is not None:
            assert ants_route.shape == (self.problem_size, n_ants)
            assert ants_cost.shape == (n_ants,)
            self.ants_route = ants_route.astype(np.int64)
            self.ants_cost = ants_cost.astype(np.float32)
        else:
            # Initialize ants' routes and costs
            self.ants_route = np.zeros((self.problem_size, n_ants), dtype=np.int64)
            self.ants_cost = np.zeros(n_ants, dtype=np.float32)

            # Fill ants' routes with initial low-cost random paths
            for i in range(n_ants):
                self.ants_route[:, i] = self.shortest_path.copy()
                self.ants_cost[i] = self.lowest_cost

        if best_ant_route is not None and best_ant_cost is not None:
            assert best_ant_route.shape == (self.problem_size,)
            self.best_ant_route = best_ant_route.astype(np.int64)
            self.best_ant_cost = best_ant_cost.astype(np.float32)
        else:
            # Initialize best ant with the shortest path
            self.best_ant_route = self.shortest_path.copy()
            self.best_ant_cost = self.lowest_cost

    def simple_heuristic(self, distances, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        assert k_sparse is not None

        topk_indices = np.argpartition(distances, k_sparse, axis=1)[:, :k_sparse]
        edge_index_u = np.repeat(np.arange(len(distances)), k_sparse)
        edge_index_v = np.ravel(topk_indices)
        sparse_distances = np.ones_like(distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = distances[edge_index_u, edge_index_v]
        heuristic = 1 / sparse_distances
        return heuristic

    def sample(self, invtemp=1.0, inference=False, start_node=None):
        """Sample paths using optimized sparse matrices and nearest neighbor selection."""
        if inference:
            # Use sparse matrices for optimized inference
            effective_beta = self.beta * invtemp
            
            # Compute sparse probability matrix (n x k_nearest)
            probmat_sparse = (self.pheromone_sparse ** self.alpha) * (self.heuristic_sparse ** effective_beta)
            
            # Convert shortest_path to numpy for FACO
            local_source_route_numpy = self.shortest_path.astype(np.uint16)
            
            # Use optimized sampling with sparse matrices
            paths = numba_sample_sparse_optimized_faco(
                probmat_sparse,
                self.nn_list,
                self.backup_nn_list,
                self.distances,
                local_source_route_numpy,
                count=self.n_ants,
                start_node=start_node,
            )
            paths = paths.T.astype(np.int64)  # Convert to (problem_size, n_ants)
            log_probs = None
        else:
            # Fallback to full matrix for training (backward compatibility)
            paths, log_probs = self.gen_path(invtemp=invtemp, require_prob=True, start_node=start_node)
        
        costs = self.gen_path_costs(paths.T)  # gen_path_costs expects (n_ants, problem_size)
        return costs, log_probs, paths

    def local_search(self, paths, inference=False):
        # paths: (problem_size, n_ants) - convert to (n_ants, problem_size) for local search
        paths_transposed = paths.T
        if self.local_search_type == "2opt":
            paths_transposed = self.two_opt(paths_transposed, inference)
        elif self.local_search_type == "nls":
            paths_transposed = self.nls(paths_transposed, inference)
        # Convert back to (problem_size, n_ants)
        return paths_transposed.T

    def run(self, n_iterations, start_node=None):
        assert n_iterations > 0

        start_time = time.time()
        for iteration in range(n_iterations):
            # print("Iteration:", iteration + 1)
            # Use enhanced sampling with optimized algorithm
            costs, _, paths = self.sample(inference=True, start_node=start_node)
            # paths is (problem_size, n_ants) from sample method
            _paths = paths.copy()

            # Update ants' routes and costs efficiently
            if KEEP_BETTER_ANT_SOL:
                # Vectorized comparison to find improved solutions
                improved_mask = costs < self.ants_cost
                improved_indices = np.where(improved_mask)[0]
                
                if len(improved_indices) > 0:
                    # Batch update only improved solutions
                    self.ants_route[:, improved_indices] = paths[:, improved_indices]
                    self.ants_cost[improved_indices] = costs[improved_indices]
            else:
                # Update all ants' routes and costs at once
                self.ants_route = paths.copy()
                self.ants_cost = costs.copy()

            iteration_best_idx = self.ants_cost.argmin()
            if costs[iteration_best_idx] < self.best_ant_cost:
                self.best_ant_route = paths[:, iteration_best_idx].copy()
                self.best_ant_cost = costs[iteration_best_idx]

            # Probabilistic selection: 1% iteration best, 99% global best
            if np.random.random() < 0.01:
                # Use iteration best ant (1% probability)
                upd_ant_route = paths[:, iteration_best_idx]
                upd_ant_cost = costs[iteration_best_idx]
            else:
                # Use global best ant (99% probability)
                upd_ant_route = self.best_ant_route
                upd_ant_cost = self.best_ant_cost

            if upd_ant_cost < self.lowest_cost:
                self.shortest_path = upd_ant_route.copy()
                self.lowest_cost = upd_ant_cost
                elapsed_time = time.time() - start_time
                # print(f"Iteration {iteration + 1}  [Runtime] {elapsed_time:.2f}s - New best cost: {self.lowest_cost:.2f}")

            # Update pheromones using probabilistic ant selection
            self.update_pheromone(upd_ant_route.reshape(-1, 1), np.array([upd_ant_cost]))
            
        end_time = time.time()

        # Pairwise Jaccard similarity between paths
        edge_sets = []
        _paths = _paths.T  # Convert to (n_ants, problem_size)
        for _p in _paths:
            edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))

        # Diversity
        jaccard_sum = 0
        for i, j in combinations(range(len(edge_sets)), 2):
            jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        return self.lowest_cost, diversity, end_time - start_time

    def update_pheromone(self, paths: np.ndarray, costs: np.ndarray):
        '''
        Args:
            paths: np.ndarray with shape (problem_size, n_ants)
            costs: np.ndarray with shape (n_ants,)
        '''
        deltas = 1.0 / costs

        # Apply decay to sparse pheromone matrix
        self.pheromone_sparse = self.pheromone_sparse * self.decay

        # Elitist
        best_idx = deltas.argmax()
        best_delta = deltas[best_idx]
        best_tour = paths[:, best_idx]
        self._update_sparse_pheromone_path(best_tour, best_delta)
       
        # trail limit factor
        _max = 1 / ((1 - self.decay) * self.lowest_cost)
        avg = self.k_nearest if hasattr(self, 'k_nearest') else self.problem_size
        p = self.p_best ** (1. / avg)
        _min = min(_max, _max * (1 - p) / ((avg - 1) * p))
        self.pheromone_sparse = np.clip(self.pheromone_sparse, a_min=_min, a_max=_max)

    def gen_path_costs(self, paths):
        '''
        Args:
            paths: numpy ndarray with shape (n_ants, problem_size)
        Returns:
            Lengths of paths: numpy ndarray with shape (n_ants,)
        '''
        u = paths
        v = np.roll(u, shift=1, axis=1)
        # assert (self.distances[u, v] > 0).all()
        return np.sum(self.distances[u, v], axis=1)

    def gen_path(self, invtemp=1.0, require_prob=False, paths=None, start_node=None, epsilon=None):
        '''
        Tour contruction for all ants using sparse matrices
        Returns:
            paths: np.ndarray with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: np.ndarray with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        # Use sparse probability matrix computation
        prob_mat_sparse = (self.pheromone_sparse ** self.alpha) * (self.heuristic_sparse ** self.beta)
        
        if paths is None:
            if start_node is None:
                start = np.random.randint(0, self.problem_size, size=self.n_ants)
            else:
                start = np.ones(self.n_ants, dtype=np.int64) * start_node
        else:
            start = paths[0]
        paths_list = [start] # paths_list[i] is the ith move (tensor) for all ants

        initial_prob = np.ones(self.n_ants) * (1 / self.problem_size)
        log_probs_list = [np.log(initial_prob)]  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions

        mask = np.ones((self.n_ants, self.problem_size), dtype=np.uint8)
        index = np.arange(self.n_ants)
        mask[index, start] = 0

        prev = start
        for i in range(self.problem_size - 1):
            # Use sparse matrix calculation for better performance
            dist = self._get_sparse_probabilities_np(prev, mask, invtemp, prob_mat_sparse)
            
            # Numpy-based sampling instead of torch Categorical
            if paths is not None:
                actions = paths[i + 1]
            else:
                actions = np.zeros(self.n_ants, dtype=np.int64)
                for ant in range(self.n_ants):
                    # Sample from the distribution for each ant
                    probs = dist[ant]
                    actions[ant] = np.random.choice(len(probs), p=probs)
            
            paths_list.append(actions)
            if require_prob:
                log_probs = np.zeros(self.n_ants)
                for ant in range(self.n_ants):
                    log_probs[ant] = np.log(dist[ant, actions[ant]] + 1e-10)  # Add small epsilon for numerical stability
                log_probs_list.append(log_probs)
                mask = mask.copy()
            prev = actions
            mask[index, actions] = 0

        if require_prob:
            return np.stack(paths_list), np.stack(log_probs_list)
        else:
            return np.stack(paths_list)

    @cached_property
    def heuristic_dist(self):
        return 1 / (self.heuristic / self.heuristic.max(-1, keepdims=True) + 1e-5)
    
    def _expand_sparse_to_full_for_2opt(self, sparse_matrix):
        """Convert sparse n×k matrix to full n×n format for 2-opt operations only."""
        full_matrix = np.full((self.problem_size, self.problem_size), 1e-10, dtype=np.float32)
        
        for i in range(self.problem_size):
            for j in range(self.k_nearest):
                neighbor = self.nn_list[i, j]
                if neighbor >= 0:  # Valid neighbor
                    full_matrix[i, neighbor] = sparse_matrix[i, j]
                    full_matrix[neighbor, i] = sparse_matrix[i, j]  # Symmetric
        
        return full_matrix

    def two_opt(self, paths: np.ndarray, inference=False):
        maxt = 10000 if inference else self.problem_size // 4
        best_paths = batched_two_opt_python(self.distances, paths, max_iterations=maxt)
        best_paths = best_paths.astype(np.int64)
        return best_paths

    def nls(self, paths: np.ndarray, inference=False, T_nls=5, T_p=20):
        maxt = 100 if inference else self.problem_size // 4
        best_paths = batched_two_opt_python(self.distances, paths, max_iterations=maxt)
        best_costs = self.gen_path_costs(best_paths)
        new_paths = best_paths

        for _ in range(T_nls):
            perturbed_paths = batched_two_opt_python(self.heuristic_dist, new_paths, max_iterations=T_p)
            new_paths = batched_two_opt_python(self.distances, perturbed_paths, max_iterations=maxt)
            new_costs = self.gen_path_costs(new_paths)

            improved_indices = new_costs < best_costs
            best_paths[improved_indices] = new_paths[improved_indices]
            best_costs[improved_indices] = new_costs[improved_indices]

        best_paths = best_paths.astype(np.int64)
        return best_paths

    def _build_neighbor_structures(self):
        """Build nearest neighbor and backup neighbor lists."""
        self.nn_list, self.backup_nn_list = build_nearest_neighbor_lists(self.distances, self.k_nearest)
        
        # Build neighbor index array for fast lookup during pheromone updates
        # nn_index_map[i, j] = k means neighbor j is at position k in node i's neighbor list
        # Use -1 to indicate that neighbor j is not in node i's neighbor list
        self.nn_index_map = np.full((self.problem_size, self.problem_size), -1, dtype=np.int32)
        for i in range(self.problem_size):
            for j in range(self.k_nearest):
                neighbor = self.nn_list[i, j]
                if neighbor >= 0:  # Valid neighbor
                    self.nn_index_map[i, neighbor] = j
    
    def _get_sparse_probabilities_np(self, current_nodes, mask, invtemp, prob_mat_sparse):
        """
        Calculate probabilities using sparse matrix representation for numpy.
        
        Args:
            current_nodes: Current positions of all ants
            mask: Mask for visited nodes
            invtemp: Inverse temperature
            prob_mat_sparse: Sparse probability matrix
            
        Returns:
            Probability distribution for next node selection
        """
        # Convert sparse probabilities to full format for current nodes
        batch_size = current_nodes.shape[0]
        full_probs = np.zeros((batch_size, self.problem_size), dtype=np.float32)
        
        for i in range(batch_size):
            current_node = current_nodes[i]
            # Get probabilities for nearest neighbors
            for j in range(self.k_nearest):
                neighbor = self.nn_list[current_node, j]
                if neighbor >= 0:  # Valid neighbor
                    full_probs[i, neighbor] = prob_mat_sparse[current_node, j]
        
        # Apply mask and temperature
        dist = (full_probs ** invtemp) * mask
        # Add small epsilon to avoid numerical issues
        dist = dist + 1e-10
        dist = dist / dist.sum(axis=1, keepdims=True)
        
        return dist
    
    def _check_sparse_convergence_np(self, threshold):
        """
        Check convergence using sparse matrix representation for numpy.
        
        Args:
            threshold: Convergence threshold
            
        Returns:
            True if converged, False otherwise
        """
        # Check if all edges in the shortest path have pheromone >= threshold
        for i in range(self.problem_size):
            current_node = self.shortest_path[i]
            next_node = self.shortest_path[(i + 1) % self.problem_size]
            
            # Find next_node in current_node's nearest neighbors
            found = False
            for j in range(self.k_nearest):
                neighbor = self.nn_list[current_node, j]
                if neighbor == next_node:
                    if self.pheromone_sparse[current_node, j] < threshold:
                        return False
                    found = True
                    break
            
            # If next_node is not in nearest neighbors, it's not converged
            if not found:
                return False
        
        return True

    def _build_sparse_heuristic_matrix(self):
        """Build sparse heuristic matrix (n x k_nearest) using nearest neighbors."""
        heuristic_sparse = np.zeros((self.problem_size, self.k_nearest), dtype=np.float32)
        
        for i in range(self.problem_size):
            for j in range(self.k_nearest):
                neighbor = self.nn_list[i, j]
                if neighbor >= 0:  # Valid neighbor
                    heuristic_sparse[i, j] = 1.0 / self.distances[i, neighbor]
                else:
                    heuristic_sparse[i, j] = 1e-10  # Small value for invalid neighbors
        
        return heuristic_sparse
    
    def _compress_full_to_sparse(self, full_matrix):
        """Convert full n×n matrix to sparse n×k format using nearest neighbors."""
        sparse_matrix = np.zeros((self.problem_size, self.k_nearest), dtype=np.float32)
        
        for i in range(self.problem_size):
            for j in range(self.k_nearest):
                neighbor = self.nn_list[i, j]
                if neighbor >= 0:  # Valid neighbor
                    sparse_matrix[i, j] = full_matrix[i, neighbor]
                else:
                    sparse_matrix[i, j] = 1e-10
        
        return sparse_matrix

    def _update_sparse_pheromone_path(self, path, delta):
        """Update pheromone for a single path using sparse matrix representation."""
        
        for i in range(len(path)):
            from_node = path[i]
            to_node = path[(i + 1) % len(path)]  # Wrap around for TSP
            
            # Update forward direction using direct array lookup
            j = self.nn_index_map[from_node, to_node]
            if j >= 0:  # Valid neighbor
                self.pheromone_sparse[from_node, j] += delta

# Enhanced ACO with optimized node selection based on nearest neighbors

@nb.jit(nb.types.UniTuple(nb.int64[:, :], 2)(nb.float32[:, :], nb.int64), nopython=True, nogil=True)
def build_nearest_neighbor_lists(distances: np.ndarray, k_nearest: int):
    """
    Build nearest neighbor lists for each node.
    
    Args:
        distances: Distance matrix (n x n)
        k_nearest: Number of nearest neighbors to keep
    
    Returns:
        Tuple of (nn_list, backup_nn_list) where:
        - nn_list: k nearest neighbors for each node
        - backup_nn_list: remaining nodes sorted by distance
    """
    n = distances.shape[0]
    nn_list = np.zeros((n, k_nearest), dtype=nb.int64)
    backup_nn_list = np.zeros((n, 64), dtype=nb.int64)  # Fixed backup size of 64
    
    for node in range(n):
        # Get all other nodes and their distances
        other_nodes = np.zeros(n - 1, dtype=nb.int64)
        node_distances = np.zeros(n - 1, dtype=nb.float32)
        
        idx = 0
        for other in range(n):
            if other != node:
                other_nodes[idx] = other
                node_distances[idx] = distances[node, other]
                idx += 1
        
        # Sort by distance
        sorted_indices = np.argsort(node_distances)
        
        # Fill nearest neighbors
        for i in range(min(k_nearest, len(sorted_indices))):
            nn_list[node, i] = other_nodes[sorted_indices[i]]
        
        # Fill backup list with remaining nodes (up to 64)
        backup_idx = 0
        for i in range(k_nearest, len(sorted_indices)):
            if backup_idx < 64:  # Limit to 64 backup neighbors
                backup_nn_list[node, backup_idx] = other_nodes[sorted_indices[i]]
                backup_idx += 1
    
    return nn_list, backup_nn_list



@nb.jit(nb.int64(nb.int64, nb.uint8[:], nb.float32[:,:], nb.int64[:,:], nb.int64[:,:], nb.int64, nb.int64, nb.float32[:,:]), 
        nopython=True, nogil=True)
def select_next_node(current_node: int,
                    visited_mask: np.ndarray,
                    probmat_sparse: np.ndarray,
                    nn_list: np.ndarray,
                    backup_nn_list: np.ndarray,
                    k_nearest: int,
                    n: int,
                    distances: np.ndarray) -> int:
    """
    Select the next node to visit using sparse probability matrix and nearest neighbor optimization.
    
    Args:
        current_node: Current position of the ant
        visited_mask: Boolean mask of visited nodes (1 = visited, 0 = unvisited)
        probmat_sparse: Sparse probability matrix (n x k_nearest)
        nn_list: Nearest neighbor lists (n x k)
        backup_nn_list: Backup neighbor lists (n x remaining)
        k_nearest: Number of nearest neighbors
        n: Total number of nodes
        distances: Distance matrix (n x n)
    
    Returns:
        Selected next node index
    """
    # Candidate list from sparse matrix
    cl = np.zeros(k_nearest, dtype=nb.int64)
    cl_products = np.zeros(k_nearest, dtype=nb.float64)
    cl_size = 0
    cl_products_sum = 0.0
    
    # Track best node found so far
    max_product = 0.0
    max_node = current_node
    
    # Build candidate list from nearest neighbors using sparse matrix
    for j in range(k_nearest):
        neighbor = nn_list[current_node, j]
        if neighbor >= 0 and visited_mask[neighbor] == 0:  # Valid unvisited neighbor
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
        # Roulette wheel selection from candidate list
        r = np.random.random() * cl_products_sum
        cumsum = 0.0
        for i in range(cl_size):
            cumsum += cl_products[i]
            if r <= cumsum:
                chosen_node = cl[i]
                break
    elif cl_size == 0:
        # No unvisited nearest neighbors, use backup list
        chosen_node = current_node  # Default fallback
        
        for i in range(len(backup_nn_list[current_node])):
            neighbor = backup_nn_list[current_node, i]
            if neighbor >= 0 and visited_mask[neighbor] == 0:
                chosen_node = neighbor
                break
        
        # If still no valid node found, find closest unvisited node by distance
        if chosen_node == current_node:
            min_distance = np.inf
            for node in range(n):
                if visited_mask[node] == 0:
                    distance = distances[current_node, node]
                    if distance < min_distance:
                        min_distance = distance
                        chosen_node = node

    return chosen_node

@nb.jit(nb.uint16[:](nb.int64, nb.int64[:,:], nb.float32[:,:], nb.int64), nopython=True, nogil=True)
def build_nn_tour(start_node: int, nn_list: np.ndarray, distances: np.ndarray, dimension: int):
    """
    Build a tour using nearest neighbors with fallback to closest unvisited node.
    Translates the C++ bitmask-based nearest neighbor tour construction.
    
    Args:
        start_node: Starting node for the tour
        nn_list: Nearest neighbor lists (n x k)
        distances: Distance matrix (n x n) 
        dimension: Number of nodes in the problem
    
    Returns:
        Complete tour as array of node indices
    """
    # Initialize tour and visited bitmask
    tour = np.zeros(dimension, dtype=np.uint16)
    visited = np.zeros(dimension, dtype=np.uint8)  # Using uint8 array as bitmask
    
    # Start the tour - equivalent to visited.set_bit(start_node) and tour.push_back(start_node)
    visited[start_node] = 1
    tour[0] = start_node
    
    # Build the rest of the tour - equivalent to for (uint32_t i = 1; i < dimension_; ++i)
    for i in range(1, dimension):
        prev = tour[i - 1]  # Equivalent to auto prev = tour.back()
        next_node = prev  # Equivalent to auto next = prev
        
        # Try to find next node from nearest neighbors
        # Equivalent to for (auto node : get_nearest_neighbors(prev, total_nn_per_node_))
        for j in range(nn_list.shape[1]):
            neighbor = nn_list[prev, j]
            if neighbor >= 0 and neighbor < dimension and visited[neighbor] == 0:
                next_node = neighbor
                break  # Equivalent to break in C++
        
        # If no unvisited nearest neighbor found, find closest unvisited node
        # Equivalent to if (next == prev) block in C++
        if next_node == prev:
            min_cost = np.inf  # Equivalent to std::numeric_limits<double>::max()
            for node in range(dimension):  # Equivalent to for (uint32_t node = 0; node < dimension_; ++node)
                if visited[node] == 0 and distances[prev, node] < min_cost:
                    min_cost = distances[prev, node]
                    next_node = node
        
        # Assert equivalent - ensure we found a valid next node
        # In production code, this should never happen if logic is correct
        if next_node == prev:
            # Emergency fallback: find any unvisited node
            for node in range(dimension):
                if visited[node] == 0:
                    next_node = node
                    break
        
        # Add node to tour and mark as visited
        # Equivalent to visited.set_bit(next) and tour.push_back(next)
        visited[next_node] = 1
        tour[i] = next_node
    
    return tour

@nb.jit(nb.uint16[:,:](nb.int64, nb.int64[:,:], nb.float32[:,:], nb.int64), nopython=True, nogil=True)
def build_multiple_nn_tours(count: int, nn_list: np.ndarray, distances: np.ndarray, dimension: int):
    """
    Build multiple tours using nearest neighbors construction with random starting points.
    
    Args:
        count: Number of tours to build
        nn_list: Nearest neighbor lists (n x k)
        distances: Distance matrix (n x n)
        dimension: Number of nodes in the problem
    
    Returns:
        Array of tours (count x dimension)
    """
    tours = np.zeros((count, dimension), dtype=np.uint16)
    
    for i in range(count):
        # Generate random starting node for each tour
        start_node = np.random.randint(0, dimension)
        tours[i] = build_nn_tour(start_node, nn_list, distances, dimension)
    
    return tours

@nb.jit(nb.uint16[:,:](nb.int64, nb.int64[:,:], nb.float32[:,:], nb.int64), nopython=True, nogil=True, parallel=True)
def build_multiple_nn_tours_parallel(count: int, nn_list: np.ndarray, distances: np.ndarray, dimension: int):
    """
    Build multiple tours using nearest neighbors construction with random starting points in parallel.
    Uses specified number of cores for maximum performance.
    
    Args:
        count: Number of tours to build (should match number of CPU cores)
        nn_list: Nearest neighbor lists (n x k)
        distances: Distance matrix (n x n)
        dimension: Number of nodes in the problem
    
    Returns:
        Array of tours (count x dimension)
    """
    tours = np.zeros((count, dimension), dtype=np.uint16)

    # Parallel loop using nb.prange for multi-core execution
    for i in nb.prange(count):
        # Deterministic starting node distribution across cores
        # This ensures good load balancing while maintaining reproducibility
        start_node = np.random.randint(0, dimension)
        tours[i] = build_nn_tour(start_node, nn_list, distances, dimension)
    
    return tours



@nb.jit(nb.int64(nb.float32[:]), nopython=True, nogil=True)
def select_next_node_simple(prob_array: np.ndarray) -> int:
    """
    Simple roulette wheel selection for backward compatibility.
    
    Args:
        prob_array: Probability array for available nodes
    
    Returns:
        Selected node index
    """
    total_prob = prob_array.sum()
    if total_prob <= 1e-10:
        # Fallback: find first non-zero probability
        for i in range(len(prob_array)):
            if prob_array[i] > 1e-10:
                return i
        return 0
    
    rand = np.random.random() * total_prob
    cumsum = 0.0
    
    for k in range(len(prob_array)):
        cumsum += prob_array[k]
        if rand <= cumsum:
            return k
    
    # Fallback to last index
    return len(prob_array) - 1

# For implentation of FACO relocate node
@nb.jit(nb.int64(nb.int64, nb.uint16[:], nb.int64[:]), nopython=True, nogil=True)
def get_succ(node, route_, positions_):
    """
    Get the successor of a given node in the route.
    
    Args:
        node: The node to find the successor for
        route_: Array containing the route sequence
        positions_: Array mapping node to its position in route_
    
    Returns:
        The successor node of the given node
    """
    pos = positions_[node]
    
    # If at the last position, successor is the first node (wrap around)
    if pos + 1 == len(route_):
        return route_[0]
    else:
        return route_[pos + 1]

@nb.jit(nb.int64(nb.int64, nb.uint16[:], nb.int64[:]), nopython=True, nogil=True)
def get_pred(node, route_, positions_):
    """
    Get the predecessor of a given node in the route.
    
    Args:
        node: The node to find the predecessor for
        route_: Array containing the route sequence
        positions_: Array mapping node to its position in route_
    
    Returns:
        The predecessor node of the given node
    """
    pos = positions_[node]
    
    # If at the first position, predecessor is the last node (wrap around)
    if pos == 0:
        return route_[len(route_) - 1]  # route_.back() equivalent
    else:
        return route_[pos - 1]

@nb.jit(nb.boolean(nb.int64, nb.int64, nb.uint16[:], nb.int64[:]), nopython=True, nogil=True)
def contains_edge(a, b, route_, positions_):
    """
    Check if there is an edge (undirected) between nodes a and b in the route.
    
    Args:
        a: First node
        b: Second node
        route_: Array containing the route sequence
        positions_: Array mapping node to its position in route_
    
    Returns:
        True if there is an edge between a and b (in either direction)
    """
    return b == get_succ(a, route_, positions_) or b == get_pred(a, route_, positions_)

@nb.jit(nb.boolean(nb.int64, nb.int64, nb.uint16[:], nb.int64[:]), nopython=True, nogil=True)
def contains_directed_edge(a, b, route_, positions_):
    """
    Check if there is a directed edge from node a to node b in the route.
    
    Args:
        a: Source node
        b: Destination node
        route_: Array containing the route sequence
        positions_: Array mapping node to its position in route_
    
    Returns:
        True if there is a directed edge from a to b
    """
    return b == get_succ(a, route_, positions_)

@nb.jit(nb.float64(nb.int64, nb.int64, nb.uint16[:], nb.int64[:], nb.float64, nb.float32[:, :]), nopython=True, nogil=True)
def relocate_node(target, node, route_, positions_, current_cost, distance_matrix):
    """
    Relocate a node to be the successor of the target node in the route.
    
    Args:
        target: The target node after which to place the relocated node
        node: The node to relocate
        route_: Array containing the route sequence (will be modified in-place)
        positions_: Array mapping node to its position in route_ (will be modified in-place)
        current_cost: Current cost of the route
        distance_matrix: Distance matrix for calculating costs
    
    Returns:
        New cost of the route after relocation
    """
    # Assertions equivalent (but numba doesn't support assert)
    if node == target or get_succ(target, route_, positions_) == node:
        return current_cost
    
    node_pos = positions_[node]
    target_pos = positions_[target]
    
    node_pred = get_pred(node, route_, positions_)
    node_succ = get_succ(node, route_, positions_)
    target_succ = get_succ(target, route_, positions_)
    
    # Calculate cost delta first (before modifying the route)
    cost_delta = (- distance_matrix[node_pred, node]
                  - distance_matrix[node, node_succ]
                  - distance_matrix[target, target_succ]
                  + distance_matrix[node_pred, node_succ]
                  + distance_matrix[target, node]
                  + distance_matrix[node, target_succ])
    
    if target_pos < node_pos:  # Case 1: target is before node
        # 1 2 3 t 5 6 n 7 8 => 1 2 3 t n 5 6 7 8
        # Rotate elements from target+1 to node (inclusive)
        # Save the node value
        node_value = route_[node_pos]
        
        # Shift elements to the right
        for i in range(node_pos, target_pos + 1, -1):
            route_[i] = route_[i - 1]
        
        # Place node after target
        route_[target_pos + 1] = node_value
        
        # Update positions for affected nodes
        for i in range(target_pos + 1, node_pos + 1):
            positions_[route_[i]] = i
            
    else:  # Case 2: target is after node
        # 1 2 3 n 5 6 t 7 8 => 1 2 3 5 6 t n 7 8
        # Save the node value
        node_value = route_[node_pos]
        
        # Shift elements to the left
        for i in range(node_pos, target_pos):
            route_[i] = route_[i + 1]
        
        # Place node after target
        route_[target_pos] = node_value
        
        # Update positions for affected nodes
        for i in range(node_pos, target_pos + 1):
            positions_[route_[i]] = i
    
    return current_cost + cost_delta

@nb.jit(nb.int32(nb.int32, nb.int32, nb.uint16[:], nb.int64[:]), nopython=True, nogil=True)
def flip_route_section(start_node: int, end_node: int, route_: np.ndarray, positions_: np.ndarray) -> int:
    """
    Flip a section of the route between start_node and end_node.
    
    Args:
        start_node: Starting node of the section to flip
        end_node: Ending node of the section to flip
        route_: Array containing the route sequence (will be modified in-place)
        positions_: Array mapping node to its position in route_ (will be modified in-place)
    
    Returns:
        Starting position of the flipped section (or 0 if alternative flip was used)
    """
    first = positions_[start_node]
    last = positions_[end_node]
    
    if first > last:
        # Swap if first > last
        temp = first
        first = last
        last = temp
    
    length = len(route_)
    segment_length = last - first
    remaining_length = length - segment_length
    
    if segment_length <= remaining_length:
        # Reverse the specified segment
        # Manually reverse the segment since numba doesn't support slice assignment with step
        left = first
        right = last - 1
        while left < right:
            # Swap elements
            temp = route_[left]
            route_[left] = route_[right]
            route_[right] = temp
            left += 1
            right -= 1
        
        # Update positions for the flipped segment
        for k in range(first, last):
            positions_[route_[k]] = k
        
        return first
    else:
        # Reverse the rest of the route, leave the segment intact
        first_adj = (first - 1) if first > 0 else length - 1
        last_adj = last % length
        
        # Swap first_adj and last_adj
        temp = first_adj
        first_adj = last_adj
        last_adj = temp
        
        l = first_adj
        r = last_adj
        i = 0
        j = length - first_adj + last_adj + 1
        
        while i < j:
            # Swap route_[l] and route_[r]
            temp = route_[l]
            route_[l] = route_[r]
            route_[r] = temp
            
            # Update positions
            positions_[route_[l]] = l
            positions_[route_[r]] = r
            
            # Update indices
            l = (l + 1) % length
            r = (r - 1) if r > 0 else length - 1
            i += 1
            j -= 1
        
        return 0

@nb.jit(nb.float64(nb.uint16[:], nb.int64[:], nb.float32[:, :], nb.int64[:, :], nb.int64[:], nb.uint32, nb.uint32), nopython=True, nogil=True)
def two_opt_nn(route_: np.ndarray, positions_: np.ndarray, distances: np.ndarray, nn_list: np.ndarray, checklist: np.ndarray, nn_list_size: int, max_changes: int) -> float:
    """
    2-opt local search using nearest neighbors optimization.
    
    Args:
        route_: Array containing the route sequence (will be modified in-place)
        positions_: Array mapping node to its position in route_ (will be modified in-place)
        distances: Distance matrix (n x n)
        nn_list: Nearest neighbor lists (n x k)
        checklist: Array of nodes to check for improvements (will be modified in-place)
        nn_list_size: Number of nearest neighbors to consider
        max_changes: Maximum number of route changes allowed
    
    Returns:
        Total cost improvement achieved
    """
    n = len(route_)
    changes_count = 0
    cost_change = 0.0
    
    # Use provided checklist
    checklist_pos = 0
    
    while checklist_pos < len(checklist) and changes_count < max_changes:
        a = checklist[checklist_pos]
        checklist_pos += 1
        
        if a >= n:
            continue
            
        a_next = get_succ(a, route_, positions_)
        a_prev = get_pred(a, route_, positions_)
        
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
                b_next = get_succ(b, route_, positions_)
                
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
                b_prev = get_pred(b, route_, positions_)
                
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
            # Perform the flip
            flip_route_section(best_move[0], best_move[1], route_, positions_)
            
            # Add affected nodes to checklist if not already present
            # Check if we need to expand the checklist
            nodes_to_add = 0
            for i in range(4):
                node = best_move[i]
                if node >= 0:
                    # Check if node is already in remaining checklist
                    found = False
                    for j in range(checklist_pos, len(checklist)):
                        if checklist[j] == node:
                            found = True
                            break
                    if not found:
                        nodes_to_add += 1
            
            # If we need to add nodes, create new checklist with additional space
            if nodes_to_add > 0:
                new_checklist = np.zeros(len(checklist) + nodes_to_add, dtype=np.int64)
                # Copy existing checklist
                for i in range(len(checklist)):
                    new_checklist[i] = checklist[i]
                
                # Add new nodes to the end
                add_pos = len(checklist)
                for i in range(4):
                    node = best_move[i]
                    if node >= 0:
                        # Check if node is already in remaining checklist
                        found = False
                        for j in range(checklist_pos, len(checklist)):
                            if checklist[j] == node:
                                found = True
                                break
                        if not found:
                            new_checklist[add_pos] = node
                            add_pos += 1
                
                checklist = new_checklist
            
            changes_count += 1
            cost_change -= max_diff

    return cost_change

# Numba JIT functions for checklist operations with array parameters
@nb.jit(nb.boolean(nb.int64[:], nb.int64), nopython=True, nogil=True)
def contains(checklist: np.ndarray, node: int):
    """
    Check if a node exists in the checklist.
    
    Args:
        checklist: Checklist array to search (checklist[0] stores current length)
        node: Node to search for
    
    Returns:
        True if node exists in checklist, False otherwise
    """
    current_length = checklist[0]
    if current_length == 0:
        return False
    for i in range(1, current_length + 1):  # Start from index 1, skip length at index 0
        if checklist[i] == node:
            return True
    return False

@nb.jit(nb.void(nb.int64[:], nb.int64), nopython=True, nogil=True)
def push_back(checklist: np.ndarray, node: int):
    """
    Add a node to the end of the checklist if there's space.
    
    Args:
        checklist: Checklist array (modified in-place, checklist[0] stores current length)
        node: Node to add
    """
    current_length = checklist[0]
    assert current_length + 1 < len(checklist)  # +1 because index 0 is reserved for length
    checklist[current_length + 1] = node  # Add at position current_length + 1
    checklist[0] = current_length + 1     # Update length stored at index 0

@nb.jit(nb.void(nb.int64[:]), nopython=True, nogil=True)
def clear(checklist: np.ndarray):
    """
    Clear the checklist by setting used elements to -1 and resetting length.
    
    Args:
        checklist: Checklist array to clear (modified in-place, checklist[0] stores current length)
    """
    current_length = checklist[0]
    for i in range(1, current_length + 1):  # Clear from index 1 to current_length
        checklist[i] = -1
    checklist[0] = 0  # Reset length to 0


@nb.jit(nb.float64(nb.uint16[:], nb.float32[:, :]), nopython=True, nogil=True)
def get_route_cost(route: np.ndarray, distances: np.ndarray) -> float:
    """
    Calculate the total cost of a route based on the distance matrix.
    
    Args:
        route: Array containing the route sequence
        distances: Distance matrix (n x n)
    
    Returns:
        Total cost of the route
    """
    n = len(route)
    cost = 0.0
    for i in range(n - 1):
        cost += distances[route[i], route[i + 1]]
    cost += distances[route[n - 1], route[0]]  # Complete the cycle
    return cost



@nb.jit(nb.uint16[:](nb.float32[:,:], nb.int64[:,:], nb.int64[:,:], nb.int64, nb.float32[:,:], nb.uint16[:]), 
        nopython=True, nogil=True)
def _numba_sample_sparse_optimized_faco(probmat_sparse: np.ndarray,
                                 nn_list: np.ndarray,
                                 backup_nn_list: np.ndarray,
                                 start_node: int,
                                 distances: np.ndarray,
                                 local_source_route: np.ndarray):
    """
    Optimized sampling using sparse probability matrix (n x k) with FACO local search.
    
    Args:
        probmat_sparse: Sparse probability matrix (n x k_nearest)
        nn_list: Nearest neighbor lists (n x k)
        backup_nn_list: Backup neighbor lists (n x remaining) 
        start_node: Starting node for the route
        distances: Distance matrix (n x n)
        local_source_route: Initial route to use as starting point
    
    Returns:
        Optimized route array
    """
    n = probmat_sparse.shape[0]
    k_nearest = probmat_sparse.shape[1]
    
    # Initialize position mapping for local source route
    local_source_positions = np.zeros(n, dtype=np.int64)
    for i, node in enumerate(local_source_route):
        local_source_positions[node] = i    
            
    # Start with a copy of the local source route
    route = local_source_route.copy()
    positions = np.zeros(n, dtype=np.int64)
    for i, node in enumerate(route):
        positions[node] = i
    
    # Calculate initial cost
    cost = get_route_cost(route, distances)
    
    # Track visited nodes for route construction
    visited_mask = np.zeros(n, dtype=np.uint8)
    
    # Initialize with start node
    visited_mask[start_node] = 1
    current_node = start_node
    
    new_edges = 0
    target_new_edges = MIN_NEW_EDGES

    ls_checklist = np.full(n + 1, -1, dtype=np.int64)
    ls_checklist[0] = 0  # Store current length at index 0

    # Modify route step by step using FACO approach
    while new_edges < target_new_edges and np.sum(visited_mask) < n:
        curr = current_node
        chosen_node = select_next_node(
            current_node, visited_mask, probmat_sparse, 
            nn_list, backup_nn_list, k_nearest, n, distances
        )
        
        # Mark node as visited
        visited_mask[chosen_node] = 1
        
        # Apply relocation operation
        cost = relocate_node(curr, chosen_node, route, positions, cost, distances)

        chos_pred = get_pred(chosen_node, route, positions)
        # Check if this creates a new edge (not in original local source)
        if not contains_edge(curr, chosen_node, local_source_route, local_source_positions):
            if not contains(ls_checklist, curr):
                push_back(ls_checklist, curr)
            if not contains(ls_checklist, chosen_node):
                push_back(ls_checklist, chosen_node)
            if not contains(ls_checklist, chos_pred):
                push_back(ls_checklist, chos_pred)

            new_edges += 1

        current_node = chosen_node
    
    if SAMPLE_TWO_OPT:
        two_opt_nn(route, positions, distances, nn_list, ls_checklist, k_nearest, n)
        
    # if SOURCE_SOL_LOCAL_UPDATE and cost < get_route_cost(local_source_route, distances):
    #     # If the new route is better than the local source, update it
    #     local_source_route[:] = route[:]

    return route

@nb.jit(nb.void(nb.float32[:,:], nb.int64[:,:], nb.int64[:,:], nb.int64[:], nb.float32[:,:], nb.uint16[:], nb.uint16[:,:]), 
        nopython=True, nogil=True, parallel=True)
def numba_parallel_sample_sparse_optimized_faco(probmat_sparse: np.ndarray,
                                                 nn_list: np.ndarray,
                                                 backup_nn_list: np.ndarray,
                                                 start_nodes: np.ndarray,
                                                 distances: np.ndarray,
                                                 local_source_route: np.ndarray,
                                                 routes: np.ndarray):
    """
    Parallel version using pure Numba parallelism for better performance.
    
    Args:
        probmat_sparse: Sparse probability matrix (n x k_nearest)
        nn_list: Nearest neighbor lists
        backup_nn_list: Backup neighbor lists  
        start_nodes: Array of starting nodes for each route
        distances: Distance matrix (n x n)
        local_source_route: Initial route to use as starting point
        routes: Output array to store generated routes (modified in-place)
    """
    count = len(start_nodes)
    
    # Parallel loop using nb.prange for multi-core execution
    for i in nb.prange(count):
        routes[i] = _numba_sample_sparse_optimized_faco(
            probmat_sparse, nn_list, backup_nn_list, start_nodes[i], 
            distances, local_source_route
        )

def numba_sample_sparse_optimized_faco(probmat_sparse: np.ndarray,
                                nn_list: np.ndarray,
                                backup_nn_list: np.ndarray,
                                distances: np.ndarray,
                                local_source_route: np.ndarray,
                                count: int = 1,
                                start_node=None,):
    """
    Generate multiple routes using sparse probability matrix optimization with FACO.
    
    Args:
        probmat_sparse: Sparse probability matrix (n x k_nearest)
        nn_list: Nearest neighbor lists
        backup_nn_list: Backup neighbor lists
        distances: Distance matrix (n x n)
        local_source_route: Array of initial routes
        count: Number of routes to generate
        start_node: Starting node (None for random)
    
    Returns:
        Array of routes (count x n)
    """
    n = probmat_sparse.shape[0]
    routes = np.zeros((count, n), dtype=np.uint16)
    probmat_sparse = probmat_sparse.astype(np.float32)
    
    if start_node is None:
        start_nodes = np.random.randint(0, n, size=count, dtype=np.int64)
    else:
        start_nodes = np.ones(count, dtype=np.int64) * start_node

    # if count <= 4 and n < 500:
    # Sequential execution for small problems
    # for i in range(count):
    #     routes[i] = _numba_sample_sparse_optimized_faco(
    #         probmat_sparse, nn_list, backup_nn_list, start_nodes[i], 
    #         distances, local_source_route
    #     )
    # else:
    # Use pure Numba parallelism for better performance
    numba_parallel_sample_sparse_optimized_faco(
        probmat_sparse, nn_list, backup_nn_list, start_nodes, 
        distances, local_source_route, routes
    )
    return routes
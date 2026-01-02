/**
 * MFACO Training Module - C++ implementation
 */

#include "mfaco_train.h"
#include <omp.h>
#include <cstring>
#include <stdexcept>

namespace mfaco {

// ============================================================================
// Constructor
// ============================================================================

MFACO_TSP::MFACO_TSP(
    const float* dist_ptr,
    int32_t n_,
    int32_t n_ants_,
    int32_t cand_list_size,
    int32_t backup_list_size,
    int32_t min_new_edges_,
    float decay,
    float alpha_,
    float p_best_,
    bool use_local_search_,
    bool random_mode_,
    bool disable_heuristic_
) : n(n_),
    n_ants(n_ants_),
    k(std::min(cand_list_size, n_ - 1)),
    bl(std::min(backup_list_size, std::max(0, n_ - 1 - k))),
    min_new_edges(min_new_edges_),
    rho(decay),
    alpha(alpha_),
    p_best(p_best_),
    use_local_search(use_local_search_),
    random_mode(random_mode_),
    disable_heuristic(disable_heuristic_)
{
    // Copy distances
    distances.assign(dist_ptr, dist_ptr + n * n);

    // Build nearest neighbor lists
    build_nn_lists();
    build_nn_pos();
    build_heuristic();

    // Initialize source/best solutions
    source_route.resize(n);
    best_route.resize(n);
    source_positions.resize(n);
    build_initial_tour();

    // Initialize pheromone
    auto [tmin, tmax] = calc_trail_limits_cl(source_cost);
    tau_min = tmin;
    tau_max = tmax;
    pheromone_sparse.assign(n * k, tau_max);

    // Initialize RNG
    rng_.seed(42);
}


// ============================================================================
// Public API
// ============================================================================

void MFACO_TSP::seed_rng(uint64_t seed) {
    rng_.seed(seed);
}


void MFACO_TSP::sample(bool require_prob, const float* residual_logits, SampleResult& result, bool parallel_traced) {
    result.clear();
    result.costs.resize(n_ants);
    result.routes.resize(n_ants);

    // Compute probability matrix: tau^alpha * eta * exp(residual)
    std::vector<float> probmat(n * k);
    compute_probmat(residual_logits, probmat);

    // Generate random start nodes
    std::vector<int32_t> start_nodes(n_ants);
    for (int32_t a = 0; a < n_ants; ++a) {
        start_nodes[a] = rng_.next_uint(n);
    }

    // Per-ant RNG seeds are only needed for OpenMP paths.
    // IMPORTANT: avoid consuming extra RNG in the default traced single-thread path.
    std::vector<uint64_t> ant_seeds;
    auto ensure_ant_seeds = [&]() {
        if (!ant_seeds.empty()) return;
        ant_seeds.resize(static_cast<size_t>(n_ants));
        for (int32_t a = 0; a < n_ants; ++a) {
            uint64_t hi = static_cast<uint64_t>(rng_.next_u32());
            uint64_t lo = static_cast<uint64_t>(rng_.next_u32());
            ant_seeds[static_cast<size_t>(a)] = (hi << 32) ^ lo ^ (0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(a));
        }
    };

    if (require_prob) {
        if (!parallel_traced) {
            // Traced mode: single-threaded (original behavior)
            result.traces.reserve(n_ants, n_ants * min_new_edges * 2);
            result.traces.starts.push_back(0);

            std::vector<int32_t> checklist;
            checklist.reserve(n);

            for (int32_t a = 0; a < n_ants; ++a) {
                result.routes[a].resize(n);
                MFACOTrace trace;
                trace.reserve(min_new_edges * 2);

                float cost = sample_ant_traced(
                    probmat.data(),
                    start_nodes[a],
                    result.routes[a],
                    checklist,
                    trace,
                    rng_
                );

                result.costs[a] = cost;

                // Append trace to batch
                result.traces.start_nodes.push_back(trace.start_node);
                for (size_t i = 0; i < trace.curr_nodes.size(); ++i) {
                    result.traces.curr_nodes.push_back(trace.curr_nodes[i]);
                    result.traces.chosen_nodes.push_back(trace.chosen_nodes[i]);
                    result.traces.is_stochastic.push_back(trace.is_stochastic[i]);
                    result.traces.used_uniform.push_back(trace.used_uniform[i]);
                    result.traces.pick_j.push_back(trace.pick_j[i]);
                    result.traces.valid_mask.push_back(trace.valid_mask[i]);
                }
                result.traces.starts.push_back(static_cast<int32_t>(result.traces.curr_nodes.size()));
            }
        } else {
            // Traced mode: parallelized with per-ant RNG and per-ant traces
            ensure_ant_seeds();
            std::vector<MFACOTrace> traces_per_ant(static_cast<size_t>(n_ants));

            #pragma omp parallel
            {
                std::vector<int32_t> checklist;
                checklist.reserve(n);

                #pragma omp for schedule(static, 1)
                for (int32_t a = 0; a < n_ants; ++a) {
                    result.routes[a].resize(n);
                    MFACOTrace& trace = traces_per_ant[static_cast<size_t>(a)];
                    trace.reserve(min_new_edges * 2);
                    Xoshiro128Plus rng_local;
                    rng_local.seed(ant_seeds[static_cast<size_t>(a)]);

                    result.costs[a] = sample_ant_traced(
                        probmat.data(),
                        start_nodes[a],
                        result.routes[a],
                        checklist,
                        trace,
                        rng_local
                    );
                }
            }

            // Merge traces in ant index order (deterministic)
            result.traces.clear();
            result.traces.starts.resize(static_cast<size_t>(n_ants) + 1);
            result.traces.start_nodes.resize(static_cast<size_t>(n_ants));
            result.traces.starts[0] = 0;
            for (int32_t a = 0; a < n_ants; ++a) {
                const MFACOTrace& t = traces_per_ant[static_cast<size_t>(a)];
                result.traces.start_nodes[static_cast<size_t>(a)] = t.start_node;
                result.traces.starts[static_cast<size_t>(a) + 1] =
                    result.traces.starts[static_cast<size_t>(a)] + static_cast<int32_t>(t.curr_nodes.size());
            }
            int32_t total = result.traces.starts[static_cast<size_t>(n_ants)];
            result.traces.curr_nodes.resize(static_cast<size_t>(total));
            result.traces.chosen_nodes.resize(static_cast<size_t>(total));
            result.traces.is_stochastic.resize(static_cast<size_t>(total));
            result.traces.used_uniform.resize(static_cast<size_t>(total));
            result.traces.pick_j.resize(static_cast<size_t>(total));
            result.traces.valid_mask.resize(static_cast<size_t>(total));

            for (int32_t a = 0; a < n_ants; ++a) {
                const MFACOTrace& t = traces_per_ant[static_cast<size_t>(a)];
                int32_t off = result.traces.starts[static_cast<size_t>(a)];
                for (size_t i = 0; i < t.curr_nodes.size(); ++i) {
                    result.traces.curr_nodes[static_cast<size_t>(off) + i] = t.curr_nodes[i];
                    result.traces.chosen_nodes[static_cast<size_t>(off) + i] = t.chosen_nodes[i];
                    result.traces.is_stochastic[static_cast<size_t>(off) + i] = t.is_stochastic[i];
                    result.traces.used_uniform[static_cast<size_t>(off) + i] = t.used_uniform[i];
                    result.traces.pick_j[static_cast<size_t>(off) + i] = t.pick_j[i];
                    result.traces.valid_mask[static_cast<size_t>(off) + i] = t.valid_mask[i];
                }
            }
        }
    } else {
        // Fast mode: parallel
        ensure_ant_seeds();
        #pragma omp parallel
        {
            std::vector<int32_t> checklist;
            checklist.reserve(n);

            #pragma omp for schedule(static, 1)
            for (int32_t a = 0; a < n_ants; ++a) {
                result.routes[a].resize(n);
                Xoshiro128Plus rng_local;
                rng_local.seed(ant_seeds[static_cast<size_t>(a)]);
                result.costs[a] = sample_ant_fast(
                    probmat.data(),
                    start_nodes[a],
                    result.routes[a],
                    checklist,
                    rng_local
                );
            }
        }
    }
}


void MFACO_TSP::update_pheromone(const int32_t* best_flat, float new_best_cost) {
    // Update global best
    if (new_best_cost < best_cost) {
        best_cost = new_best_cost;
        std::copy(best_flat, best_flat + n, best_route.begin());
    }

    // Update trail limits based on global best (MMAS-style)
    auto [tmin, tmax] = calc_trail_limits_cl(best_cost);
    tau_min = tmin;
    tau_max = tmax;

    // Evaporate
    float decay_factor = 1.0f - rho;
    for (int32_t i = 0; i < n * k; ++i) {
        pheromone_sparse[i] *= decay_factor;
        pheromone_sparse[i] = std::max(tau_min, std::min(tau_max, pheromone_sparse[i]));
    }

    // Deposit on best route
    float deposit = 1.0f / (new_best_cost + EPS);
    int32_t prev = best_flat[n - 1];  // last node before wrap
    for (int32_t i = 0; i < n; ++i) {
        int32_t cur = best_flat[i];
        
        // Forward edge (prev -> cur)
        int32_t j = nn_pos[prev * n + cur];
        if (j >= 0) {
            pheromone_sparse[prev * k + j] = std::min(pheromone_sparse[prev * k + j] + deposit, tau_max);
        }
        
        // Symmetric: reverse edge (cur -> prev)
        int32_t jr = nn_pos[cur * n + prev];
        if (jr >= 0) {
            pheromone_sparse[cur * k + jr] = std::min(pheromone_sparse[cur * k + jr] + deposit, tau_max);
        }
        
        prev = cur;
    }

    // Update source solution
    std::copy(best_flat, best_flat + n, source_route.begin());
    source_cost = new_best_cost;
    for (int32_t i = 0; i < n; ++i) {
        source_positions[source_route[i]] = i;
    }
}


void MFACO_TSP::load_snapshot(
    const float* pheromone_ptr,
    const int32_t* source_route_ptr,
    float source_cost_,
    const int32_t* best_route_ptr,
    float best_cost_,
    float tau_min_,
    float tau_max_,
    const int32_t* nn_list_ptr,
    const int32_t* backup_list_ptr
) {
    // Copy pheromone
    std::copy(pheromone_ptr, pheromone_ptr + n * k, pheromone_sparse.begin());

    // Copy source route and cost
    std::copy(source_route_ptr, source_route_ptr + n, source_route.begin());
    source_cost = source_cost_;

    // Copy best route and cost
    std::copy(best_route_ptr, best_route_ptr + n, best_route.begin());
    best_cost = best_cost_;

    // Copy tau limits
    tau_min = tau_min_;
    tau_max = tau_max_;

    // Copy nn_list and backup_list
    std::copy(nn_list_ptr, nn_list_ptr + n * k, nn_list.begin());
    if (bl > 0) {
        std::copy(backup_list_ptr, backup_list_ptr + n * bl, backup_list.begin());
    }

    // Rebuild nn_pos from nn_list
    build_nn_pos();

    // Rebuild heuristic (in case nn_list changed)
    build_heuristic();

    // Rebuild source_positions
    for (int32_t i = 0; i < n; ++i) {
        source_positions[source_route[i]] = i;
    }
}


void MFACO_TSP::set_pheromone(const float* pheromone_ptr) {
    std::copy(pheromone_ptr, pheromone_ptr + n * k, pheromone_sparse.begin());
}


// ============================================================================
// Private methods
// ============================================================================

void MFACO_TSP::build_nn_lists() {
    nn_list.resize(n * k);
    backup_list.resize(n * bl);

    // For each node, sort others by distance and pick nearest
    std::vector<std::pair<float, int32_t>> dist_idx(n - 1);

    for (int32_t u = 0; u < n; ++u) {
        // Build (distance, index) pairs excluding self
        int32_t idx = 0;
        for (int32_t v = 0; v < n; ++v) {
            if (v != u) {
                dist_idx[idx++] = {distances[u * n + v], v};
            }
        }

        // Sort by distance (stable sort for determinism)
        std::stable_sort(dist_idx.begin(), dist_idx.end());

        // Fill nn_list
        for (int32_t j = 0; j < k; ++j) {
            nn_list[u * k + j] = dist_idx[j].second;
        }

        // Fill backup_list
        for (int32_t j = 0; j < bl; ++j) {
            backup_list[u * bl + j] = dist_idx[k + j].second;
        }
    }
}


void MFACO_TSP::build_nn_pos() {
    nn_pos.assign(n * n, -1);
    for (int32_t u = 0; u < n; ++u) {
        for (int32_t j = 0; j < k; ++j) {
            int32_t v = nn_list[u * k + j];
            nn_pos[u * n + v] = j;
        }
    }
}


void MFACO_TSP::build_heuristic() {
    heuristic_sparse.resize(n * k);
    if (disable_heuristic) {
        std::fill(heuristic_sparse.begin(), heuristic_sparse.end(), 1.0f);
        return;
    }
    for (int32_t u = 0; u < n; ++u) {
        for (int32_t j = 0; j < k; ++j) {
            int32_t v = nn_list[u * k + j];
            float d = distances[u * n + v];
            heuristic_sparse[u * k + j] = (d > 0) ? (1.0f / d) : 1.0f;
        }
    }
}


void MFACO_TSP::build_initial_tour() {
    // Build greedy NN tours from multiple starts, keep best
    float best = std::numeric_limits<float>::max();
    std::vector<int32_t> best_tour(n);

    int32_t num_starts = std::min(8, n);
    std::vector<int32_t> tour(n);
    std::vector<uint8_t> visited(n);

    for (int32_t start = 0; start < num_starts; ++start) {
        std::fill(visited.begin(), visited.end(), 0);
        tour[0] = start;
        visited[start] = 1;

        for (int32_t i = 1; i < n; ++i) {
            int32_t curr = tour[i - 1];
            int32_t next = -1;

            // Try nn_list first
            for (int32_t j = 0; j < k; ++j) {
                int32_t v = nn_list[curr * k + j];
                if (!visited[v]) {
                    next = v;
                    break;
                }
            }

            // Fallback: find closest unvisited
            if (next < 0) {
                float min_dist = std::numeric_limits<float>::max();
                for (int32_t v = 0; v < n; ++v) {
                    if (!visited[v] && distances[curr * n + v] < min_dist) {
                        min_dist = distances[curr * n + v];
                        next = v;
                    }
                }
            }

            tour[i] = next;
            visited[next] = 1;
        }

        float cost = get_route_cost(tour);
        if (cost < best) {
            best = cost;
            best_tour = tour;
        }
    }

    source_route = best_tour;
    source_cost = best;
    best_route = best_tour;
    best_cost = best;

    for (int32_t i = 0; i < n; ++i) {
        source_positions[source_route[i]] = i;
    }
}


std::pair<float, float> MFACO_TSP::calc_trail_limits_cl(float solution_cost) const {
    float tau_max_ = 1.0f / (solution_cost * (1.0f - rho) + EPS);
    float avg = static_cast<float>(std::max(2, k));
    float p = std::pow(p_best, 1.0f / avg);
    float tau_min_ = std::min(tau_max_, tau_max_ * (1.0f - p) / ((avg - 1.0f) * p + EPS));
    return {tau_min_, tau_max_};
}


void MFACO_TSP::compute_probmat(const float* residual_logits, std::vector<float>& probmat) {
    // probmat = tau^alpha * eta * exp(residual)
    // When disable_heuristic=true, eta is pre-set to 1.0 in heuristic_sparse.
    for (int32_t u = 0; u < n; ++u) {
        for (int32_t j = 0; j < k; ++j) {
            int32_t idx = u * k + j;
            float tau = pheromone_sparse[idx];
            float eta = heuristic_sparse[idx];
            float w = std::pow(tau + EPS, alpha) * eta;

            if (residual_logits != nullptr) {
                // Clamp residual to prevent overflow
                float r = residual_logits[idx];
                r = std::max(-10.0f, std::min(10.0f, r));
                w *= std::exp(r);
            }

            probmat[idx] = std::max(w, EPS);
        }
    }
}


float MFACO_TSP::sample_ant_fast(
    const float* probmat,
    int32_t start_node,
    std::vector<int32_t>& route_out,
    std::vector<int32_t>& checklist,
    Xoshiro128Plus& rng
) {
    // Initialize route as copy of source
    std::vector<int32_t> route = source_route;
    std::vector<int32_t> positions(n);
    for (int32_t i = 0; i < n; ++i) {
        positions[route[i]] = i;
    }

    std::vector<uint8_t> visited(n, 0);
    visited[start_node] = 1;
    int32_t visited_count = 1;

    checklist.clear();
    checklist.push_back(start_node);

    int32_t new_edges = 0;
    int32_t curr = start_node;

    while (new_edges < min_new_edges && visited_count < n) {
        int16_t pick_j = -1;
        uint64_t valid_mask = 0;
        auto [chosen, is_stoch, used_unif] = select_next_node(curr, &probmat[curr * k], visited.data(), rng, pick_j, valid_mask);

        // Check if this creates a new edge
        if (!contains_edge(curr, chosen, positions)) {
            ++new_edges;
            // Add endpoints to checklist
            if (std::find(checklist.begin(), checklist.end(), curr) == checklist.end()) {
                checklist.push_back(curr);
            }
            if (std::find(checklist.begin(), checklist.end(), chosen) == checklist.end()) {
                checklist.push_back(chosen);
            }
            int32_t chosen_pred = get_pred(chosen, route, positions);
            if (std::find(checklist.begin(), checklist.end(), chosen_pred) == checklist.end()) {
                checklist.push_back(chosen_pred);
            }
        }

        // Relocate chosen to be successor of curr
        relocate_node(curr, chosen, route, positions);

        visited[chosen] = 1;
        ++visited_count;
        curr = chosen;
    }

    // Apply local search if enabled
    if (use_local_search && !checklist.empty()) {
        two_opt_nn(route, positions, checklist);
    }

    // Copy result
    route_out = route;
    return get_route_cost(route);
}


float MFACO_TSP::sample_ant_traced(
    const float* probmat,
    int32_t start_node,
    std::vector<int32_t>& route_out,
    std::vector<int32_t>& checklist,
    MFACOTrace& trace,
    Xoshiro128Plus& rng
) {
    trace.clear();
    trace.start_node = start_node;
    trace.reserve(min_new_edges * 2);

    // Initialize route as copy of source
    std::vector<int32_t> route = source_route;
    std::vector<int32_t> positions(n);
    for (int32_t i = 0; i < n; ++i) {
        positions[route[i]] = i;
    }

    std::vector<uint8_t> visited(n, 0);
    visited[start_node] = 1;
    int32_t visited_count = 1;

    checklist.clear();
    checklist.push_back(start_node);

    int32_t new_edges = 0;
    int32_t curr = start_node;

    while (new_edges < min_new_edges && visited_count < n) {
        int16_t pick_j = -1;
        uint64_t valid_mask = 0;
        auto [chosen, is_stoch, used_unif] = select_next_node(curr, &probmat[curr * k], visited.data(), rng, pick_j, valid_mask);

        // Record decision
        trace.curr_nodes.push_back(curr);
        trace.chosen_nodes.push_back(chosen);
        trace.is_stochastic.push_back(is_stoch ? 1 : 0);
        trace.used_uniform.push_back(used_unif ? 1 : 0);
        trace.pick_j.push_back(pick_j);
        trace.valid_mask.push_back(valid_mask);

        // Check if this creates a new edge
        if (!contains_edge(curr, chosen, positions)) {
            ++new_edges;
            if (std::find(checklist.begin(), checklist.end(), curr) == checklist.end()) {
                checklist.push_back(curr);
            }
            if (std::find(checklist.begin(), checklist.end(), chosen) == checklist.end()) {
                checklist.push_back(chosen);
            }
            int32_t chosen_pred = get_pred(chosen, route, positions);
            if (std::find(checklist.begin(), checklist.end(), chosen_pred) == checklist.end()) {
                checklist.push_back(chosen_pred);
            }
        }

        // Relocate chosen to be successor of curr
        relocate_node(curr, chosen, route, positions);

        visited[chosen] = 1;
        ++visited_count;
        curr = chosen;
    }

    // Apply local search if enabled
    if (use_local_search && !checklist.empty()) {
        two_opt_nn(route, positions, checklist);
    }

    // Copy result
    route_out = route;
    return get_route_cost(route);
}


std::tuple<int32_t, bool, bool> MFACO_TSP::select_next_node(
    int32_t curr,
    const float* probmat_row,
    const uint8_t* visited,
    Xoshiro128Plus& rng,
    int16_t& out_pick_j,
    uint64_t& out_valid_mask
) {
    // Build candidate list from nn_list
    int32_t cl[MAX_CAND_LIST_SIZE];
    float cl_prods[MAX_CAND_LIST_SIZE];
    int16_t cl_jidx[MAX_CAND_LIST_SIZE];
    int32_t cl_size = 0;
    float sum = 0.0f;
    float max_prod = 0.0f;
    int32_t max_node = curr;
    int16_t max_j = -1;

    out_valid_mask = 0;
    out_pick_j = -1;

    for (int32_t j = 0; j < k; ++j) {
        int32_t v = nn_list[curr * k + j];
        if (v >= 0 && !visited[v]) {
            if (j < 64) {
                out_valid_mask |= (1ULL << static_cast<uint64_t>(j));
            }
            float prod = probmat_row[j];
            cl[cl_size] = v;
            cl_prods[cl_size] = prod;
            cl_jidx[cl_size] = static_cast<int16_t>(j);
            sum += prod;
            if (prod > max_prod) {
                max_prod = prod;
                max_node = v;
                max_j = static_cast<int16_t>(j);
            }
            ++cl_size;
        }
    }

    bool is_stochastic = false;
    bool used_uniform = false;
    int32_t chosen = max_node;
    out_pick_j = max_j;

    if (cl_size > 1) {
        is_stochastic = true;

        // Check for underflow - if sum is too small, use uniform
        if (sum < EPS) {
            used_uniform = true;
            int32_t idx = rng.next_uint(cl_size);
            chosen = cl[idx];
            out_pick_j = cl_jidx[idx];
        } else if (random_mode) {
            // Uniform random
            used_uniform = true;
            int32_t idx = rng.next_uint(cl_size);
            chosen = cl[idx];
            out_pick_j = cl_jidx[idx];
        } else {
            // Roulette wheel selection
            float r = rng.next_float() * sum;
            float cumsum = 0.0f;
            chosen = cl[cl_size - 1];
            out_pick_j = cl_jidx[cl_size - 1];
            for (int32_t i = 0; i < cl_size; ++i) {
                cumsum += cl_prods[i];
                if (r <= cumsum) {
                    chosen = cl[i];
                    out_pick_j = cl_jidx[i];
                    break;
                }
            }
        }
    } else if (cl_size == 0) {
        // No candidate in nn_list, try backup list
        for (int32_t j = 0; j < bl; ++j) {
            int32_t v = backup_list[curr * bl + j];
            if (v >= 0 && !visited[v]) {
                chosen = v;
                break;
            }
        }

        // Still nothing? Find closest unvisited globally
        if (chosen == curr) {
            float min_dist = std::numeric_limits<float>::max();
            for (int32_t v = 0; v < n; ++v) {
                if (!visited[v] && distances[curr * n + v] < min_dist) {
                    min_dist = distances[curr * n + v];
                    chosen = v;
                }
            }
        }
    }

    return {chosen, is_stochastic, used_uniform};
}


float MFACO_TSP::relocate_node(
    int32_t target,
    int32_t node,
    std::vector<int32_t>& route,
    std::vector<int32_t>& positions
) {
    if (node == target) return 0.0f;
    
    int32_t target_succ = get_succ(target, route, positions);
    if (target_succ == node) return 0.0f;

    int32_t node_pos = positions[node];
    int32_t target_pos = positions[target];

    int32_t node_pred = get_pred(node, route, positions);
    int32_t node_succ = get_succ(node, route, positions);

    // Calculate cost delta
    float cost_delta = (
        - distances[node_pred * n + node]
        - distances[node * n + node_succ]
        - distances[target * n + target_succ]
        + distances[node_pred * n + node_succ]
        + distances[target * n + node]
        + distances[node * n + target_succ]
    );

    // Perform relocation
    if (target_pos < node_pos) {
        // Case 1: target is before node
        int32_t node_value = route[node_pos];
        for (int32_t i = node_pos; i > target_pos + 1; --i) {
            route[i] = route[i - 1];
        }
        route[target_pos + 1] = node_value;
        for (int32_t i = target_pos + 1; i <= node_pos; ++i) {
            positions[route[i]] = i;
        }
    } else {
        // Case 2: target is after node
        int32_t node_value = route[node_pos];
        for (int32_t i = node_pos; i < target_pos; ++i) {
            route[i] = route[i + 1];
        }
        route[target_pos] = node_value;
        for (int32_t i = node_pos; i <= target_pos; ++i) {
            positions[route[i]] = i;
        }
    }

    return cost_delta;
}


float MFACO_TSP::two_opt_nn(
    std::vector<int32_t>& route,
    std::vector<int32_t>& positions,
    std::vector<int32_t>& checklist
) {
    const int32_t max_changes = 1000;
    int32_t changes_count = 0;
    float total_change = 0.0f;
    size_t checklist_pos = 0;

    while (checklist_pos < checklist.size() && changes_count < max_changes) {
        int32_t a = checklist[checklist_pos++];
        if (a < 0 || a >= n) continue;

        int32_t a_next = get_succ(a, route, positions);
        int32_t a_prev = get_pred(a, route, positions);

        float dist_a_to_next = distances[a * n + a_next];
        float dist_a_to_prev = distances[a_prev * n + a];

        float max_diff = 0.0f;
        int32_t best_move[4] = {-1, -1, -1, -1};

        // Check moves with a -> a_next edge
        for (int32_t j = 0; j < k; ++j) {
            int32_t b = nn_list[a * k + j];
            if (b < 0 || b >= n) break;

            float dist_ab = distances[a * n + b];
            if (dist_a_to_next > dist_ab) {
                int32_t b_next = get_succ(b, route, positions);
                float diff = dist_a_to_next + distances[b * n + b_next] - dist_ab - distances[a_next * n + b_next];
                if (diff > max_diff) {
                    best_move[0] = a_next;
                    best_move[1] = b_next;
                    best_move[2] = a;
                    best_move[3] = b;
                    max_diff = diff;
                }
            } else {
                break;
            }
        }

        // Check moves with a_prev -> a edge
        for (int32_t j = 0; j < k; ++j) {
            int32_t b = nn_list[a * k + j];
            if (b < 0 || b >= n) break;

            float dist_ab = distances[a * n + b];
            if (dist_a_to_prev > dist_ab) {
                int32_t b_prev = get_pred(b, route, positions);
                float diff = dist_a_to_prev + distances[b_prev * n + b] - dist_ab - distances[a_prev * n + b_prev];
                if (diff > max_diff) {
                    best_move[0] = a;
                    best_move[1] = b;
                    best_move[2] = a_prev;
                    best_move[3] = b_prev;
                    max_diff = diff;
                }
            } else {
                break;
            }
        }

        if (max_diff > 0) {
            flip_route_section(best_move[0], best_move[1], route, positions);
            ++changes_count;
            total_change -= max_diff;
        }
    }

    return total_change;
}


float MFACO_TSP::get_route_cost(const std::vector<int32_t>& route) const {
    float cost = 0.0f;
    for (int32_t i = 0; i < n - 1; ++i) {
        cost += distances[route[i] * n + route[i + 1]];
    }
    cost += distances[route[n - 1] * n + route[0]];
    return cost;
}


bool MFACO_TSP::contains_edge(int32_t a, int32_t b, const std::vector<int32_t>& positions) const {
    // Need to use source positions for edge checking
    int32_t a_pos = source_positions[a];
    int32_t b_pos = source_positions[b];
    
    // Check if adjacent in source route
    int32_t diff = std::abs(a_pos - b_pos);
    return diff == 1 || diff == n - 1;
}


int32_t MFACO_TSP::get_succ(int32_t node, const std::vector<int32_t>& route, const std::vector<int32_t>& positions) const {
    int32_t pos = positions[node];
    return route[(pos + 1) % n];
}


int32_t MFACO_TSP::get_pred(int32_t node, const std::vector<int32_t>& route, const std::vector<int32_t>& positions) const {
    int32_t pos = positions[node];
    return route[(pos - 1 + n) % n];
}


void MFACO_TSP::flip_route_section(
    int32_t start_node,
    int32_t end_node,
    std::vector<int32_t>& route,
    std::vector<int32_t>& positions
) {
    int32_t first = positions[start_node];
    int32_t last = positions[end_node];

    if (first > last) {
        std::swap(first, last);
    }

    int32_t segment_length = last - first;
    int32_t remaining_length = n - segment_length;

    if (segment_length <= remaining_length) {
        // Flip the segment
        int32_t left = first;
        int32_t right = last - 1;
        while (left < right) {
            std::swap(route[left], route[right]);
            ++left;
            --right;
        }
        for (int32_t i = first; i < last; ++i) {
            positions[route[i]] = i;
        }
    } else {
        // Flip the other segment (wrap around)
        int32_t first_adj = (first > 0) ? first - 1 : n - 1;
        int32_t last_adj = last % n;
        std::swap(first_adj, last_adj);

        int32_t l = first_adj;
        int32_t r = last_adj;
        int32_t i = 0;
        int32_t j = n - first_adj + last_adj + 1;

        while (i < j) {
            std::swap(route[l], route[r]);
            positions[route[l]] = l;
            positions[route[r]] = r;
            l = (l + 1) % n;
            r = (r - 1 + n) % n;
            ++i;
            --j;
        }
    }
}

} // namespace mfaco

/**
 * MFACO Training Module - C++ implementation for fast neural-guided ACO training.
 * 
 * Designed to match the Python MFACO_TSP API in faco_tsp.py for seamless integration
 * with the train.py REINFORCE/A2C training pipeline.
 * 
 * Key features:
 * - Sparse candidate-list pheromone storage (n x k)
 * - Relocate-based focused editing from source solution
 * - Optional neural residual logits for learned biases
 * - Trace recording for log-probability replay
 * - Checklist-guided 2-opt local search
 */

#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include <numeric>

namespace mfaco {

// ============================================================================
// Constants
// ============================================================================

static constexpr uint32_t MAX_CAND_LIST_SIZE = 64;
static constexpr float EPS = 1e-12f;
static constexpr float LOG_EPS = 1e-12f;

// ============================================================================
// Random number generator (xoshiro128+)
// ============================================================================

class Xoshiro128Plus {
public:
    void seed(uint64_t s) {
        // SplitMix64 initialization
        auto splitmix = [](uint64_t& x) {
            x += 0x9e3779b97f4a7c15ULL;
            uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        };
        state_[0] = splitmix(s);
        state_[1] = splitmix(s);
    }

    uint32_t next_u32() {
        const uint64_t s0 = state_[0];
        uint64_t s1 = state_[1];
        const uint64_t result = s0 + s1;
        s1 ^= s0;
        state_[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
        state_[1] = rotl(s1, 36);
        return static_cast<uint32_t>(result >> 32);
    }

    // Uniform [0, max_exclusive)
    uint32_t next_uint(uint32_t max_exclusive) {
        uint32_t x = next_u32();
        uint64_t m = static_cast<uint64_t>(x) * static_cast<uint64_t>(max_exclusive);
        uint32_t l = static_cast<uint32_t>(m);
        if (l < max_exclusive) {
            uint32_t t = -max_exclusive % max_exclusive;
            while (l < t) {
                x = next_u32();
                m = static_cast<uint64_t>(x) * static_cast<uint64_t>(max_exclusive);
                l = static_cast<uint32_t>(m);
            }
        }
        return static_cast<uint32_t>(m >> 32);
    }

    // Uniform [0, 1)
    float next_float() {
        return static_cast<float>(next_u32() >> 8) * (1.0f / (1u << 24));
    }

private:
    static constexpr uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
    uint64_t state_[2];
};


// ============================================================================
// Trace structure for log-prob replay
// ============================================================================

struct MFACOTrace {
    int32_t start_node;
    std::vector<int32_t> curr_nodes;      // current node at each decision
    std::vector<int32_t> chosen_nodes;    // chosen next node at each decision
    std::vector<uint8_t> is_stochastic;   // 1 if decision had multiple valid candidates
    std::vector<uint8_t> used_uniform;    // 1 if underflow caused uniform fallback
    std::vector<int16_t> pick_j;          // index in nn_list row (or -1)
    std::vector<uint64_t> valid_mask;     // bitmask over k candidates at decision time

    void clear() {
        start_node = -1;
        curr_nodes.clear();
        chosen_nodes.clear();
        is_stochastic.clear();
        used_uniform.clear();
        pick_j.clear();
        valid_mask.clear();
    }

    void reserve(size_t n) {
        curr_nodes.reserve(n);
        chosen_nodes.reserve(n);
        is_stochastic.reserve(n);
        used_uniform.reserve(n);
        pick_j.reserve(n);
        valid_mask.reserve(n);
    }
};


// ============================================================================
// Batch trace structure (compact storage for all ants)
// ============================================================================

struct MFACOTraceBatch {
    // Prefix sums: decision range for ant a is [starts[a], starts[a+1])
    std::vector<int32_t> starts;        // length n_ants + 1
    std::vector<int32_t> curr_nodes;    // length D (total decisions)
    std::vector<int32_t> chosen_nodes;  // length D
    std::vector<uint8_t> is_stochastic; // length D
    std::vector<uint8_t> used_uniform;  // length D
    std::vector<int16_t> pick_j;        // length D, index in nn_list row (or -1)
    std::vector<uint64_t> valid_mask;   // length D, bitmask over k candidates at decision time
    std::vector<int32_t> start_nodes;   // length n_ants (start node per ant)

    void clear() {
        starts.clear();
        curr_nodes.clear();
        chosen_nodes.clear();
        is_stochastic.clear();
        used_uniform.clear();
        pick_j.clear();
        valid_mask.clear();
        start_nodes.clear();
    }

    void reserve(int32_t n_ants, int32_t max_decisions) {
        starts.reserve(n_ants + 1);
        curr_nodes.reserve(max_decisions);
        chosen_nodes.reserve(max_decisions);
        is_stochastic.reserve(max_decisions);
        used_uniform.reserve(max_decisions);
        pick_j.reserve(max_decisions);
        valid_mask.reserve(max_decisions);
        start_nodes.reserve(n_ants);
    }
};


// ============================================================================
// Sample result structure
// ============================================================================

struct SampleResult {
    std::vector<float> costs;                    // (n_ants,)
    std::vector<std::vector<int32_t>> routes;    // (n_ants, n) - each route without repetition
    MFACOTraceBatch traces;                      // only populated if require_prob=true

    void clear() {
        costs.clear();
        routes.clear();
        traces.clear();
    }
};


// ============================================================================
// MFACO_TSP class - main solver
// ============================================================================

class MFACO_TSP {
public:
    // Configuration
    int32_t n;              // number of nodes
    int32_t n_ants;
    int32_t k;              // candidate list size
    int32_t bl;             // backup list size
    int32_t min_new_edges;
    float rho;              // pheromone decay (1 - evaporation rate)
    float alpha;            // pheromone exponent
    float p_best;           // for tau limits
    bool use_local_search;
    bool random_mode;       // uniform random selection instead of roulette
    bool disable_heuristic; // if true, set eta=1 so sampling uses pheromone (+ residual) only

    // State arrays
    std::vector<float> distances;           // (n, n) row-major
    std::vector<int32_t> nn_list;           // (n, k) row-major: nearest neighbors
    std::vector<int32_t> backup_list;       // (n, bl) row-major: backup neighbors
    std::vector<float> pheromone_sparse;    // (n, k) row-major: pheromone on candidate edges
    std::vector<float> heuristic_sparse;    // (n, k) row-major: 1/dist for candidate edges
    std::vector<int32_t> nn_pos;            // (n, n) row-major: nn_pos[u,v] = j if nn_list[u,j]=v, else -1

    // Solution state
    std::vector<int32_t> source_route;      // (n,) current source solution
    float source_cost;
    std::vector<int32_t> best_route;        // (n,) best solution found
    float best_cost;
    float tau_min;
    float tau_max;

    // Source positions cache
    std::vector<int32_t> source_positions;  // (n,) source_positions[v] = position of v in source_route

    // RNG
    Xoshiro128Plus rng_;

    // ========================================================================
    // Constructor
    // ========================================================================
    
    MFACO_TSP(
        const float* dist_ptr,      // (n, n) row-major
        int32_t n,
        int32_t n_ants,
        int32_t cand_list_size = 32,
        int32_t backup_list_size = 32,
        int32_t min_new_edges = 8,
        float decay = 0.9f,
        float alpha = 1.0f,
        float p_best = 0.05f,
        bool use_local_search = true,
        bool random_mode = false,
        bool disable_heuristic = false
    );

    // ========================================================================
    // API Methods
    // ========================================================================

    /**
     * Seed the RNG for reproducibility.
     */
    void seed_rng(uint64_t seed);

    /**
     * Sample solutions from all ants.
     * 
     * @param require_prob If true, record traces for log-prob replay.
        * @param parallel_traced If true and require_prob=true, parallelize traced sampling.
        *        Traces are generated per-ant and merged after; RNG is per-ant.
     * @param residual_logits Optional (n, k) neural residual logits. Pass nullptr if not used.
     * @param result Output structure filled with costs, routes, and traces.
     */
        void sample(bool require_prob, const float* residual_logits, SampleResult& result, bool parallel_traced = false);

    /**
     * Update pheromone: evaporate + deposit on best route.
     * Updates source solution and tau limits.
     * 
     * @param best_flat Route as (n+1,) array where last == first.
     * @param best_cost Cost of the route.
     */
    void update_pheromone(const int32_t* best_flat, float best_cost);

    /**
     * Load state from snapshot arrays. Matches Python load_snapshot().
     */
    void load_snapshot(
        const float* pheromone_ptr,     // (n, k)
        const int32_t* source_route_ptr,// (n,)
        float source_cost,
        const int32_t* best_route_ptr,  // (n,)
        float best_cost,
        float tau_min,
        float tau_max,
        const int32_t* nn_list_ptr,     // (n, k)
        const int32_t* backup_list_ptr  // (n, bl)
    );

    /**
     * Sync pheromone from external numpy array (for compatibility with Python).
     */
    void set_pheromone(const float* pheromone_ptr);

    /**
     * Get raw pointers to internal arrays for numpy views.
     */
    float* pheromone_data() { return pheromone_sparse.data(); }
    int32_t* nn_list_data() { return nn_list.data(); }
    int32_t* backup_list_data() { return backup_list.data(); }
    float* heuristic_data() { return heuristic_sparse.data(); }
    int32_t* source_route_data() { return source_route.data(); }
    int32_t* best_route_data() { return best_route.data(); }
    int32_t* nn_pos_data() { return nn_pos.data(); }
    const float* distances_data() const { return distances.data(); }

private:
    // ========================================================================
    // Internal methods
    // ========================================================================

    /**
     * Build nearest neighbor and backup lists from distances.
     */
    void build_nn_lists();

    /**
     * Build nn_pos mapping from nn_list.
     */
    void build_nn_pos();

    /**
     * Build heuristic_sparse from distances and nn_list.
     */
    void build_heuristic();

    /**
     * Build initial greedy NN tour.
     */
    void build_initial_tour();

    /**
     * Calculate tau limits using MMAS candidate-list formula.
     */
    std::pair<float, float> calc_trail_limits_cl(float solution_cost) const;

    /**
     * Sample a single ant's solution (fast mode, no tracing).
     */
    float sample_ant_fast(
        const float* probmat,           // (n, k) precomputed weights
        int32_t start_node,
        std::vector<int32_t>& route_out,
        std::vector<int32_t>& checklist,
        Xoshiro128Plus& rng
    );

    /**
     * Sample a single ant's solution with trace recording.
     */
    float sample_ant_traced(
        const float* probmat,           // (n, k) precomputed weights
        int32_t start_node,
        std::vector<int32_t>& route_out,
        std::vector<int32_t>& checklist,
        MFACOTrace& trace,
        Xoshiro128Plus& rng
    );

    /**
     * Select next node using roulette wheel on candidate list.
     * Returns (chosen_node, is_stochastic, used_uniform).
     */
    std::tuple<int32_t, bool, bool> select_next_node(
        int32_t curr,
        const float* probmat_row,       // probmat[curr*k : curr*k + k]
        const uint8_t* visited,
        Xoshiro128Plus& rng,
        int16_t& out_pick_j,
        uint64_t& out_valid_mask
    );

    /**
     * Relocate node to be successor of target in route.
     * Updates route, positions, and returns cost delta.
     */
    float relocate_node(
        int32_t target,
        int32_t node,
        std::vector<int32_t>& route,
        std::vector<int32_t>& positions
    );

    /**
     * 2-opt local search using checklist and nearest neighbor optimization.
     */
    float two_opt_nn(
        std::vector<int32_t>& route,
        std::vector<int32_t>& positions,
        std::vector<int32_t>& checklist
    );

    /**
     * Get route cost.
     */
    float get_route_cost(const std::vector<int32_t>& route) const;

    /**
     * Check if edge (a, b) exists in route (undirected).
     */
    bool contains_edge(int32_t a, int32_t b, const std::vector<int32_t>& positions) const;

    /**
     * Get successor of node in route.
     */
    int32_t get_succ(int32_t node, const std::vector<int32_t>& route, const std::vector<int32_t>& positions) const;

    /**
     * Get predecessor of node in route.
     */
    int32_t get_pred(int32_t node, const std::vector<int32_t>& route, const std::vector<int32_t>& positions) const;

    /**
     * Flip route section for 2-opt.
     */
    void flip_route_section(
        int32_t start_node,
        int32_t end_node,
        std::vector<int32_t>& route,
        std::vector<int32_t>& positions
    );

    /**
     * Compute probability matrix: tau^alpha * eta * exp(residual).
     */
    void compute_probmat(const float* residual_logits, std::vector<float>& probmat);
};

} // namespace mfaco

/**
 * pybind11 bindings for MFACO Training Module
 * 
 * Exposes MFACO_TSP and MFACOTrace to Python with numpy array views.
 * Module name: faco_cpp (to avoid conflict with existing faco_tsp.py)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include "mfaco_train.h"

namespace py = pybind11;
using namespace mfaco;

// ============================================================================
// Helper: create numpy array view from vector (no copy)
// ============================================================================

template<typename T>
py::array_t<T> make_view(T* data, std::vector<py::ssize_t> shape) {
    // Compute strides (row-major)
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t stride = sizeof(T);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return py::array_t<T>(shape, strides, data, py::none());
}

template<typename T>
py::array_t<T> make_1d_view(T* data, py::ssize_t len) {
    return make_view<T>(data, {len});
}

template<typename T>
py::array_t<T> make_2d_view(T* data, py::ssize_t rows, py::ssize_t cols) {
    return make_view<T>(data, {rows, cols});
}


// ============================================================================
// MFACOTraceBatch Python wrapper
// ============================================================================

class PyMFACOTrace {
public:
    // The trace batch from C++
    MFACOTraceBatch batch;
    int32_t n_ants;

    PyMFACOTrace() : n_ants(0) {}

    // Property accessors returning numpy views
    py::array_t<int32_t> get_starts() {
        return make_1d_view(batch.starts.data(), batch.starts.size());
    }

    py::array_t<int32_t> get_curr_nodes() {
        return make_1d_view(batch.curr_nodes.data(), batch.curr_nodes.size());
    }

    py::array_t<int32_t> get_chosen_nodes() {
        return make_1d_view(batch.chosen_nodes.data(), batch.chosen_nodes.size());
    }

    py::array_t<uint8_t> get_is_stochastic() {
        return make_1d_view(batch.is_stochastic.data(), batch.is_stochastic.size());
    }

    py::array_t<uint8_t> get_used_uniform() {
        return make_1d_view(batch.used_uniform.data(), batch.used_uniform.size());
    }

    py::array_t<int16_t> get_pick_j() {
        return make_1d_view(batch.pick_j.data(), batch.pick_j.size());
    }

    py::array_t<uint64_t> get_valid_mask() {
        return make_1d_view(batch.valid_mask.data(), batch.valid_mask.size());
    }

    py::array_t<int32_t> get_start_nodes() {
        return make_1d_view(batch.start_nodes.data(), batch.start_nodes.size());
    }

    int32_t n_decisions() const {
        return static_cast<int32_t>(batch.curr_nodes.size());
    }

    // Convert batch to list of Python-compatible trace dicts for replay_logp_batch
    py::list to_trace_list() {
        py::list result;
        for (int32_t a = 0; a < n_ants; ++a) {
            py::dict trace;
            trace["start_node"] = batch.start_nodes[a];
            
            int32_t start = batch.starts[a];
            int32_t end = batch.starts[a + 1];
            int32_t count = end - start;
            
            py::list curr, chosen, is_stoch, used_unif;
            for (int32_t i = start; i < end; ++i) {
                curr.append(batch.curr_nodes[i]);
                chosen.append(batch.chosen_nodes[i]);
                is_stoch.append(batch.is_stochastic[i] != 0);
                used_unif.append(batch.used_uniform[i] != 0);
            }
            
            trace["curr_nodes"] = curr;
            trace["chosen_nodes"] = chosen;
            trace["is_stochastic"] = is_stoch;
            trace["used_uniform_fallback"] = used_unif;
            
            result.append(trace);
        }
        return result;
    }
};


// ============================================================================
// MFACO_TSP Python wrapper
// ============================================================================

class PyMFACO_TSP {
public:
    std::unique_ptr<MFACO_TSP> solver;

    PyMFACO_TSP(
        py::array_t<float, py::array::c_style | py::array::forcecast> distances,
        int32_t n_ants,
        int32_t cand_list_size = 32,
        int32_t backup_list_size = 32,
        int32_t min_new_edges = 8,
        float decay = 0.9f,
        float alpha = 1.0f,
        float p_best = 0.05f,
        bool use_local_search = true,
        bool random_mode = false,
        const std::string& device = "cpu"  // ignored for now, C++ is CPU-only
    ) {
        auto buf = distances.request();
        if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
            throw std::runtime_error("distances must be a square 2D array");
        }
        int32_t n = static_cast<int32_t>(buf.shape[0]);
        const float* dist_ptr = static_cast<const float*>(buf.ptr);

        solver = std::make_unique<MFACO_TSP>(
            dist_ptr, n, n_ants,
            cand_list_size, backup_list_size, min_new_edges,
            decay, alpha, p_best,
            use_local_search, random_mode
        );
    }

    // ========================================================================
    // Properties (numpy views, no copy)
    // ========================================================================

    int32_t get_n() const { return solver->n; }
    int32_t get_n_ants() const { return solver->n_ants; }
    int32_t get_k() const { return solver->k; }
    int32_t get_bl() const { return solver->bl; }
    int32_t get_min_new_edges() const { return solver->min_new_edges; }
    float get_rho() const { return solver->rho; }
    float get_alpha() const { return solver->alpha; }
    float get_p_best() const { return solver->p_best; }
    bool get_use_local_search() const { return solver->use_local_search; }
    bool get_random_mode() const { return solver->random_mode; }
    float get_source_cost() const { return solver->source_cost; }
    float get_best_cost() const { return solver->best_cost; }
    float get_tau_min() const { return solver->tau_min; }
    float get_tau_max() const { return solver->tau_max; }

    py::array_t<float> get_pheromone_sparse_np() {
        return make_2d_view(solver->pheromone_data(), solver->n, solver->k);
    }

    py::array_t<int32_t> get_nn_list() {
        return make_2d_view(solver->nn_list_data(), solver->n, solver->k);
    }

    py::array_t<int32_t> get_backup_list() {
        return make_2d_view(solver->backup_list_data(), solver->n, solver->bl);
    }

    py::array_t<float> get_heuristic_sparse_np() {
        return make_2d_view(solver->heuristic_data(), solver->n, solver->k);
    }

    py::array_t<int32_t> get_source_route() {
        return make_1d_view(solver->source_route_data(), solver->n);
    }

    py::array_t<int32_t> get_best_route() {
        return make_1d_view(solver->best_route_data(), solver->n);
    }

    py::array_t<int32_t> get_nn_pos() {
        return make_2d_view(solver->nn_pos_data(), solver->n, solver->n);
    }

    py::array_t<int32_t> get_source_positions() {
        return make_1d_view(solver->source_positions.data(), solver->n);
    }

    // ========================================================================
    // API Methods
    // ========================================================================

    void seed_rng(uint64_t seed) {
        solver->seed_rng(seed);
    }

    /**
     * Sample solutions from all ants.
     * 
     * Returns: (costs, flats, touched_list, logps_nondiff, traces)
     *   - costs: np.ndarray (n_ants,) float32
     *   - flats: list of np.ndarray (n+1,) int32, each with last == first
     *   - touched_list: list of np.ndarray (placeholder, empty for now)
     *   - logps_nondiff: None (placeholder)
     *   - traces: PyMFACOTrace if require_prob else None
     */
    py::tuple sample(
        float invtemp = 1.0f,  // unused for now, heuristic is pre-baked
        bool require_prob = false,
        py::object residual_logits_obj = py::none(),
        bool parallel_traced = false
    ) {
        const float* residual_ptr = nullptr;
        py::array_t<float> residual_arr;
        
        if (!residual_logits_obj.is_none()) {
            residual_arr = residual_logits_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto buf = residual_arr.request();
            if (buf.ndim != 2 || buf.shape[0] != solver->n || buf.shape[1] != solver->k) {
                throw std::runtime_error("residual_logits must be shape (n, k)");
            }
            residual_ptr = static_cast<const float*>(buf.ptr);
        }

        SampleResult result;
        
        {
            py::gil_scoped_release release;
            solver->sample(require_prob, residual_ptr, result, parallel_traced);
        }

        // Convert costs to numpy
        py::array_t<float> costs(solver->n_ants);
        auto costs_buf = costs.mutable_unchecked<1>();
        for (int32_t a = 0; a < solver->n_ants; ++a) {
            costs_buf(a) = result.costs[a];
        }

        // Convert routes to list of flats (n+1, last == first)
        py::list flats;
        for (int32_t a = 0; a < solver->n_ants; ++a) {
            py::array_t<int32_t> flat(solver->n + 1);
            auto flat_buf = flat.mutable_unchecked<1>();
            for (int32_t i = 0; i < solver->n; ++i) {
                flat_buf(i) = result.routes[a][i];
            }
            flat_buf(solver->n) = result.routes[a][0];  // close the cycle
            flats.append(flat);
        }

        // Placeholder touched_list (empty arrays)
        py::list touched_list;
        for (int32_t a = 0; a < solver->n_ants; ++a) {
            touched_list.append(py::array_t<int32_t>(0));
        }

        // Traces
        py::object traces_obj = py::none();
        if (require_prob) {
            auto traces = std::make_unique<PyMFACOTrace>();
            traces->batch = std::move(result.traces);
            traces->n_ants = solver->n_ants;
            traces_obj = py::cast(std::move(traces));
        }

        return py::make_tuple(costs, flats, touched_list, py::none(), traces_obj);
    }

    /**
     * Update pheromone: evaporate + deposit.
     */
    void update_pheromone_from_flat(
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> best_flat,
        float best_cost
    ) {
        auto buf = best_flat.request();
        if (buf.ndim != 1 || buf.shape[0] < solver->n) {
            throw std::runtime_error("best_flat must be at least length n");
        }
        const int32_t* flat_ptr = static_cast<const int32_t*>(buf.ptr);

        py::gil_scoped_release release;
        solver->update_pheromone(flat_ptr, best_cost);
    }

    /**
     * Load snapshot from arrays.
     */
    void load_snapshot(
        py::array_t<float, py::array::c_style | py::array::forcecast> pheromone,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> source_route,
        float source_cost,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> best_route,
        float best_cost,
        float tau_min,
        float tau_max,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> nn_list,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> backup_list
    ) {
        auto phe_buf = pheromone.request();
        auto src_buf = source_route.request();
        auto best_buf = best_route.request();
        auto nn_buf = nn_list.request();
        auto bl_buf = backup_list.request();

        solver->load_snapshot(
            static_cast<const float*>(phe_buf.ptr),
            static_cast<const int32_t*>(src_buf.ptr),
            source_cost,
            static_cast<const int32_t*>(best_buf.ptr),
            best_cost,
            tau_min,
            tau_max,
            static_cast<const int32_t*>(nn_buf.ptr),
            static_cast<const int32_t*>(bl_buf.ptr)
        );
    }

    /**
     * Sync pheromone from external array.
     */
    void set_pheromone(py::array_t<float, py::array::c_style | py::array::forcecast> pheromone) {
        auto buf = pheromone.request();
        solver->set_pheromone(static_cast<const float*>(buf.ptr));
    }

    /**
     * Sync pheromone TO torch (no-op in C++ version, just for API compat).
     */
    void sync_pheromone_to_torch() {
        // No-op - C++ doesn't maintain torch tensors
    }
};


// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(faco_cpp, m) {
    m.doc() = "C++ MFACO Training Module for fast neural-guided ACO training";

    // OpenMP controls (global to the process)
    m.def(
        "set_num_threads",
        [](int n_threads) {
            if (n_threads <= 0) {
                throw std::runtime_error("n_threads must be > 0");
            }
            omp_set_num_threads(n_threads);
        },
        py::arg("n_threads"),
        "Set OpenMP thread count for faco_cpp (process-global)."
    );

    m.def(
        "get_max_threads",
        []() { return omp_get_max_threads(); },
        "Get current OpenMP max threads."
    );

    m.def(
        "get_num_procs",
        []() { return omp_get_num_procs(); },
        "Get number of processors visible to OpenMP runtime."
    );

    m.def(
        "set_dynamic",
        [](bool enabled) { omp_set_dynamic(enabled ? 1 : 0); },
        py::arg("enabled"),
        "Enable/disable OpenMP dynamic adjustment of threads."
    );

    m.def(
        "get_dynamic",
        []() { return omp_get_dynamic() != 0; },
        "Get whether OpenMP dynamic threads are enabled."
    );

    // MFACOTrace
    py::class_<PyMFACOTrace>(m, "MFACOTrace")
        .def(py::init<>())
        .def_property_readonly("starts", &PyMFACOTrace::get_starts)
        .def_property_readonly("curr_nodes", &PyMFACOTrace::get_curr_nodes)
        .def_property_readonly("chosen_nodes", &PyMFACOTrace::get_chosen_nodes)
        .def_property_readonly("is_stochastic", &PyMFACOTrace::get_is_stochastic)
        .def_property_readonly("used_uniform", &PyMFACOTrace::get_used_uniform)
        .def_property_readonly("pick_j", &PyMFACOTrace::get_pick_j)
        .def_property_readonly("valid_mask", &PyMFACOTrace::get_valid_mask)
        .def_property_readonly("start_nodes", &PyMFACOTrace::get_start_nodes)
        .def_property_readonly("n_decisions", &PyMFACOTrace::n_decisions)
        .def_property_readonly("n_ants", [](const PyMFACOTrace& t) { return t.n_ants; })
        .def("to_trace_list", &PyMFACOTrace::to_trace_list,
             "Convert batch trace to list of dict traces for Python replay_logp_batch");

    // MFACO_TSP
    py::class_<PyMFACO_TSP>(m, "MFACO_TSP")
        .def(py::init<
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            int32_t, int32_t, int32_t, int32_t, float, float, float, bool, bool, const std::string&
        >(),
            py::arg("distances"),
            py::arg("n_ants"),
            py::arg("cand_list_size") = 32,
            py::arg("backup_list_size") = 32,
            py::arg("min_new_edges") = 8,
            py::arg("decay") = 0.9f,
            py::arg("alpha") = 1.0f,
            py::arg("p_best") = 0.05f,
            py::arg("use_local_search") = true,
            py::arg("random_mode") = false,
            py::arg("device") = "cpu"
        )
        // Properties
        .def_property_readonly("n", &PyMFACO_TSP::get_n)
        .def_property_readonly("n_ants", &PyMFACO_TSP::get_n_ants)
        .def_property_readonly("k", &PyMFACO_TSP::get_k)
        .def_property_readonly("bl", &PyMFACO_TSP::get_bl)
        .def_property_readonly("min_new_edges", &PyMFACO_TSP::get_min_new_edges)
        .def_property_readonly("rho", &PyMFACO_TSP::get_rho)
        .def_property_readonly("alpha", &PyMFACO_TSP::get_alpha)
        .def_property_readonly("p_best", &PyMFACO_TSP::get_p_best)
        .def_property_readonly("use_local_search", &PyMFACO_TSP::get_use_local_search)
        .def_property_readonly("random_mode", &PyMFACO_TSP::get_random_mode)
        .def_property_readonly("source_cost", &PyMFACO_TSP::get_source_cost)
        .def_property_readonly("best_cost", &PyMFACO_TSP::get_best_cost)
        .def_property_readonly("tau_min", &PyMFACO_TSP::get_tau_min)
        .def_property_readonly("tau_max", &PyMFACO_TSP::get_tau_max)
        // Array properties (views)
        .def_property_readonly("pheromone_sparse_np", &PyMFACO_TSP::get_pheromone_sparse_np)
        .def_property_readonly("nn_list", &PyMFACO_TSP::get_nn_list)
        .def_property_readonly("backup_list", &PyMFACO_TSP::get_backup_list)
        .def_property_readonly("heuristic_sparse_np", &PyMFACO_TSP::get_heuristic_sparse_np)
        .def_property_readonly("source_route", &PyMFACO_TSP::get_source_route)
        .def_property_readonly("best_route", &PyMFACO_TSP::get_best_route)
        .def_property_readonly("nn_pos", &PyMFACO_TSP::get_nn_pos)
        .def_property_readonly("source_positions", &PyMFACO_TSP::get_source_positions)
        // Methods
        .def("seed_rng", &PyMFACO_TSP::seed_rng, py::arg("seed"))
        .def("sample", &PyMFACO_TSP::sample,
            py::arg("invtemp") = 1.0f,
            py::arg("require_prob") = false,
            py::arg("residual_logits") = py::none(),
            py::arg("parallel_traced") = false,
            "Sample solutions from all ants. Returns (costs, flats, touched_list, logps_nondiff, traces)")
        .def("_update_pheromone_from_flat", &PyMFACO_TSP::update_pheromone_from_flat,
            py::arg("best_flat"), py::arg("best_cost"),
            "Update pheromone: evaporate + deposit on best route")
        .def("load_snapshot", &PyMFACO_TSP::load_snapshot,
            py::arg("pheromone"), py::arg("source_route"), py::arg("source_cost"),
            py::arg("best_route"), py::arg("best_cost"),
            py::arg("tau_min"), py::arg("tau_max"),
            py::arg("nn_list"), py::arg("backup_list"),
            "Load state from snapshot arrays")
        .def("set_pheromone", &PyMFACO_TSP::set_pheromone, py::arg("pheromone"))
        .def("sync_pheromone_to_torch", &PyMFACO_TSP::sync_pheromone_to_torch,
            "No-op for API compatibility (C++ doesn't maintain torch tensors)");
}

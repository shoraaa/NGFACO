#!/usr/bin/env python
"""
Benchmark script to compare performance of faco.py vs faco_tsp.py
"""
import torch
import numpy as np
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Benchmark FACO implementations")
    parser.add_argument("--n", type=int, default=100, help="Problem size")
    parser.add_argument("--n_ants", type=int, default=20, help="Number of ants")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per instance")
    parser.add_argument("--n_instances", type=int, default=5, help="Number of instances")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations for JIT")
    args = parser.parse_args()

    print("=" * 60)
    print("Performance Comparison: faco_tsp.py vs faco.py")
    print("=" * 60)
    print(f"Config: n={args.n}, ants={args.n_ants}, iters={args.iterations}, instances={args.n_instances}")

    # Generate random TSP instances
    coords_list = [torch.rand(args.n, 2) for _ in range(args.n_instances)]
    dist_list = [torch.cdist(c, c) for c in coords_list]

    # ============ Test faco.py ============
    print("\n[1] Testing faco.py (original optimized implementation)")
    from faco import FACO as FACO_Original

    # Warmup JIT
    print("    Warming up JIT...")
    solver = FACO_Original(dist_list[0], n_ants=args.n_ants, k_sparse=32)
    for _ in range(args.warmup):
        solver.sample(inference=True)

    # Benchmark
    times_faco = []
    costs_faco = []
    for i, dist in enumerate(dist_list):
        solver = FACO_Original(dist, n_ants=args.n_ants, k_sparse=32)
        t0 = time.time()
        for _ in range(args.iterations):
            costs, log_probs, paths = solver.sample(inference=True)
            solver.update_pheromone(paths, costs)
        t1 = time.time()
        times_faco.append((t1 - t0) / args.iterations * 1000)
        costs_faco.append(float(costs.min()))
        print(f"    Instance {i+1}: {times_faco[-1]:.2f} ms/iter, best={costs_faco[-1]:.4f}")

    print(f"  Avg time per iteration: {np.mean(times_faco):.2f} ms")
    print(f"  Avg best cost: {np.mean(costs_faco):.4f}")

    # ============ Test faco_tsp.py (fast path) ============
    print("\n[2] Testing faco_tsp.py FAST path (inference)")
    from faco_tsp import MFACO_TSP

    # Warmup JIT
    print("    Warming up JIT...")
    solver = MFACO_TSP(dist_list[0], n_ants=args.n_ants, min_new_edges=8, use_local_search=True)
    for _ in range(args.warmup):
        solver._sample_fast(invtemp=1.0)

    # Benchmark fast path
    times_mfaco_fast = []
    costs_mfaco_fast = []
    for i, dist in enumerate(dist_list):
        solver = MFACO_TSP(dist, n_ants=args.n_ants, min_new_edges=8, use_local_search=True)
        t0 = time.time()
        for _ in range(args.iterations):
            costs, flats, _, _, _ = solver._sample_fast(invtemp=1.0)
            best_idx = int(np.argmin(costs))
            solver._update_pheromone_from_flat(flats[best_idx], costs[best_idx])
        t1 = time.time()
        times_mfaco_fast.append((t1 - t0) / args.iterations * 1000)
        costs_mfaco_fast.append(float(min(costs)))
        print(f"    Instance {i+1}: {times_mfaco_fast[-1]:.2f} ms/iter, best={costs_mfaco_fast[-1]:.4f}")

    print(f"  Avg time per iteration: {np.mean(times_mfaco_fast):.2f} ms")
    print(f"  Avg best cost: {np.mean(costs_mfaco_fast):.4f}")

    # ============ Test faco_tsp.py (traced path) ============
    print("\n[3] Testing faco_tsp.py TRACED path (training)")

    # Warmup JIT for traced path
    print("    Warming up JIT...")
    solver = MFACO_TSP(dist_list[0], n_ants=args.n_ants, min_new_edges=8, use_local_search=True)
    for _ in range(args.warmup):
        solver._sample_with_trace(invtemp=1.0)

    # Use fewer iterations for traced path (slower)
    traced_iters = min(20, args.iterations)
    traced_instances = min(2, args.n_instances)

    times_mfaco_traced = []
    costs_mfaco_traced = []
    for i, dist in enumerate(dist_list[:traced_instances]):
        solver = MFACO_TSP(dist, n_ants=args.n_ants, min_new_edges=8, use_local_search=True)
        t0 = time.time()
        for _ in range(traced_iters):
            costs, flats, _, logps, traces = solver._sample_with_trace(invtemp=1.0)
            best_idx = int(np.argmin(costs))
            solver._update_pheromone_from_flat(flats[best_idx], costs[best_idx])
        t1 = time.time()
        times_mfaco_traced.append((t1 - t0) / traced_iters * 1000)
        costs_mfaco_traced.append(float(min(costs)))
        print(f"    Instance {i+1}: {times_mfaco_traced[-1]:.2f} ms/iter, best={costs_mfaco_traced[-1]:.4f}")
        print(f"      Sample logps: {logps[:3].tolist()}")

    print(f"  Avg time per iteration: {np.mean(times_mfaco_traced):.2f} ms")
    print(f"  Avg best cost: {np.mean(costs_mfaco_traced):.4f}")

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  faco.py:              {np.mean(times_faco):7.2f} ms/iter")
    print(f"  faco_tsp.py (fast):   {np.mean(times_mfaco_fast):7.2f} ms/iter")
    print(f"  faco_tsp.py (traced): {np.mean(times_mfaco_traced):7.2f} ms/iter")
    print()
    
    speedup_fast = np.mean(times_faco) / np.mean(times_mfaco_fast)
    speedup_traced = np.mean(times_faco) / np.mean(times_mfaco_traced)
    
    if speedup_fast > 1:
        print(f"  faco_tsp fast is {speedup_fast:.2f}x FASTER than faco.py")
    else:
        print(f"  faco_tsp fast is {1/speedup_fast:.2f}x SLOWER than faco.py")
    
    if speedup_traced > 1:
        print(f"  faco_tsp traced is {speedup_traced:.2f}x FASTER than faco.py")
    else:
        print(f"  faco_tsp traced is {1/speedup_traced:.2f}x SLOWER than faco.py")


if __name__ == "__main__":
    main()

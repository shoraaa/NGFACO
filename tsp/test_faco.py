#!/usr/bin/env python3
"""
test_faco.py
Test pure FACO (Fast Ant Colony Optimization) performance without neural network.

Usage:
    python test_faco.py --n 100 --n_ants 20 --iterations 50 --n_instances 10
    
    # Late-iteration comparison (stagnation analysis for roulette vs random)
    python test_faco.py --late_comparison --n 100 --warmup_iters 50 --continuation_iters 20
    
    # Neural-enhanced comparison - end-to-end (default)
    python test_faco.py --neural_comparison --checkpoint checkpoints/best_model.pt --n 100 --iterations 50
    
    # Neural-enhanced comparison - late-iteration mode (with warmup)
    python test_faco.py --neural_comparison --checkpoint checkpoints/best_model.pt --n 100 --warmup --warmup_iters 50 --iterations 70
"""

import argparse
import copy
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict, Any

import numpy as np
import torch
from torch_geometric.data import Data as PyGData

from faco_tsp_unified import MFACO_TSP, has_cpp_backend, set_faco_cpp_threads
from utils import gen_distance_matrix


@dataclass
class TestConfig:
    # Instance selection
    n: int = 1000
    n_instances: int = 100
    instances: Optional[List[Dict[str, Any]]] = None

    # FACO parameters
    n_ants: int = 64
    iterations: int = 1000
    cand_list_size: int = 32
    backup_list_size: int = 32
    min_new_edges: int = 8
    decay: float = 0.9
    use_local_search: bool = True
    random_mode: bool = False

    # Late comparison / warmup
    warmup_iters: int = 50
    continuation_iters: int = 20
    warmup: bool = False

    # Neural comparison
    checkpoint: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)
    gamma: float = 0.2

    # Runtime
    device: str = "cpu"
    seed: int = 42
    verbose: bool = True

    # C++ backend threading (OpenMP)
    faco_cpp_threads: Optional[int] = None


def _infer_n_and_count(instances: Optional[List[Dict[str, Any]]], fallback_n: int, fallback_count: int) -> tuple[int | str, int]:
    if not instances:
        return fallback_n, fallback_count
    ns = [int(inst["coords"].shape[0]) for inst in instances]
    inferred_n: int | str = ns[0] if all(v == ns[0] for v in ns) else "mixed"
    return inferred_n, len(instances)


def _get_iterable_instances(config: TestConfig) -> List[Dict[str, Any]]:
    if config.instances is not None:
        return list(config.instances)
    return list(
        _iter_problem_instances(
            mode="random",
            device=config.device,
            n=config.n,
            n_instances=config.n_instances,
        )
    )


def _load_net_from_checkpoint(checkpoint_path: str, device: str, verbose: bool) -> tuple[object, dict, str]:
    """Load Net + checkpoint meta; returns (net, ckpt, resolved_path)."""
    from net import Net

    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    if verbose:
        print(f"\nLoading model from {resolved_checkpoint_path}...")

    ckpt = torch.load(resolved_checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    net = Net(
        node_feats=config.get("node_feats", 2),
        edge_feats=config.get("edge_feats", 1),
        extra_feats=config.get("extra_feats", 4),
        units=config.get("units", 64),
        depth=config.get("encoder_depth", 4),
        scorer_hidden=config.get("scorer_hidden", 64),
    ).to(device)
    net.load_state_dict(ckpt["net_state_dict"])
    net.eval()
    if verbose:
        print(
            f"Model loaded (epoch {ckpt.get('epoch', '?')} - improve {ckpt.get('improve', '?')})"
        )
    return net, ckpt, resolved_checkpoint_path


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve a checkpoint path and provide a sensible fallback.

    If the given path doesn't exist, this will try:
    - Interpreting it relative to the repo root (parent of the `tsp/` folder)
    - If it points inside a directory (or a missing file inside one), selecting an
      available `*.pt` checkpoint (prefers highest `epoch_*.pt`, else newest mtime).
    """
    raw = Path(checkpoint_path)
    if raw.exists():
        if raw.is_dir():
            # Treat directory as "pick best .pt inside".
            checkpoint_path = str(raw)
        else:
            return str(raw)

    repo_root = Path(__file__).resolve().parents[1]
    rel_to_repo = repo_root / raw
    if rel_to_repo.exists():
        if rel_to_repo.is_dir():
            checkpoint_path = str(rel_to_repo)
        else:
            return str(rel_to_repo)

    search_dirs: list[Path] = []
    if raw.suffix == ".pt":
        search_dirs.append(raw.parent)
        search_dirs.append((repo_root / raw).parent)
    else:
        search_dirs.append(raw)
        search_dirs.append(repo_root / raw)

    for d in search_dirs:
        if not d.exists() or not d.is_dir():
            continue

        candidates = sorted(d.glob("*.pt"))
        if not candidates:
            continue

        epoch_candidates: list[tuple[int, Path]] = []
        for c in candidates:
            m = re.match(r"epoch_(\d+)\.pt$", c.name)
            if m:
                epoch_candidates.append((int(m.group(1)), c))

        if epoch_candidates:
            chosen = max(epoch_candidates, key=lambda t: t[0])[1]
        else:
            chosen = max(candidates, key=lambda p: p.stat().st_mtime)

        print(
            f"[warning] Checkpoint not found: {checkpoint_path}. "
            f"Falling back to: {chosen}"
        )
        return str(chosen)

    tried = [str(raw), str(rel_to_repo)]
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. Tried: {tried}. "
        f"If you intended to use the latest checkpoint, point --checkpoint to a "
        f"directory containing '*.pt' files (e.g. 'checkpoints/')."
    )


def _read_tsplib_dimension(tsp_path: Path) -> Optional[int]:
    try:
        with tsp_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(2000):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                if s.upper().startswith("DIMENSION"):
                    # DIMENSION: 52  |  DIMENSION : 52
                    parts = re.split(r"[:\s]+", s)
                    for p in reversed(parts):
                        if p.isdigit():
                            return int(p)
    except OSError:
        return None
    return None


def _read_tsplib_edge_weight_type(tsp_path: Path) -> Optional[str]:
    try:
        with tsp_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(2000):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                if s.upper().startswith("EDGE_WEIGHT_TYPE"):
                    # EDGE_WEIGHT_TYPE: EUC_2D
                    if ":" in s:
                        return s.split(":", 1)[1].strip().upper()
                    parts = s.split()
                    return parts[-1].strip().upper() if parts else None
    except OSError:
        return None
    return None


def _parse_tsplib_coords(tsp_path: Path) -> Tuple[str, int, np.ndarray]:
    """Parse a TSPLIB .tsp file (NODE_COORD_SECTION).

    Supports common 2D coordinate instances with EDGE_WEIGHT_TYPE in:
    - EUC_2D
    - CEIL_2D
    - ATT
    - GEO
    """
    ew_type = _read_tsplib_edge_weight_type(tsp_path) or "EUC_2D"
    dim = _read_tsplib_dimension(tsp_path)

    coords: list[tuple[float, float]] = []
    in_section = False
    with tsp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            up = s.upper()
            if up.startswith("NODE_COORD_SECTION"):
                in_section = True
                continue
            if up.startswith("EOF") or up.startswith("DISPLAY_DATA_SECTION") or up.startswith("TOUR_SECTION"):
                break
            if not in_section:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            # id x y
            try:
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue
            coords.append((x, y))

    if dim is None:
        dim = len(coords)
    if dim != len(coords):
        # Some files may include extra whitespace/lines; still guard hard mismatch.
        if len(coords) < dim:
            raise ValueError(f"Failed to parse all coords from {tsp_path} (DIMENSION={dim}, got={len(coords)})")
        coords = coords[:dim]

    if ew_type not in {"EUC_2D", "CEIL_2D", "ATT", "GEO"}:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE={ew_type} in {tsp_path}")
    return ew_type, dim, np.asarray(coords, dtype=np.float32)


def _tsplib_distance_matrix(coords: np.ndarray, ew_type: str) -> np.ndarray:
    """Compute TSPLIB-style distance matrix from coordinates."""
    xy = coords.astype(np.float64, copy=False)
    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]

    if ew_type == "EUC_2D":
        d = np.sqrt(dx * dx + dy * dy)
        return np.rint(d).astype(np.float32)
    if ew_type == "CEIL_2D":
        d = np.sqrt(dx * dx + dy * dy)
        return np.ceil(d).astype(np.float32)
    if ew_type == "ATT":
        rij = np.sqrt((dx * dx + dy * dy) / 10.0)
        tij = np.rint(rij)
        dij = np.where(tij < rij, tij + 1.0, tij)
        return dij.astype(np.float32)

    if ew_type == "GEO":
        # TSPLIB GEO: coordinates are DDD.MM where MM are minutes.
        # Convert to radians: lat = pi * (deg + 5 * min / 3) / 180.
        def _geo_to_rad(v: np.ndarray) -> np.ndarray:
            deg = np.floor(v).astype(np.float64)
            minutes = v - deg
            return np.pi * (deg + (5.0 * minutes) / 3.0) / 180.0

        lat = _geo_to_rad(xy[:, 0])
        lon = _geo_to_rad(xy[:, 1])

        lat_i = lat[:, None]
        lat_j = lat[None, :]
        lon_i = lon[:, None]
        lon_j = lon[None, :]

        q1 = np.cos(lon_i - lon_j)
        q2 = np.cos(lat_i - lat_j)
        q3 = np.cos(lat_i + lat_j)
        R = 6378.388
        # Cast to float64 to keep numerical stability; clamp to [-1, 1] for acos.
        arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
        arg = np.clip(arg, -1.0, 1.0)
        dij = R * np.arccos(arg) + 1.0
        return np.floor(dij).astype(np.float32)

    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE={ew_type}")


def _resolve_best_known_path(instances_dir: Path) -> Optional[Path]:
    # Preference order:
    # 1) instances_dir/best-known.json
    # 2) repo_root/tsp/best-known.json
    repo_root = Path(__file__).resolve().parents[1]
    p1 = instances_dir / "best-known.json"
    if p1.exists():
        return p1
    p2 = repo_root / "tsp" / "best-known.json"
    if p2.exists():
        return p2
    return None


def _load_best_known(best_known_path: Optional[Path]) -> dict[str, float]:
    if best_known_path is None:
        return {}
    with best_known_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _select_tsplib_instances(
    instances_dir: Path,
    n_instances: int,
    instance: Optional[str] = None,
    max_n: Optional[int] = None,
    take_all: bool = False,
) -> List[Path]:
    if instance:
        p = Path(instance)
        if not p.suffix:
            p = p.with_suffix(".tsp")
        if not p.is_absolute():
            p = instances_dir / p
        if not p.exists():
            raise FileNotFoundError(f"TSPLIB instance not found: {p}")
        ew = _read_tsplib_edge_weight_type(p) or "EUC_2D"
        if ew.upper() != "EUC_2D":
            raise ValueError(
                f"TSPLIB instance {p.name} has EDGE_WEIGHT_TYPE={ew}. "
                "This script is currently restricted to EUC_2D only."
            )
        return [p]

    all_tsp = sorted(instances_dir.glob("*.tsp"))
    if not all_tsp:
        raise FileNotFoundError(f"No .tsp files found in {instances_dir}")

    scored: list[tuple[int, str, Path]] = []
    for p in all_tsp:
        dim = _read_tsplib_dimension(p)
        if dim is None:
            continue
        ew = _read_tsplib_edge_weight_type(p) or "EUC_2D"
        if ew.upper() != "EUC_2D":
            # Skip non-EUC_2D for now.
            continue
        if max_n is not None and dim > max_n:
            continue
        scored.append((dim, p.name, p))

    # Selection order:
    # - If max_n is set, users typically want instances close to that scale, so prefer larger dims first.
    # - Otherwise keep the historical behavior (smallest first).
    if max_n is not None:
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    else:
        scored.sort(key=lambda t: (t[0], t[1]))
    all_selected = [p for _, _, p in scored]
    if take_all:
        return all_selected
    return all_selected[: max(1, int(n_instances))]


def _iter_problem_instances(
    *,
    mode: str,
    device: str,
    n: int,
    n_instances: int,
    tsplib_paths: Optional[List[Path]] = None,
    best_known: Optional[dict[str, float]] = None,
) -> Iterable[Dict[str, Any]]:
    """Yield problem instances as dicts with coords/distances/name/best_known."""
    if mode == "random":
        for i in range(n_instances):
            coords, distances = generate_random_instance(n, device=device)
            yield {
                "name": f"random_{i+1}",
                "coords": coords,
                "distances": distances,
                "best_known": None,
            }
        return

    if mode != "tsplib":
        raise ValueError(f"Unknown instance mode: {mode}")

    if not tsplib_paths:
        return

    bk = best_known or {}
    for p in tsplib_paths:
        inst_name = p.stem
        ew_type, dim, coords_np = _parse_tsplib_coords(p)

        # Guard against accidental OOM: distance matrix is O(n^2).
        # (This script uses a full distance matrix for MFACO_TSP.)
        if dim > 5000:
            raise ValueError(
                f"TSPLIB instance {inst_name} has n={dim}, which is too large for a full distance matrix. "
                f"Use a smaller --max_n (e.g. 300/700/1500), or choose a smaller instance."
            )

        dist_np = _tsplib_distance_matrix(coords_np, ew_type)
        coords_t = torch.from_numpy(coords_np).to(device=device)
        dist_t = torch.from_numpy(dist_np).to(device=device)
        yield {
            "name": inst_name,
            "coords": coords_t,
            "distances": dist_t,
            "best_known": bk.get(inst_name),
        }


def generate_random_instance(n: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random TSP instance with n nodes.
    
    Args:
        n: Number of nodes
        device: Device to use
        
    Returns:
        coords: (n, 2) node coordinates in [0, 1]^2
        distances: (n, n) distance matrix
    """
    coords = torch.rand(n, 2, device=device)
    distances = gen_distance_matrix(coords)
    return coords, distances


def run_faco_test(
    config: TestConfig,
) -> dict:
    """
    Run pure FACO test on random TSP instances.
    
    Args:
        n: Number of nodes per instance
        n_ants: Number of ants
        n_iterations: Number of FACO iterations
        n_instances: Number of random instances to test
        cand_list_size: Candidate list size
        backup_list_size: Backup list size
        min_new_edges: Minimum new edges per ant
        decay: Pheromone decay rate
        use_local_search: Whether to use local 2-opt search
        random_mode: Whether to use uniform random selection instead of roulette
        device: Device to use
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with test results
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    iterable = _get_iterable_instances(config)
    inferred_n, inferred_count = _infer_n_and_count(iterable, config.n, config.n_instances)
    
    results = {
        "n": inferred_n,
        "n_ants": config.n_ants,
        "n_iterations": config.iterations,
        "n_instances": inferred_count,
        "cand_list_size": config.cand_list_size,
        "use_local_search": config.use_local_search,
        "random_mode": config.random_mode,
        "costs": [],
        "init_costs": [],
        "improvements": [],
        "times": [],
        "iter_times": [],
        "instance_names": [],
        "best_known": [],
        "gap_pct": [],
    }
    
    if config.verbose:
        print(f"\n{'='*60}")
        print(f"FACO Test Configuration:")
        print(f"  Nodes (n): {results['n']}")
        print(f"  Ants: {config.n_ants}")
        print(f"  Iterations: {config.iterations}")
        print(f"  Instances: {results['n_instances']}")
        print(f"  Candidate list size: {config.cand_list_size}")
        print(f"  Local search: {config.use_local_search}")
        print(f"  Random mode: {config.random_mode}")
        print(f"  Device: {config.device}")
        print(f"{'='*60}\n")

    for inst_idx, inst in enumerate(iterable):
        coords = inst["coords"]
        distances = inst["distances"]
        inst_name = str(inst.get("name", f"inst_{inst_idx+1}"))
        best_k = inst.get("best_known")
        
        # Initialize FACO solver
        faco = MFACO_TSP(
            distances=distances,
            n_ants=config.n_ants,
            cand_list_size=config.cand_list_size,
            backup_list_size=config.backup_list_size,
            min_new_edges=config.min_new_edges,
            decay=config.decay,
            use_local_search=config.use_local_search,
            random_mode=config.random_mode,
            device=config.device,
        )
        
        init_cost = faco.best_cost
        iter_times = []
        
        # Run FACO iterations
        start_time = time.time()
        
        for it in range(config.iterations):
            iter_start = time.time()
            
            # Sample solutions
            costs, flats, touched_list, _, _ = faco.sample(invtemp=1.0, require_prob=False)
            
            # Find best solution in this iteration
            best_idx = int(np.argmin(costs))
            best_cost = float(costs[best_idx])
            best_flat = flats[best_idx]
            
            # Update pheromone with best solution
            faco._update_pheromone_from_flat(best_flat, best_cost)
            
            iter_time = time.time() - iter_start
            iter_times.append(iter_time)
            
            if config.verbose and (it + 1) % max(1, config.iterations // 5) == 0:
                print(f"  Instance {inst_idx+1}/{len(iterable)}, Iter {it+1}/{config.iterations}: "
                      f"Best={faco.best_cost:.4f}, Iter Best={best_cost:.4f}")
        
        total_time = time.time() - start_time
        final_cost = faco.best_cost
        improvement = (init_cost - final_cost) / init_cost * 100
        
        results["costs"].append(final_cost)
        results["init_costs"].append(init_cost)
        results["improvements"].append(improvement)
        results["times"].append(total_time)
        results["iter_times"].append(iter_times)

        results["instance_names"].append(inst_name)
        results["best_known"].append(best_k)
        if best_k is not None and best_k > 0:
            results["gap_pct"].append((float(final_cost) / float(best_k) - 1.0) * 100.0)
        else:
            results["gap_pct"].append(None)
        
        if config.verbose:
            msg = (
                f"  {inst_name} ({inst_idx+1}/{len(iterable)}): "
                f"Init={init_cost:.4f} -> Final={final_cost:.4f} "
                f"(Improvement: {improvement:.2f}%, Time: {total_time:.2f}s)"
            )
            if best_k is not None and best_k > 0:
                gap = (float(final_cost) / float(best_k) - 1.0) * 100.0
                msg += f" | BestKnown={best_k:.0f} (gap {gap:+.2f}%)"
            print(msg + "\n")
    
    # Compute statistics
    results["mean_cost"] = float(np.mean(results["costs"]))
    results["std_cost"] = float(np.std(results["costs"]))
    results["mean_init_cost"] = float(np.mean(results["init_costs"]))
    results["mean_improvement"] = float(np.mean(results["improvements"]))
    results["mean_time"] = float(np.mean(results["times"]))
    results["mean_iter_time"] = float(np.mean([np.mean(it) for it in results["iter_times"]]))
    
    return results


def print_summary(results: dict) -> None:
    """Print test results summary."""
    print(f"\n{'='*60}")
    print("FACO Test Results Summary")
    print(f"{'='*60}")
    print(f"  Configuration:")
    print(f"    Nodes: {results['n']}")
    print(f"    Ants: {results['n_ants']}")
    print(f"    Iterations: {results['n_iterations']}")
    print(f"    Instances: {results['n_instances']}")
    print(f"    Local Search: {results['use_local_search']}")
    print(f"    Random Mode: {results['random_mode']}")
    print()
    print(f"  Performance:")
    print(f"    Mean Initial Cost: {results['mean_init_cost']:.4f}")
    print(f"    Mean Final Cost:   {results['mean_cost']:.4f} ± {results['std_cost']:.4f}")
    print(f"    Mean Improvement:  {results['mean_improvement']:.2f}%")
    gaps = [g for g in results.get("gap_pct", []) if g is not None]
    if gaps:
        print(f"    Mean Gap to BestKnown: {float(np.mean(gaps)):+.2f}%")
    print()
    print(f"  Timing:")
    print(f"    Mean Total Time:   {results['mean_time']:.3f}s")
    print(f"    Mean Iter Time:    {results['mean_iter_time']*1000:.2f}ms")
    print(f"{'='*60}\n")


def run_scaling_test(
    n_values: List[int],
    n_ants: int = 20,
    n_iterations: int = 50,
    n_instances: int = 5,
    device: str = "cpu",
    seed: int = 42,
) -> List[dict]:
    """
    Run FACO scaling test across different problem sizes.
    
    Args:
        n_values: List of problem sizes to test
        n_ants: Number of ants
        n_iterations: Number of iterations
        n_instances: Number of instances per size
        device: Device to use
        seed: Random seed
        
    Returns:
        List of results for each problem size
    """
    print("\n" + "="*70)
    print("FACO Scaling Test")
    print("="*70)
    
    all_results = []
    
    for n in n_values:
        print(f"\nTesting n={n}...")
        results = run_faco_test(
            n=n,
            n_ants=n_ants,
            n_iterations=n_iterations,
            n_instances=n_instances,
            device=device,
            seed=seed,
            verbose=False,
        )
        all_results.append(results)
        print(f"  n={n}: Cost={results['mean_cost']:.4f}±{results['std_cost']:.4f}, "
              f"Improve={results['mean_improvement']:.2f}%, "
              f"Time={results['mean_time']:.3f}s")
    
    # Print scaling summary table
    print("\n" + "-"*70)
    print(f"{'n':>8} | {'Init Cost':>12} | {'Final Cost':>14} | {'Improve%':>10} | {'Time(s)':>10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['n']:>8} | {r['mean_init_cost']:>12.4f} | "
              f"{r['mean_cost']:>7.4f}±{r['std_cost']:.4f} | "
              f"{r['mean_improvement']:>9.2f}% | {r['mean_time']:>10.3f}")
    print("-"*70 + "\n")
    
    return all_results


def clone_faco_state(faco: MFACO_TSP) -> dict:
    """
    Clone the essential state of a FACO solver for later restoration.
    """
    if hasattr(faco, "save_snapshot"):
        return faco.save_snapshot()

    state = {
        "source_route": faco.source_route.copy(),
        "source_cost": faco.source_cost,
        "best_route": faco.best_route.copy(),
        "best_cost": faco.best_cost,
        "pheromone_sparse": faco.pheromone_sparse.clone(),
        "tau_min": faco.tau_min,
        "tau_max": faco.tau_max,
    }
    if hasattr(faco, "ant_routes"):
        state["ant_routes"] = [r.copy() for r in faco.ant_routes]
    if hasattr(faco, "ant_costs"):
        state["ant_costs"] = faco.ant_costs.copy()
    return state


def restore_faco_state(faco: MFACO_TSP, state: dict) -> None:
    """
    Restore a FACO solver to a previously cloned state.
    """
    if hasattr(faco, "load_snapshot") and (
        "pheromone_sparse_np" in state
        and "source_route" in state
        and "best_route" in state
    ):
        faco.load_snapshot(state, validate=False)
        return

    faco.source_route = state["source_route"].copy()
    faco.source_cost = state["source_cost"]
    faco.best_route = state["best_route"].copy()
    faco.best_cost = state["best_cost"]
    faco.pheromone_sparse = state["pheromone_sparse"].clone()
    faco.tau_min = state["tau_min"]
    faco.tau_max = state["tau_max"]
    if "ant_routes" in state:
        faco.ant_routes = [r.copy() for r in state["ant_routes"]]
    if "ant_costs" in state:
        faco.ant_costs = state["ant_costs"].copy()


def run_late_iteration_comparison(
    config: TestConfig,
) -> dict:
    """
    Run late-iteration comparison: warm up FACO to stagnation, then compare
    roulette vs random continuation from the same snapshot.
    
    Args:
        n: Number of nodes per instance
        n_ants: Number of ants
        warmup_iters: Iterations to run before snapshot (to reach stagnation)
        continuation_iters: Iterations to run after snapshot for comparison
        n_instances: Number of random instances to test
        cand_list_size: Candidate list size
        backup_list_size: Backup list size
        min_new_edges: Minimum new edges per ant
        decay: Pheromone decay rate
        use_local_search: Whether to use local 2-opt search
        device: Device to use
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with comparison results
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    iterable = _get_iterable_instances(config)
    inferred_n, inferred_count = _infer_n_and_count(iterable, config.n, config.n_instances)
    
    results = {
        "n": inferred_n,
        "n_ants": config.n_ants,
        "warmup_iters": config.warmup_iters,
        "continuation_iters": config.continuation_iters,
        "n_instances": inferred_count,
        "cand_list_size": config.cand_list_size,
        "use_local_search": config.use_local_search,
        # Per-instance results
        "snapshot_costs": [],          # Cost at snapshot (after warmup)
        "roulette_final_costs": [],    # Final cost with roulette continuation
        "random_final_costs": [],      # Final cost with random continuation
        "roulette_improvements": [],   # % improvement from snapshot
        "random_improvements": [],     # % improvement from snapshot
        # Comparison metrics
        "random_wins": [],             # 1 if random < roulette, else 0
        "random_margin": [],           # (roulette - random) / snapshot * 100
        "instance_names": [],
    }
    
    if config.verbose:
        print(f"\n{'='*70}")
        print(f"Late-Iteration Comparison (Stagnation Analysis)")
        print(f"{'='*70}")
        print(f"  Nodes (n): {results['n']}")
        print(f"  Ants: {config.n_ants}")
        print(f"  Warmup iterations: {config.warmup_iters}")
        print(f"  Continuation iterations: {config.continuation_iters}")
        print(f"  Instances: {results['n_instances']}")
        print(f"  Candidate list size: {config.cand_list_size}")
        print(f"  Local search: {config.use_local_search}")
        print(f"  Device: {config.device}")
        print(f"{'='*70}\n")

    for inst_idx, inst in enumerate(iterable):
        coords = inst["coords"]
        distances = inst["distances"]
        inst_name = str(inst.get("name", f"inst_{inst_idx+1}"))
        
        # Initialize FACO solver (roulette mode for warmup)
        faco = MFACO_TSP(
            distances=distances,
            n_ants=config.n_ants,
            cand_list_size=config.cand_list_size,
            backup_list_size=config.backup_list_size,
            min_new_edges=config.min_new_edges,
            decay=config.decay,
            use_local_search=config.use_local_search,
            random_mode=False,  # Roulette for warmup
            device=config.device,
        )
        
        init_cost = faco.best_cost
        
        # Warmup phase: run roulette to reach stagnation
        for it in range(config.warmup_iters):
            costs, flats, _, _, _ = faco.sample(invtemp=1.0, require_prob=False)
            best_idx = int(np.argmin(costs))
            faco._update_pheromone_from_flat(flats[best_idx], float(costs[best_idx]))
        
        snapshot_cost = faco.best_cost
        
        # Clone state for fair comparison
        state_snapshot = clone_faco_state(faco)
        
        # --- Roulette continuation ---
        for it in range(config.continuation_iters):
            costs, flats, _, _, _ = faco.sample(invtemp=1.0, require_prob=False)
            best_idx = int(np.argmin(costs))
            faco._update_pheromone_from_flat(flats[best_idx], float(costs[best_idx]))
        roulette_final = faco.best_cost
        
        # --- Random continuation (from same snapshot) ---
        faco_rand = MFACO_TSP(
            distances=distances,
            n_ants=config.n_ants,
            cand_list_size=config.cand_list_size,
            backup_list_size=config.backup_list_size,
            min_new_edges=config.min_new_edges,
            decay=config.decay,
            use_local_search=config.use_local_search,
            random_mode=True,
            device=config.device,
        )
        restore_faco_state(faco_rand, state_snapshot)
        for it in range(config.continuation_iters):
            costs, flats, _, _, _ = faco_rand.sample(invtemp=1.0, require_prob=False)
            best_idx = int(np.argmin(costs))
            faco_rand._update_pheromone_from_flat(flats[best_idx], float(costs[best_idx]))
        random_final = faco_rand.best_cost
        
        # Compute metrics
        roulette_improve = (snapshot_cost - roulette_final) / snapshot_cost * 100
        random_improve = (snapshot_cost - random_final) / snapshot_cost * 100
        random_wins = 1 if random_final < roulette_final else 0
        random_margin = (roulette_final - random_final) / snapshot_cost * 100
        
        results["snapshot_costs"].append(snapshot_cost)
        results["roulette_final_costs"].append(roulette_final)
        results["random_final_costs"].append(random_final)
        results["roulette_improvements"].append(roulette_improve)
        results["random_improvements"].append(random_improve)
        results["random_wins"].append(random_wins)
        results["random_margin"].append(random_margin)
        results["instance_names"].append(inst_name)
        
        if config.verbose:
            winner = "RANDOM" if random_wins else "ROULETTE"
            print(f"  {inst_name} ({inst_idx+1}/{len(iterable)}): "
                  f"Snapshot={snapshot_cost:.4f} | "
                  f"Roulette={roulette_final:.4f} ({roulette_improve:+.2f}%) | "
                  f"Random={random_final:.4f} ({random_improve:+.2f}%) | "
                  f"Winner: {winner}")
    
    # Compute summary statistics
    results["mean_snapshot_cost"] = float(np.mean(results["snapshot_costs"]))
    results["mean_roulette_final"] = float(np.mean(results["roulette_final_costs"]))
    results["mean_random_final"] = float(np.mean(results["random_final_costs"]))
    results["mean_roulette_improve"] = float(np.mean(results["roulette_improvements"]))
    results["mean_random_improve"] = float(np.mean(results["random_improvements"]))
    
    # Win-rate: P(random < roulette)
    results["win_rate"] = float(np.mean(results["random_wins"]))
    
    # Tail improvement metrics
    margins = np.array(results["random_margin"])
    results["tail_05pct"] = float(np.mean(margins > 0.5))  # Fraction with >0.5% improvement
    results["tail_1pct"] = float(np.mean(margins > 1.0))   # Fraction with >1% improvement
    results["tail_2pct"] = float(np.mean(margins > 2.0))   # Fraction with >2% improvement
    results["mean_margin"] = float(np.mean(margins))
    results["std_margin"] = float(np.std(margins))
    
    return results


def print_late_comparison_summary(results: dict) -> None:
    """Print late-iteration comparison results summary."""
    print(f"\n{'='*70}")
    print("Late-Iteration Comparison Results")
    print(f"{'='*70}")
    print(f"  Configuration:")
    print(f"    Nodes: {results['n']}")
    print(f"    Ants: {results['n_ants']}")
    print(f"    Warmup iterations: {results['warmup_iters']}")
    print(f"    Continuation iterations: {results['continuation_iters']}")
    print(f"    Instances: {results['n_instances']}")
    print(f"    Local Search: {results['use_local_search']}")
    print()
    print(f"  Costs:")
    print(f"    Mean Snapshot Cost:       {results['mean_snapshot_cost']:.4f}")
    print(f"    Mean Roulette Final:      {results['mean_roulette_final']:.4f} "
          f"({results['mean_roulette_improve']:+.3f}%)")
    print(f"    Mean Random Final:        {results['mean_random_final']:.4f} "
          f"({results['mean_random_improve']:+.3f}%)")
    print()
    print(f"  Key Metrics:")
    print(f"    Win-rate (P(random < roulette)): {results['win_rate']*100:.1f}%")
    print(f"    Mean margin (random vs roulette): {results['mean_margin']:+.3f}% ± {results['std_margin']:.3f}%")
    print()
    print(f"  Tail Improvement (fraction of runs where random beats roulette by):")
    print(f"    > 0.5%: {results['tail_05pct']*100:.1f}%")
    print(f"    > 1.0%: {results['tail_1pct']*100:.1f}%")
    print(f"    > 2.0%: {results['tail_2pct']*100:.1f}%")
    print()
    
    # Interpretation
    print(f"  Interpretation:")
    if results['win_rate'] > 0.3:
        print(f"    ✓ Random sometimes wins ({results['win_rate']*100:.0f}% win-rate)")
        print(f"      → BvB continuation reward should have positive advantages to learn from")
    else:
        print(f"    ✗ Random rarely wins ({results['win_rate']*100:.0f}% win-rate)")
        print(f"      → May need larger batches / longer horizons / stronger perturbations")
    
    if results['tail_05pct'] > 0.1:
        print(f"    ✓ Meaningful wins exist ({results['tail_05pct']*100:.0f}% with >0.5% margin)")
    else:
        print(f"    ✗ Few meaningful wins ({results['tail_05pct']*100:.0f}% with >0.5% margin)")
    
    print(f"{'='*70}\n")


# =============================================================================
# Neural-Enhanced Comparison
# =============================================================================

def build_pyg_from_faco(faco: MFACO_TSP, coords: torch.Tensor, device: str = "cpu") -> PyGData:
    """Build PyG Data object from FACO's nn_list."""
    n = coords.shape[0]
    k = faco.k
    
    x = coords.float()
    
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


def compute_residual_logits(net, faco: MFACO_TSP, device: str, gamma: float = 0.2) -> np.ndarray:
    """Compute (n, k) residual logits from neural network."""
    n = faco.n
    k = faco.k
    
    residual = np.zeros((n, k), dtype=np.float32)

    # Detect whether the network expects extra context features.
    expected_extra = 0
    try:
        in_features = int(net.scorer.scorer[0].in_features)
        expected_extra = max(0, in_features - (2 * int(net.units) + 1))
    except Exception:
        expected_extra = 0

    tau_row_t = None
    logtau_z = None
    tau_rank01 = None
    if expected_extra > 0:
        tau_row_t = faco.pheromone_sparse.detach().to(device).clamp_min(1e-12)  # (n,k)
        logtau = torch.log(tau_row_t)
        mu = logtau.mean(dim=1, keepdim=True)
        std = logtau.std(dim=1, keepdim=True).clamp_min(1e-6)
        logtau_z = (logtau - mu) / std  # (n,k)

        # Rank feature in [0,1] within each row.
        ranks = torch.argsort(torch.argsort(tau_row_t, dim=1), dim=1).float()
        tau_rank01 = ranks / (k - 1 + 1e-8)  # (n,k)
    
    for u in range(n):
        nn = faco.nn_list[u]
        valid_mask = nn >= 0
        valid_idx = np.nonzero(valid_mask)[0]
        
        if len(valid_idx) == 0:
            continue
        
        cand_nodes = nn[valid_idx]
        cand_nodes_t = torch.as_tensor(cand_nodes, device=device, dtype=torch.long)
        cand_dist = torch.from_numpy(faco.dist_np32[u, cand_nodes]).float().to(device)

        extra_feats_t = None
        if expected_extra > 0:
            # Provide the feature layout used during training when available:
            # [logtau_z, tau_rank01, stagn0, prog0] then pad/truncate to expected_extra.
            stagn0 = torch.zeros((len(valid_idx), 1), device=device, dtype=torch.float32)
            prog0 = torch.zeros((len(valid_idx), 1), device=device, dtype=torch.float32)
            base = torch.cat(
                [
                    logtau_z[u, valid_idx].unsqueeze(-1),
                    tau_rank01[u, valid_idx].unsqueeze(-1),
                    stagn0,
                    prog0,
                ],
                dim=-1,
            )  # (m,4)

            if expected_extra <= base.size(1):
                extra_feats_t = base[:, :expected_extra]
            else:
                pad = torch.zeros((base.size(0), expected_extra - base.size(1)), device=device, dtype=base.dtype)
                extra_feats_t = torch.cat([base, pad], dim=-1)
        
        with torch.no_grad():
            scores = net.score_candidates(u, cand_nodes_t, cand_dist, extra_feats=extra_feats_t)
        
        residual[u, valid_idx] = gamma * scores.cpu().numpy()
    
    return residual


def run_neural_comparison(
    config: TestConfig,
    checkpoint_path: str,
    warmup_iters: int,
) -> dict:
    """
    Compare baseline FACO vs neural-enhanced FACO.
    
    By default runs end-to-end comparison (warmup_iters=0).
    Set warmup_iters > 0 for late-iteration/stagnation analysis.
    
    Args:
        n: Number of nodes per instance
        n_ants: Number of ants
        n_iterations: Total iterations to run for comparison
        n_instances: Number of random instances to test
        checkpoint_path: Path to trained model checkpoint
        warmup_iters: Iterations to run before snapshot (0 = end-to-end comparison)
        gamma: Residual logit scale factor
        cand_list_size: Candidate list size
        backup_list_size: Backup list size
        min_new_edges: Minimum new edges per ant
        decay: Pheromone decay rate
        use_local_search: Whether to use local 2-opt search
        device: Device to use
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with comparison results
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    net, ckpt, resolved_checkpoint_path = _load_net_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=config.device,
        verbose=config.verbose,
    )
    
    # Compute continuation iterations (total - warmup)
    continuation_iters = config.iterations - warmup_iters
    if continuation_iters <= 0:
        raise ValueError(
            f"iterations ({config.iterations}) must be > warmup_iters ({warmup_iters})"
        )
    
    mode_str = "End-to-End" if warmup_iters == 0 else f"Late-Iteration (warmup={warmup_iters})"
    
    results = {
        "n": config.n,
        "n_ants": config.n_ants,
        "n_iterations": config.iterations,
        "warmup_iters": warmup_iters,
        "continuation_iters": continuation_iters,
        "n_instances": config.n_instances,
        "gamma": config.gamma,
        "cand_list_size": config.cand_list_size,
        "use_local_search": config.use_local_search,
        "checkpoint": resolved_checkpoint_path,
        "mode": mode_str,
        # Per-instance results
        "init_costs": [],              # Initial cost (before any iterations)
        "snapshot_costs": [],          # Cost at snapshot (after warmup, if any)
        "baseline_final_costs": [],
        "neural_final_costs": [],
        "baseline_improvements": [],   # % improvement from init/snapshot
        "neural_improvements": [],
        # Comparison metrics
        "neural_wins": [],
        "neural_margin": [],
        # Optional TSPLIB context
        "instance_names": [],
        "best_known": [],
        "baseline_gap_pct": [],
        "neural_gap_pct": [],
    }

    iterable = _get_iterable_instances(config)
    inferred_n, inferred_count = _infer_n_and_count(iterable, config.n, config.n_instances)
    results["n"] = inferred_n
    results["n_instances"] = inferred_count
    
    if config.verbose:
        print(f"\n{'='*70}")
        print(f"Neural-Enhanced FACO Comparison ({mode_str})")
        print(f"{'='*70}")
        print(f"  Nodes (n): {results['n']}")
        print(f"  Ants: {config.n_ants}")
        print(f"  Total iterations: {config.iterations}")
        if warmup_iters > 0:
            print(f"  Warmup iterations: {warmup_iters}")
            print(f"  Continuation iterations: {continuation_iters}")
        print(f"  Instances: {results['n_instances']}")
        print(f"  Gamma (residual scale): {config.gamma}")
        print(f"  Candidate list size: {config.cand_list_size}")
        print(f"  Local search: {config.use_local_search}")
        print(f"  Device: {config.device}")
        print(f"{'='*70}\n")

    for inst_idx, inst in enumerate(iterable):
        coords = inst["coords"]
        distances = inst["distances"]
        inst_name = str(inst.get("name", f"inst_{inst_idx+1}"))
        best_k = inst.get("best_known")
        
        # Initialize FACO solver
        faco = MFACO_TSP(
            distances=distances,
            n_ants=config.n_ants,
            cand_list_size=config.cand_list_size,
            backup_list_size=config.backup_list_size,
            min_new_edges=config.min_new_edges,
            decay=config.decay,
            use_local_search=config.use_local_search,
            random_mode=False,
            device=config.device,
        )
        
        init_cost = faco.best_cost
        
        # Warmup phase (if any)
        if warmup_iters > 0:
            for it in range(warmup_iters):
                costs, flats, _, _, _ = faco.sample(invtemp=1.0, require_prob=False)
                best_idx = int(np.argmin(costs))
                faco._update_pheromone_from_flat(flats[best_idx], float(costs[best_idx]))
        
        snapshot_cost = faco.best_cost
        state_snapshot = clone_faco_state(faco)
        
        # --- Baseline continuation (no neural guidance) ---
        for it in range(continuation_iters):
            costs, flats, _, _, _ = faco.sample(invtemp=1.0, require_prob=False)
            best_idx = int(np.argmin(costs))
            faco._update_pheromone_from_flat(flats[best_idx], float(costs[best_idx]))
        baseline_final = faco.best_cost
        
        # --- Neural-enhanced continuation (from same snapshot) ---
        restore_faco_state(faco, state_snapshot)
        
        # Encode graph once
        pyg = build_pyg_from_faco(faco, coords, config.device)
        net.encode(pyg)
        
        for it in range(continuation_iters):
            # Compute neural residual logits
            residual_logits = compute_residual_logits(net, faco, config.device, config.gamma)
            
            # Sample with neural guidance
            costs, flats, _, _, _ = faco.sample(invtemp=1.0, require_prob=False, 
                                                 residual_logits=residual_logits)
            best_idx = int(np.argmin(costs))
            faco._update_pheromone_from_flat(flats[best_idx], float(costs[best_idx]))
        neural_final = faco.best_cost
        
        # Compute metrics (improvements relative to init_cost for end-to-end comparison)
        ref_cost = init_cost if warmup_iters == 0 else snapshot_cost
        baseline_improve = (ref_cost - baseline_final) / ref_cost * 100
        neural_improve = (ref_cost - neural_final) / ref_cost * 100
        neural_wins = 1 if neural_final < baseline_final else 0
        neural_margin = (baseline_final - neural_final) / ref_cost * 100
        
        results["init_costs"].append(init_cost)
        results["snapshot_costs"].append(snapshot_cost)
        results["baseline_final_costs"].append(baseline_final)
        results["neural_final_costs"].append(neural_final)
        results["baseline_improvements"].append(baseline_improve)
        results["neural_improvements"].append(neural_improve)
        results["neural_wins"].append(neural_wins)
        results["neural_margin"].append(neural_margin)

        results["instance_names"].append(inst_name)
        results["best_known"].append(best_k)
        if best_k is not None and best_k > 0:
            results["baseline_gap_pct"].append((float(baseline_final) / float(best_k) - 1.0) * 100.0)
            results["neural_gap_pct"].append((float(neural_final) / float(best_k) - 1.0) * 100.0)
        else:
            results["baseline_gap_pct"].append(None)
            results["neural_gap_pct"].append(None)
        
        if config.verbose:
            winner = "NEURAL" if neural_wins else "BASELINE"
            margin_str = f"{neural_margin:+.3f}%" if neural_wins else f"{-neural_margin:+.3f}%"
            name_prefix = f"{inst_name} "
            if warmup_iters == 0:
                msg = (
                    f"  {name_prefix}({inst_idx+1}/{len(iterable)}): "
                    f"Init={init_cost:.4f} | "
                    f"Baseline={baseline_final:.4f} ({baseline_improve:+.2f}%) | "
                    f"Neural={neural_final:.4f} ({neural_improve:+.2f}%) | "
                    f"Winner: {winner} ({margin_str})"
                )
            else:
                msg = (
                    f"  {name_prefix}({inst_idx+1}/{len(iterable)}): "
                    f"Snapshot={snapshot_cost:.4f} | "
                    f"Baseline={baseline_final:.4f} ({baseline_improve:+.2f}%) | "
                    f"Neural={neural_final:.4f} ({neural_improve:+.2f}%) | "
                    f"Winner: {winner} ({margin_str})"
                )
            if best_k is not None and best_k > 0:
                base_gap = (float(baseline_final) / float(best_k) - 1.0) * 100.0
                neur_gap = (float(neural_final) / float(best_k) - 1.0) * 100.0
                msg += f" | BestKnown={best_k:.0f} (gap: baseline {base_gap:+.2f}%, neural {neur_gap:+.2f}%)"
            print(msg)
    
    # Compute summary statistics
    results["mean_init_cost"] = float(np.mean(results["init_costs"]))
    results["mean_snapshot_cost"] = float(np.mean(results["snapshot_costs"]))
    results["mean_baseline_final"] = float(np.mean(results["baseline_final_costs"]))
    results["mean_neural_final"] = float(np.mean(results["neural_final_costs"]))
    results["mean_baseline_improve"] = float(np.mean(results["baseline_improvements"]))
    results["mean_neural_improve"] = float(np.mean(results["neural_improvements"]))
    results["win_rate"] = float(np.mean(results["neural_wins"]))
    
    margins = np.array(results["neural_margin"])
    results["mean_margin"] = float(np.mean(margins))
    results["std_margin"] = float(np.std(margins))
    results["tail_05pct"] = float(np.mean(margins > 0.5))
    results["tail_1pct"] = float(np.mean(margins > 1.0))
    results["tail_2pct"] = float(np.mean(margins > 2.0))
    
    return results


def run_multi_model_comparison(config: TestConfig, checkpoint_paths: List[str], warmup_iters: int) -> dict:
    """Compare multiple checkpoints against the same baseline on the same instances.

    Baseline is standard FACO continuation (no residual logits).
    Each model runs neural-guided continuation from the same snapshot.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must be non-empty")

    nets: list[tuple[str, object, dict]] = []
    for cp in checkpoint_paths:
        net, ckpt, resolved = _load_net_from_checkpoint(cp, device=config.device, verbose=config.verbose)
        nets.append((resolved, net, ckpt))

    continuation_iters = config.iterations - warmup_iters
    if continuation_iters <= 0:
        raise ValueError(
            f"iterations ({config.iterations}) must be > warmup_iters ({warmup_iters})"
        )

    iterable = _get_iterable_instances(config)
    inferred_n, inferred_count = _infer_n_and_count(iterable, config.n, config.n_instances)

    mode_str = "End-to-End" if warmup_iters == 0 else f"Late-Iteration (warmup={warmup_iters})"

    results: dict[str, Any] = {
        "n": inferred_n,
        "n_ants": config.n_ants,
        "n_iterations": config.iterations,
        "warmup_iters": warmup_iters,
        "continuation_iters": continuation_iters,
        "n_instances": inferred_count,
        "gamma": config.gamma,
        "cand_list_size": config.cand_list_size,
        "use_local_search": config.use_local_search,
        "mode": mode_str,
        "checkpoints": [r for r, _, _ in nets],
        "baseline_final_costs": [],
        "instance_names": [],
        "best_known": [],
        "baseline_gap_pct": [],
        # per model
        "models": {},
    }

    for resolved, _, _ in nets:
        results["models"][resolved] = {
            "final_costs": [],
            "improvements": [],
            "wins_vs_baseline": [],
            "margins_vs_baseline": [],
            "gap_pct": [],
        }

    if config.verbose:
        print(f"\n{'='*70}")
        print(f"Multi-Model Neural FACO Comparison ({mode_str})")
        print(f"{'='*70}")
        print(f"  Nodes (n): {results['n']}")
        print(f"  Ants: {config.n_ants}")
        print(f"  Total iterations: {config.iterations}")
        if warmup_iters > 0:
            print(f"  Warmup iterations: {warmup_iters}")
            print(f"  Continuation iterations: {continuation_iters}")
        print(f"  Instances: {results['n_instances']}")
        print(f"  Gamma (residual scale): {config.gamma}")
        print(f"  Models: {len(nets)}")
        for resolved, _, ckpt in nets:
            print(f"    - {resolved} (epoch {ckpt.get('epoch','?')}, improve {ckpt.get('improve','?')})")
        print(f"{'='*70}\n")

    for inst_idx, inst in enumerate(iterable):
        coords = inst["coords"]
        distances = inst["distances"]
        inst_name = str(inst.get("name", f"inst_{inst_idx+1}"))
        best_k = inst.get("best_known")

        faco = MFACO_TSP(
            distances=distances,
            n_ants=config.n_ants,
            cand_list_size=config.cand_list_size,
            backup_list_size=config.backup_list_size,
            min_new_edges=config.min_new_edges,
            decay=config.decay,
            use_local_search=config.use_local_search,
            random_mode=False,
            device=config.device,
        )
        init_cost = faco.best_cost

        if warmup_iters > 0:
            for _ in range(warmup_iters):
                costs, flats, _, _, _ = faco.sample(invtemp=1.0, require_prob=False)
                best_idx = int(np.argmin(costs))
                faco._update_pheromone_from_flat(flats[best_idx], float(costs[best_idx]))

        snapshot_cost = faco.best_cost
        state_snapshot = clone_faco_state(faco)

        # Baseline continuation
        for _ in range(continuation_iters):
            costs, flats, _, _, _ = faco.sample(invtemp=1.0, require_prob=False)
            best_idx = int(np.argmin(costs))
            faco._update_pheromone_from_flat(flats[best_idx], float(costs[best_idx]))
        baseline_final = faco.best_cost

        results["baseline_final_costs"].append(baseline_final)
        results["instance_names"].append(inst_name)
        results["best_known"].append(best_k)
        if best_k is not None and best_k > 0:
            results["baseline_gap_pct"].append((float(baseline_final) / float(best_k) - 1.0) * 100.0)
        else:
            results["baseline_gap_pct"].append(None)

        # Run each model from the same snapshot
        # Build pyg once (needs current faco state restored).
        ref_cost = init_cost if warmup_iters == 0 else snapshot_cost

        best_model = None
        best_model_cost = None

        for resolved, net, _ in nets:
            restore_faco_state(faco, state_snapshot)
            pyg = build_pyg_from_faco(faco, coords, config.device)
            net.encode(pyg)

            for _ in range(continuation_iters):
                residual_logits = compute_residual_logits(net, faco, config.device, config.gamma)
                costs, flats, _, _, _ = faco.sample(
                    invtemp=1.0,
                    require_prob=False,
                    residual_logits=residual_logits,
                )
                best_idx = int(np.argmin(costs))
                faco._update_pheromone_from_flat(flats[best_idx], float(costs[best_idx]))

            final_cost = faco.best_cost
            improve = (ref_cost - final_cost) / ref_cost * 100.0
            win = 1 if final_cost < baseline_final else 0
            margin = (baseline_final - final_cost) / ref_cost * 100.0

            m = results["models"][resolved]
            m["final_costs"].append(final_cost)
            m["improvements"].append(improve)
            m["wins_vs_baseline"].append(win)
            m["margins_vs_baseline"].append(margin)
            if best_k is not None and best_k > 0:
                m["gap_pct"].append((float(final_cost) / float(best_k) - 1.0) * 100.0)
            else:
                m["gap_pct"].append(None)

            if best_model_cost is None or final_cost < best_model_cost:
                best_model_cost = final_cost
                best_model = resolved

        if config.verbose:
            line = f"  {inst_name} ({inst_idx+1}/{len(iterable)}): Baseline={baseline_final:.4f}"
            if best_model is not None and best_model_cost is not None:
                line += f" | BestModel={Path(best_model).name} ({best_model_cost:.4f})"
            if best_k is not None and best_k > 0:
                base_gap = (float(baseline_final) / float(best_k) - 1.0) * 100.0
                line += f" | BestKnown={best_k:.0f} (baseline gap {base_gap:+.2f}%)"
            print(line)

    # Aggregate
    results["mean_baseline_final"] = float(np.mean(results["baseline_final_costs"]))
    base_gaps = [g for g in results.get("baseline_gap_pct", []) if g is not None]
    results["mean_baseline_gap_pct"] = float(np.mean(base_gaps)) if base_gaps else None

    for resolved in list(results["models"].keys()):
        m = results["models"][resolved]
        m["mean_final"] = float(np.mean(m["final_costs"]))
        m["mean_improve"] = float(np.mean(m["improvements"]))
        m["win_rate_vs_baseline"] = float(np.mean(m["wins_vs_baseline"]))
        margins = np.asarray(m["margins_vs_baseline"], dtype=np.float64)
        m["mean_margin_vs_baseline"] = float(np.mean(margins))
        m["std_margin_vs_baseline"] = float(np.std(margins))
        gaps = [g for g in m.get("gap_pct", []) if g is not None]
        m["mean_gap_pct"] = float(np.mean(gaps)) if gaps else None

    return results


def print_multi_model_comparison_summary(results: dict) -> None:
    print(f"\n{'='*70}")
    print(f"Multi-Model Neural FACO Comparison Results ({results.get('mode', 'Unknown')})")
    print(f"{'='*70}")
    print(f"  Configuration:")
    print(f"    Nodes: {results['n']}")
    print(f"    Ants: {results['n_ants']}")
    print(f"    Total iterations: {results['n_iterations']}")
    if results.get("warmup_iters", 0) > 0:
        print(f"    Warmup iterations: {results['warmup_iters']}")
        print(f"    Continuation iterations: {results['continuation_iters']}")
    print(f"    Instances: {results['n_instances']}")
    print(f"    Gamma: {results['gamma']}")
    print()
    print(f"  Baseline:")
    print(f"    Mean Final Cost: {results['mean_baseline_final']:.4f}")
    if results.get("mean_baseline_gap_pct") is not None:
        print(f"    Mean Gap to BestKnown: {results['mean_baseline_gap_pct']:+.2f}%")
    print()
    print(f"  Models:")
    # Sort by mean_final
    items = sorted(results["models"].items(), key=lambda kv: kv[1].get("mean_final", float("inf")))
    for resolved, m in items:
        name = Path(resolved).name
        print(f"    {name}:")
        print(f"      Mean Final Cost: {m['mean_final']:.4f}")
        print(f"      Win-rate vs baseline: {m['win_rate_vs_baseline']*100:.1f}%")
        print(
            f"      Mean margin vs baseline: {m['mean_margin_vs_baseline']:+.3f}% ± {m['std_margin_vs_baseline']:.3f}%"
        )
        if m.get("mean_gap_pct") is not None:
            print(f"      Mean Gap to BestKnown: {m['mean_gap_pct']:+.2f}%")
    print(f"{'='*70}\n")


def print_neural_comparison_summary(results: dict) -> None:
    """Print neural comparison results summary."""
    print(f"\n{'='*70}")
    print(f"Neural-Enhanced FACO Comparison Results ({results.get('mode', 'Unknown')})")
    print(f"{'='*70}")
    print(f"  Configuration:")
    print(f"    Checkpoint: {results['checkpoint']}")
    print(f"    Nodes: {results['n']}")
    print(f"    Ants: {results['n_ants']}")
    print(f"    Total iterations: {results['n_iterations']}")
    if results['warmup_iters'] > 0:
        print(f"    Warmup iterations: {results['warmup_iters']}")
        print(f"    Continuation iterations: {results['continuation_iters']}")
    print(f"    Instances: {results['n_instances']}")
    print(f"    Gamma: {results['gamma']}")
    print(f"    Local Search: {results['use_local_search']}")
    print()
    print(f"  Costs:")
    if results['warmup_iters'] == 0:
        print(f"    Mean Initial Cost:     {results['mean_init_cost']:.4f}")
    else:
        print(f"    Mean Snapshot Cost:    {results['mean_snapshot_cost']:.4f}")
    print(f"    Mean Baseline Final:   {results['mean_baseline_final']:.4f} "
          f"({results['mean_baseline_improve']:+.3f}%)")
    print(f"    Mean Neural Final:     {results['mean_neural_final']:.4f} "
          f"({results['mean_neural_improve']:+.3f}%)")

    base_gaps = [g for g in results.get("baseline_gap_pct", []) if g is not None]
    neur_gaps = [g for g in results.get("neural_gap_pct", []) if g is not None]
    if base_gaps or neur_gaps:
        mb = float(np.mean(base_gaps)) if base_gaps else float("nan")
        mn = float(np.mean(neur_gaps)) if neur_gaps else float("nan")
        if base_gaps and neur_gaps:
            print(f"    Mean Gap to BestKnown: baseline {mb:+.2f}%, neural {mn:+.2f}%")
        elif base_gaps:
            print(f"    Mean Gap to BestKnown: baseline {mb:+.2f}%")
        else:
            print(f"    Mean Gap to BestKnown: neural {mn:+.2f}%")
    print()
    print(f"  Key Metrics:")
    print(f"    Win-rate (P(neural < baseline)): {results['win_rate']*100:.1f}%")
    print(f"    Mean margin (neural vs baseline): {results['mean_margin']:+.3f}% ± {results['std_margin']:.3f}%")
    print()
    print(f"  Tail Improvement (fraction where neural beats baseline by):")
    print(f"    > 0.5%: {results['tail_05pct']*100:.1f}%")
    print(f"    > 1.0%: {results['tail_1pct']*100:.1f}%")
    print(f"    > 2.0%: {results['tail_2pct']*100:.1f}%")
    print()
    
    # Interpretation
    print(f"  Interpretation:")
    if results['win_rate'] > 0.5:
        print(f"    ✓ Neural policy outperforms baseline ({results['win_rate']*100:.0f}% win-rate)")
        if results['mean_margin'] > 0.1:
            print(f"    ✓ Meaningful improvement: {results['mean_margin']:+.3f}% average margin")
        else:
            print(f"    ~ Marginal improvement: {results['mean_margin']:+.3f}% average margin")
    elif results['win_rate'] > 0.3:
        print(f"    ~ Neural policy shows promise ({results['win_rate']*100:.0f}% win-rate)")
        print(f"      → Consider more training or hyperparameter tuning")
    else:
        print(f"    ✗ Neural policy underperforms ({results['win_rate']*100:.0f}% win-rate)")
        print(f"      → Model may need more training or different architecture")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Test pure FACO performance on TSP")
    
    # Problem parameters
    parser.add_argument("--n", type=int, default=100, help="Number of nodes")
    parser.add_argument("--n_instances", type=int, default=10, help="Number of test instances")

    # TSPLIB instances
    parser.add_argument("--tsplib", action="store_true", help="Use TSPLIB instances from --instances_dir")
    parser.add_argument(
        "--instances_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "instances"),
        help="Directory containing TSPLIB .tsp files (default: tsp/instances)",
    )
    parser.add_argument("--instance", type=str, default=None, help="Run a single TSPLIB instance (name or path)")
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["lt300", "lt700", "lt1500", "all"],
        help="Select TSPLIB instances by size scale (ignored if --max_n is set)",
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=None,
        help="Maximum TSPLIB instance size (DIMENSION) to include (e.g. 300/700/1500)",
    )
    parser.add_argument(
        "--all_tsplib",
        action="store_true",
        help="Use all matching TSPLIB instances (ignores --n_instances).",
    )
    
    # FACO parameters
    parser.add_argument("--n_ants", type=int, default=20, help="Number of ants")
    parser.add_argument("--iterations", type=int, default=100, help="Number of FACO iterations")
    parser.add_argument("--cand_list_size", type=int, default=32, help="Candidate list size")
    parser.add_argument("--backup_list_size", type=int, default=32, help="Backup list size")
    parser.add_argument("--min_new_edges", type=int, default=8, help="Minimum new edges per ant")
    parser.add_argument("--decay", type=float, default=0.9, help="Pheromone decay rate")
    parser.add_argument("--no_local_search", action="store_true", help="Disable local 2-opt search")
    parser.add_argument("--random_mode", action="store_true", 
                        help="Use uniform random selection instead of roulette wheel")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    # C++ backend threading (OpenMP)
    parser.add_argument(
        "--faco_cpp_threads",
        type=int,
        default=None,
        help="OpenMP threads for faco_cpp backend (process-global). Only affects C++ backend and require_prob=False.",
    )
    
    # Scaling test
    parser.add_argument("--scaling", action="store_true", 
                        help="Run scaling test across multiple problem sizes")
    parser.add_argument("--n_values", type=int, nargs="+", default=[20, 50, 100, 200],
                        help="Problem sizes for scaling test")
    
    # Late-iteration comparison (roulette vs random)
    parser.add_argument("--late_comparison", action="store_true",
                        help="Run late-iteration comparison (stagnation analysis)")
    parser.add_argument("--warmup_iters", type=int, default=50,
                        help="Warmup iterations before snapshot (for late_comparison)")
    parser.add_argument("--continuation_iters", type=int, default=20,
                        help="Continuation iterations after snapshot (for late_comparison)")
    
    # Neural-enhanced comparison
    parser.add_argument("--neural_comparison", action="store_true",
                        help="Run neural-enhanced vs baseline comparison")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (for neural_comparison)")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Compare multiple checkpoints (space-separated) in one run",
    )
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Residual logit scale factor (for neural_comparison)")
    parser.add_argument("--warmup", action="store_true",
                        help="Use warmup phase for neural_comparison (late-iteration mode)")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Apply C++ backend thread setting early (process-global)
    if args.faco_cpp_threads is not None:
        if has_cpp_backend():
            try:
                set_faco_cpp_threads(int(args.faco_cpp_threads))
                if not args.quiet:
                    print(f"[info] faco_cpp OpenMP threads set to {int(args.faco_cpp_threads)}")
            except Exception as e:
                print(f"[warning] Failed to set faco_cpp threads: {e}")
        else:
            print("[warning] --faco_cpp_threads was set but faco_cpp backend is not available.")

    # Prepare TSPLIB instances (optional)
    tsplib_instances: Optional[List[Dict[str, Any]]] = None
    if args.tsplib:
        instances_dir = Path(args.instances_dir)
        if args.max_n is not None:
            max_n = int(args.max_n)
        elif args.scale == "lt300":
            max_n = 300
        elif args.scale == "lt700":
            max_n = 700
        elif args.scale == "lt1500":
            max_n = 1500
        elif args.scale == "all":
            max_n = None
        else:
            # Default scale if user just says --tsplib
            max_n = 1500

        best_known_path = _resolve_best_known_path(instances_dir)
        best_known = _load_best_known(best_known_path)
        tsplib_paths = _select_tsplib_instances(
            instances_dir=instances_dir,
            n_instances=args.n_instances,
            instance=args.instance,
            max_n=max_n,
            take_all=bool(args.all_tsplib) and args.instance is None,
        )

        if not tsplib_paths:
            available = [(p.name, _read_tsplib_dimension(p)) for p in sorted(instances_dir.glob("*.tsp"))]
            print(f"No TSPLIB instances matched (max_n={max_n}). Available:")
            for name, dim in available:
                print(f"  - {name}: n={dim}")
            return

        tsplib_instances = list(
            _iter_problem_instances(
                mode="tsplib",
                device=args.device,
                n=args.n,
                n_instances=args.n_instances,
                tsplib_paths=tsplib_paths,
                best_known=best_known,
            )
        )

    # Build shared config
    config = TestConfig(
        n=int(args.n),
        n_instances=int(args.n_instances),
        instances=tsplib_instances,
        n_ants=int(args.n_ants),
        iterations=int(args.iterations),
        cand_list_size=int(args.cand_list_size),
        backup_list_size=int(args.backup_list_size),
        min_new_edges=int(args.min_new_edges),
        decay=float(args.decay),
        use_local_search=not args.no_local_search,
        random_mode=bool(args.random_mode),
        warmup_iters=int(args.warmup_iters),
        continuation_iters=int(args.continuation_iters),
        warmup=bool(args.warmup),
        checkpoint=args.checkpoint,
        checkpoints=args.checkpoints,
        gamma=float(args.gamma),
        device=str(args.device),
        seed=int(args.seed),
        verbose=not args.quiet,
        faco_cpp_threads=(int(args.faco_cpp_threads) if args.faco_cpp_threads is not None else None),
    )
    
    if args.neural_comparison:
        # Run neural-enhanced comparison
        checkpoint_list: list[str] = []
        if config.checkpoints:
            checkpoint_list.extend(list(config.checkpoints))
        if config.checkpoint:
            checkpoint_list.append(config.checkpoint)
        # De-dup while preserving order
        seen = set()
        checkpoint_list = [c for c in checkpoint_list if not (c in seen or seen.add(c))]

        if not checkpoint_list:
            # Sensible default so `--neural_comparison` runs without extra flags.
            checkpoint_list = ["checkpoints/best_model.pt"]
        
        # Determine warmup iterations: 0 by default (end-to-end), or use --warmup_iters if --warmup is set
        warmup = config.warmup_iters if config.warmup else 0

        if len(checkpoint_list) == 1:
            results = run_neural_comparison(
                config=config,
                checkpoint_path=checkpoint_list[0],
                warmup_iters=warmup,
            )
            print_neural_comparison_summary(results)
        else:
            results = run_multi_model_comparison(
                config=config,
                checkpoint_paths=checkpoint_list,
                warmup_iters=warmup,
            )
            print_multi_model_comparison_summary(results)
    elif args.late_comparison:
        # Run late-iteration comparison
        results = run_late_iteration_comparison(config=config)
        print_late_comparison_summary(results)
    elif args.scaling:
        # Run scaling test
        run_scaling_test(
            n_values=args.n_values,
            n_ants=args.n_ants,
            n_iterations=args.iterations,
            n_instances=args.n_instances,
            device=args.device,
            seed=args.seed,
        )
    else:
        # Run single configuration test
        results = run_faco_test(config=config)
        
        print_summary(results)


if __name__ == "__main__":
    main()

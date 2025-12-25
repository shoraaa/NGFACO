import copy
import time
import numpy as np
import torch
from torch.nn import functional as F

from faco import FACO  
from utils import gen_pyg_data, gen_instance  
from net import Net 


# ------------------------------------------------------------
# Hyperparams (tune)
# ------------------------------------------------------------
ROLLOUT_STEPS = 4          # n-step FACO iterations per instance rollout
PPO_EPOCHS = 2             # PPO epochs per rollout batch
MINIBATCH_SIZE = 256       # number of ant-trajectories per PPO minibatch

INV_TEMP = 1.0
CLIP_RATIO = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01
MAX_GRAD_NORM = 3.0

# FACO params
K_NEAREST = 20
MIN_NEW_EDGES = 8
SAMPLE_TWO_OPT = True


# ------------------------------------------------------------
# Helpers: build "state-conditioned" edge_attr
# (distance, in_best, pheromone) depending on model input size
# ------------------------------------------------------------
def _edge_feat_in_dim(model, pyg):
    # best effort: detect what emb_net expects
    if hasattr(model, "emb_net") and hasattr(model.emb_net, "e_lin0"):
        return int(model.emb_net.e_lin0.in_features)
    # fallback: whatever pyg currently holds
    if pyg.edge_attr is None:
        return 1
    return int(pyg.edge_attr.size(-1))


def attach_faco_state_to_pyg(pyg_base, solver: FACO, ref_flat: np.ndarray, tau_sparse: torch.Tensor):
    """
    Returns a pyg Data object with edge_attr augmented based on FACO state:
      - dist (always)
      - in_best edge indicator (optional)
      - pheromone value (optional)
    """
    pyg = pyg_base  # we mutate in-place, caller should pass a cloned pyg if needed

    device = pyg.x.device
    n = solver.n

    # base distance feature (E,1)
    dist_feat = pyg.edge_attr
    if dist_feat is None:
        # If your pyg doesn't store edge_attr, create from solver.dist on edge_index
        u = pyg.edge_index[0]
        v = pyg.edge_index[1]
        dist_feat = solver.dist_t[u, v].unsqueeze(-1)
    else:
        # enforce shape (E,1) if user already has more dims
        if dist_feat.dim() == 1:
            dist_feat = dist_feat.unsqueeze(-1)
        if dist_feat.size(-1) > 1:
            dist_feat = dist_feat[:, :1]

    # detect expected input dims
    in_dim = _edge_feat_in_dim(model=net, pyg=pyg)  # uses global net in notebook; adjust if needed

    if in_dim == 1:
        pyg.edge_attr = dist_feat
        return pyg

    # build best-edge mask on full (n,n) then gather on edges
    best_adj = torch.zeros((n, n), device=device, dtype=torch.float32)
    u_b = torch.tensor(ref_flat[:-1], device=device, dtype=torch.long)
    v_b = torch.tensor(ref_flat[1:], device=device, dtype=torch.long)
    best_adj[u_b, v_b] = 1.0

    u_e = pyg.edge_index[0]
    v_e = pyg.edge_index[1]
    in_best = best_adj[u_e, v_e].unsqueeze(-1)  # (E,1)

    if in_dim == 2:
        pyg.edge_attr = torch.cat([dist_feat, in_best], dim=-1)
        return pyg

    # in_dim >= 3: add pheromone on those edges
    # expand tau_sparse (n,k) to full (n,n) on kNN edges only
    nn = torch.from_numpy(solver.nn_list).to(device)  # (n,k)
    rows = torch.arange(n, device=device).unsqueeze(1).expand(n, solver.k)
    tau_full = torch.zeros((n, n), device=device, dtype=torch.float32)
    tau_full[rows, nn.clamp_min(0)] = tau_sparse.detach().clamp_min(1e-10)

    tau_e = tau_full[u_e, v_e].unsqueeze(-1)  # (E,1)

    if in_dim == 3:
        pyg.edge_attr = torch.cat([dist_feat, in_best, tau_e], dim=-1)
        return pyg

    # if you later add more edge channels, append here
    pyg.edge_attr = torch.cat([dist_feat, in_best, tau_e], dim=-1)
    return pyg


# ------------------------------------------------------------
# Rollout collector
# ------------------------------------------------------------
@torch.no_grad()
def faco_rollout_collect(n, model, n_ants, rollout_steps=ROLLOUT_STEPS):
    """
    Collect a rollout of length rollout_steps.
    Returns a list of experience dicts (one per ant trajectory per step),
    and also a fresh solver (for debug/optional usage).
    """
    model.eval()

    demands, distances = gen_instance(n, device)              # torch
    pyg_base = gen_pyg_data(demands, distances, device)       # torch_geometric.data.Data

    solver = FACO(
        distances=distances,
        demand=demands,
        n_ants=n_ants,
        k_nearest=K_NEAREST,
        capacity=int(getattr(__import__("aco"), "CAPACITY", 50)) if False else 50,  # keep 50 unless you override
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        min_new_edges=MIN_NEW_EDGES,
        sample_two_opt=SAMPLE_TWO_OPT,
        device=device,
    )

    exps = []

    for t in range(rollout_steps):
        # snapshot state for replay
        ref_flat = solver.best_flat.copy()
        tau_snap = solver.pheromone_sparse.detach().clone()

        # build pyg with state-features
        pyg = copy.copy(pyg_base)
        pyg = attach_faco_state_to_pyg(pyg, solver, ref_flat, tau_snap)

        # forward -> heuristic + value
        out = model(pyg, return_value=True) if "return_value" in model.forward.__code__.co_varnames else model(pyg)
        if isinstance(out, tuple) or isinstance(out, list):
            heu_vec, vpred = out
        else:
            heu_vec, vpred = out, torch.zeros((), device=device)

        # heuristics -> full matrix -> solver.set_heuristic
        heu_full = model.reshape(pyg, heu_vec) + 1e-10
        solver.set_heuristic(heu_full)

        # sample (old policy) from FACO
        costs_np, flats, touched, old_logps, traces = solver.sample(invtemp=INV_TEMP, require_prob=True)

        rewards = -torch.tensor(costs_np, device=device, dtype=torch.float32)  # (n_ants,)

        # store each ant trajectory as one PPO sample
        # vpred should be scalar baseline for this state
        if torch.is_tensor(vpred) and vpred.numel() > 1:
            vpred_scalar = vpred.mean()
        else:
            vpred_scalar = vpred.reshape(())

        for i in range(n_ants):
            exps.append({
                "ref_flat": ref_flat,              # numpy
                "tau": tau_snap,                   # torch (n,k)
                "old_logp": old_logps[i].detach(), # torch scalar
                "reward": rewards[i].detach(),     # torch scalar
                "trace": traces[i],                # FACOTrace
                "pyg_base": pyg_base,              # static graph
                "distances": distances,            # torch
                "demands": demands,                # torch
            })

        # advance environment exactly like one FACO iteration
        best_i = int(np.argmin(costs_np))
        best_cost = float(costs_np[best_i])
        best_flat = flats[best_i]

        if best_cost < solver.best_cost:
            solver.best_cost = best_cost
            solver.best_flat = best_flat

        solver._update_pheromone_from_flat(best_flat, best_cost)

    return exps


# ------------------------------------------------------------
# PPO update using replay_logp
# ------------------------------------------------------------
def ppo_update_from_exps(model, optimizer, exps, clip_ratio=CLIP_RATIO,
                         value_coeff=VALUE_COEFF, entropy_coeff=ENTROPY_COEFF):
    model.train()

    # shuffle indices
    idx = torch.randperm(len(exps))
    exps = [exps[i] for i in idx.tolist()]

    total_policy = 0.0
    total_value = 0.0
    total_ent = 0.0
    total_loss = 0.0
    n_batches = 0

    for _ in range(PPO_EPOCHS):
        for start in range(0, len(exps), MINIBATCH_SIZE):
            batch = exps[start:start + MINIBATCH_SIZE]
            if len(batch) == 0:
                continue

            policy_losses = []
            value_losses = []
            entropies = []

            for item in batch:
                distances = item["distances"]
                demands = item["demands"]

                # reconstruct a solver for this sample state (cheap for n<=101; optimize later)
                solver = FACO(
                    distances=distances,
                    demand=demands,
                    n_ants=1,  # we replay one trace at a time
                    k_nearest=K_NEAREST,
                    capacity=50,
                    decay=0.9,
                    alpha=1.0,
                    beta=1.0,
                    min_new_edges=MIN_NEW_EDGES,
                    sample_two_opt=False,  # LS not part of logπ; keep off for replay stability
                    device=device,
                )

                # restore state snapshot
                ref_flat = item["ref_flat"]
                solver.best_flat = ref_flat
                solver.pheromone_sparse = item["tau"].detach().clone()

                # build pyg with restored state features
                pyg_base = item["pyg_base"]
                pyg = copy.copy(pyg_base)
                pyg = attach_faco_state_to_pyg(pyg, solver, ref_flat, solver.pheromone_sparse)

                # forward current policy
                out = model(pyg, return_value=True) if "return_value" in model.forward.__code__.co_varnames else model(pyg)
                if isinstance(out, tuple) or isinstance(out, list):
                    heu_vec, vpred = out
                else:
                    heu_vec, vpred = out, torch.zeros((), device=device)

                heu_full = model.reshape(pyg, heu_vec) + 1e-10
                solver.set_heuristic(heu_full)

                # new logp from replay
                new_logp = solver.replay_logp(
                    item["trace"],
                    invtemp=INV_TEMP,
                    ref_flat=ref_flat,
                    prob_sparse=solver.prob_sparse_torch(INV_TEMP),
                )

                old_logp = item["old_logp"]
                reward = item["reward"]

                # value scalar
                if torch.is_tensor(vpred) and vpred.numel() > 1:
                    vpred_scalar = vpred.mean()
                else:
                    vpred_scalar = vpred.reshape(())

                # advantage (simple one-step, terminal reward)
                adv = (reward - vpred_scalar.detach())
                # optional normalization inside minibatch
                # (comment out if you already shared-energy-normalize elsewhere)
                # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_loss = F.mse_loss(vpred_scalar, reward)

                # entropy bonus (coarse): entropy of normalized prob_sparse rows
                prob = solver.prob_sparse_torch(INV_TEMP)
                prob = prob / (prob.sum(dim=1, keepdim=True) + 1e-10)
                entropy = -(prob * torch.log(prob + 1e-10)).sum(dim=1).mean()

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy = torch.stack(entropies).mean()

            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_policy += float(policy_loss.detach().cpu())
            total_value += float(value_loss.detach().cpu())
            total_ent += float(entropy.detach().cpu())
            total_loss += float(loss.detach().cpu())
            n_batches += 1

    return {
        "policy_loss": total_policy / max(1, n_batches),
        "value_loss": total_value / max(1, n_batches),
        "entropy": total_ent / max(1, n_batches),
        "total_loss": total_loss / max(1, n_batches),
    }


# ------------------------------------------------------------
# Train step: collect rollout + PPO update
# ------------------------------------------------------------
def train_instance_ppo(n, model, optimizer, n_ants):
    exps = faco_rollout_collect(n, model, n_ants, rollout_steps=ROLLOUT_STEPS)
    stats = ppo_update_from_exps(model, optimizer, exps)
    return stats


def train_epoch(n_node, n_ants, steps_per_epoch, net, optimizer):
    stats_acc = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
    for _ in range(steps_per_epoch):
        st = train_instance_ppo(n_node, net, optimizer, n_ants)
        for k in stats_acc:
            stats_acc[k] += st[k]
    for k in stats_acc:
        stats_acc[k] /= max(1, steps_per_epoch)
    return stats_acc


# ------------------------------------------------------------
# Validation: run FACO for T steps, refreshing heuristic each step
# ------------------------------------------------------------
@torch.no_grad()
def validation(n_node, n_ants, net, n_val=50, T_eval=5):
    net.eval()
    sum_bl = 0.0
    sum_best = 0.0
    sum_faco = 0.0

    for _ in range(n_val):
        demands, distances = gen_instance(n_node, device)
        pyg_base = gen_pyg_data(demands, distances, device)

        solver = FACO(
            distances=distances,
            demand=demands,
            n_ants=n_ants,
            k_nearest=K_NEAREST,
            capacity=50,
            decay=0.9,
            alpha=1.0,
            beta=1.0,
            min_new_edges=MIN_NEW_EDGES,
            sample_two_opt=SAMPLE_TWO_OPT,
            device=device,
        )

        # baseline (first step sampling mean)
        ref_flat = solver.best_flat.copy()
        tau = solver.pheromone_sparse.detach().clone()
        pyg = copy.copy(pyg_base)
        pyg = attach_faco_state_to_pyg(pyg, solver, ref_flat, tau)
        out = net(pyg, return_value=True) if "return_value" in net.forward.__code__.co_varnames else net(pyg)
        heu_vec = out[0] if isinstance(out, (tuple, list)) else out
        heu_full = net.reshape(pyg, heu_vec) + 1e-10
        solver.set_heuristic(heu_full)

        costs_np, flats, _touched = solver.sample(invtemp=INV_TEMP, require_prob=False)
        sum_bl += float(np.mean(costs_np))
        sum_best += float(np.min(costs_np))

        # run T_eval steps, refreshing heuristic each step
        for _t in range(T_eval):
            ref_flat = solver.best_flat.copy()
            tau = solver.pheromone_sparse.detach().clone()

            pyg = copy.copy(pyg_base)
            pyg = attach_faco_state_to_pyg(pyg, solver, ref_flat, tau)

            out = net(pyg, return_value=True) if "return_value" in net.forward.__code__.co_varnames else net(pyg)
            heu_vec = out[0] if isinstance(out, (tuple, list)) else out
            heu_full = net.reshape(pyg, heu_vec) + 1e-10
            solver.set_heuristic(heu_full)

            costs_np, flats, _touched = solver.sample(invtemp=INV_TEMP, require_prob=False)
            best_i = int(np.argmin(costs_np))
            best_cost = float(costs_np[best_i])
            best_flat = flats[best_i]

            if best_cost < solver.best_cost:
                solver.best_cost = best_cost
                solver.best_flat = best_flat
            solver._update_pheromone_from_flat(best_flat, best_cost)

        sum_faco += float(solver.best_cost)

    return sum_bl / n_val, sum_best / n_val, sum_faco / n_val


def train(n_node, n_ants, steps_per_epoch, epochs):
    net = Net().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    val_best = 1e30
    for epoch in range(epochs):
        t0 = time.time()
        stats = train_epoch(n_node, n_ants, steps_per_epoch, net, optimizer)
        bl, best, faco_best = validation(n_node, n_ants, net, n_val=50, T_eval=T)

        dt = time.time() - t0
        print(
            f"epoch {epoch:03d} | "
            f"loss {stats['total_loss']:.4f} (pi {stats['policy_loss']:.4f}, v {stats['value_loss']:.4f}, ent {stats['entropy']:.4f}) | "
            f"val: bl {bl:.2f}, best {best:.2f}, FACO(T={T}) {faco_best:.2f} | "
            f"time {dt:.1f}s"
        )

        if faco_best < val_best:
            val_best = faco_best
            torch.save(net.state_dict(), f"../pretrained/cvrp/cvrp{n_node}_ppo_nstep.pt")

    return net

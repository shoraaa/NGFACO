from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
import torch_geometric.nn as gnn


class NodeEncoder(nn.Module):
    """
    Lightweight node embedding encoder using message passing on kNN graph.
    Run once per instance to get node embeddings.
    
    Args:
        node_feats: input node feature dim (2 for coords)
        edge_feats: input edge feature dim (1 for distance)
        units: hidden dimension
        depth: number of message passing layers (keep small, e.g., 3-6)
        act_fn: activation function name
    """
    def __init__(self, node_feats=2, edge_feats=1, units=64, depth=4, act_fn='silu'):
        super().__init__()
        self.units = units
        self.depth = depth
        self.act_fn = getattr(F, act_fn)
        
        # Initial projections
        self.node_proj = nn.Linear(node_feats, units)
        self.edge_proj = nn.Linear(edge_feats, units)
        
        # Message passing layers (simple but effective)
        self.mp_layers = nn.ModuleList([
            MPLayer(units, act_fn) for _ in range(depth)
        ])
        
        # Layer norms for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(units) for _ in range(depth)
        ])
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: node features (n, node_feats)
            edge_index: (2, num_edges) 
            edge_attr: edge features (num_edges, edge_feats)
        Returns:
            node_emb: (n, units)
        """
        # Initial projections
        h = self.act_fn(self.node_proj(x))
        e = self.act_fn(self.edge_proj(edge_attr))
        
        # Message passing with residual connections
        for i in range(self.depth):
            h_new = self.mp_layers[i](h, edge_index, e)
            h = h + h_new  # residual
            h = self.layer_norms[i](h)
        
        return h


class MPLayer(MessagePassing):
    """
    Simple message passing layer with edge features.
    """
    def __init__(self, units, act_fn='silu'):
        super().__init__(aggr='mean')
        self.units = units
        self.act_fn = getattr(F, act_fn)
        
        # Message MLP: combines source node + edge features
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * units, units),
            nn.SiLU(),
            nn.Linear(units, units)
        )
        
        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * units, units),
            nn.SiLU(),
            nn.Linear(units, units)
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # x_j: source node features, edge_attr: edge features
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))
    
    def update(self, aggr_out, x):
        # Combine aggregated messages with node's own features
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class ResidualScorer(nn.Module):
    """
    Lightweight scorer for residual logits.
    Takes current node embedding, candidate node embeddings, and edge features.
    Outputs unbounded real-valued residual logits.
    
    Called during FACO sampling for each step.
    """
    def __init__(self, units=64, hidden=64, act_fn='silu'):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        
        # Input: h_curr (units) + h_cand (units) + edge_feats (variable)
        # We'll handle edge features flexibly
        self.scorer = nn.Sequential(
            nn.Linear(2 * units + 1, hidden),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)  # NO sigmoid - output raw logits
        )

            
    def forward(self, h_curr, h_cand, dist):
        """
        Score candidate edges from current node.
        
        Args:
            h_curr: current node embedding (units,) or (1, units)
            h_cand: candidate node embeddings (k, units)
            dist: distances to candidates (k,) or (k, 1)
        
        Returns:
            residual_logits: (k,) unbounded real values
        """
        if h_curr.dim() == 1:
            h_curr = h_curr.unsqueeze(0)
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        
        k = h_cand.shape[0]
        h_curr_expanded = h_curr.expand(k, -1)  # (k, units)
        
        # Concatenate features
        feats = torch.cat([h_curr_expanded, h_cand, dist], dim=-1)
        
        # Score (no sigmoid - raw logits)
        logits = self.scorer(feats).squeeze(-1)  # (k,)
        return logits


class ResidualScorerWithContext(nn.Module):
    """
    Extended scorer that can use additional context features:
    - pheromone info
    - stagnation features
    - reference solution features
    """
    def __init__(self, units=64, hidden=64, extra_feats=0, act_fn='silu'):
        super().__init__()
        # Input: h_curr + h_cand + dist + extra_feats
        input_dim = 2 * units + 1 + extra_feats
        
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)  # raw logits
        )
    
    def forward(self, h_curr, h_cand, dist, extra_feats=None):
        """
        Args:
            h_curr: (units,) or (1, units)
            h_cand: (k, units)
            dist: (k,) or (k, 1)
            extra_feats: optional (k, extra_feats) - pheromone, stagnation, etc.
        """
        if h_curr.dim() == 1:
            h_curr = h_curr.unsqueeze(0)
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        
        k = h_cand.shape[0]
        h_curr_expanded = h_curr.expand(k, -1)
        
        feats = [h_curr_expanded, h_cand, dist]
        if extra_feats is not None:
            if extra_feats.dim() == 1:
                extra_feats = extra_feats.unsqueeze(-1)
            feats.append(extra_feats)
        
        feats = torch.cat(feats, dim=-1)
        return self.scorer(feats).squeeze(-1)


class Net(nn.Module):
    """
    Residual-on-FACO network.
    
    Two-stage architecture:
    1. encode(pyg) -> node_emb  (run once per instance)
    2. score_candidates(node_emb, curr, cand_nodes, dist) -> residual_logits
    
    The residual logits modify FACO's base weights:
        w = w_base * exp(gamma * residual_logit)
    """
    def __init__(self, node_feats=2, edge_feats=1, units=64, depth=4, 
                 scorer_hidden=64, extra_feats=0, act_fn='silu'):
        super().__init__()
        self.units = units
        
        # Node embedding encoder
        self.encoder = NodeEncoder(
            node_feats=node_feats,
            edge_feats=edge_feats,
            units=units,
            depth=depth,
            act_fn=act_fn
        )
        
        # Residual scorer
        if extra_feats > 0:
            self.scorer = ResidualScorerWithContext(
                units=units,
                hidden=scorer_hidden,
                extra_feats=extra_feats,
                act_fn=act_fn
            )
        else:
            self.scorer = ResidualScorer(
                units=units,
                hidden=scorer_hidden,
                act_fn=act_fn
            )
        
        # Cache for node embeddings (set during encode)
        self._node_emb = None
        self._dist_matrix = None
    
    def encode(self, pyg):
        """
        Encode instance graph to get node embeddings.
        Call once per instance, then reuse for all sampling steps.
        
        Args:
            pyg: PyG Data object with x, edge_index, edge_attr
        
        Returns:
            node_emb: (n, units)
        """
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        self._node_emb = self.encoder(x, edge_index, edge_attr)
        return self._node_emb
    
    def score_candidates(self, curr, cand_nodes, dist, extra_feats=None):
        """
        Score candidate edges from current node.
        
        Args:
            curr: current node index (int)
            cand_nodes: candidate node indices (k,) tensor
            dist: distances to candidates (k,) tensor
            extra_feats: optional additional features (k, extra_feats)
        
        Returns:
            residual_logits: (k,) unbounded real values
        """
        assert self._node_emb is not None, "Must call encode() first"
        
        h_curr = self._node_emb[curr]
        h_cand = self._node_emb[cand_nodes]
        
        # Only pass extra_feats if scorer is ResidualScorerWithContext
        if isinstance(self.scorer, ResidualScorerWithContext) and extra_feats is not None:
            return self.scorer(h_curr, h_cand, dist, extra_feats)
        else:
            return self.scorer(h_curr, h_cand, dist)
    
    def score_all_edges(self, edge_index, dist):
        """
        Score all edges at once (for batch training).
        
        Args:
            edge_index: (2, num_edges)
            dist: (num_edges,)
        
        Returns:
            residual_logits: (num_edges,)
        """
        assert self._node_emb is not None, "Must call encode() first"
        
        src, dst = edge_index
        h_src = self._node_emb[src]
        h_dst = self._node_emb[dst]
        
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        
        feats = torch.cat([h_src, h_dst, dist], dim=-1)
        return self.scorer.scorer(feats).squeeze(-1)
    
    def score_knn_matrix(
        self,
        nn_torch: torch.Tensor,      # (n,k) candidate node ids
        dist_nk: torch.Tensor,       # (n,k) distances (or any scalar edge feature)
        extra_nkC: Optional[torch.Tensor] = None,  # (n,k,C)
    ) -> torch.Tensor:
        """
        Vectorized scoring for all (u, j) edges in the kNN candidate set.
        Returns residual logits (n,k).
        """
        assert self._node_emb is not None, "Must call encode()/encode_nodes() first"
        h_u = self._node_emb.unsqueeze(1).expand(-1, nn_torch.size(1), -1)   # (n,k,d)
        h_v = self._node_emb[nn_torch]                                      # (n,k,d)

        if dist_nk.dim() == 2:
            dist_feat = dist_nk.unsqueeze(-1)                               # (n,k,1)
        else:
            raise ValueError("dist_nk must be (n,k)")

        feats = [h_u, h_v, dist_feat]
        if extra_nkC is not None:
            if extra_nkC.dim() == 2:
                extra_nkC = extra_nkC.unsqueeze(-1)                         # (n,k,C=1)
            feats.append(extra_nkC)

        feats = torch.cat(feats, dim=-1)                                    # (n,k,F)
        logits = self.scorer.scorer(feats).squeeze(-1)                      # (n,k)
        return logits
    
    def forward(self, pyg):
        """
        Full forward pass: encode + score all edges.
        Useful for training.
        
        Returns:
            residual_logits: (num_edges,) 
        """
        self.encode(pyg)
        dist = pyg.edge_attr
        if dist.dim() == 2:
            dist = dist.squeeze(-1)
        return self.score_all_edges(pyg.edge_index, dist)
    
    def forward_knn(self, pyg, nn_torch, dist_nk, extra_nkC=None):
        self.encode(pyg)
        return self.score_knn_matrix(nn_torch, dist_nk, extra_nkC)

    def freeze_encoder(self):
        """Freeze encoder weights (for fine-tuning scorer only)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def clear_cache(self):
        """Clear cached embeddings."""
        self._node_emb = None
        self._dist_matrix = None
    
    @staticmethod
    def reshape(pyg, vector):
        """
        Turn edge vector into matrix with zero padding.
        Kept for backward compatibility.
        """
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix


# Legacy compatibility aliases
class MLP(nn.Module):
    """Legacy MLP class - kept for compatibility."""
    @property
    def device(self):
        return self._dummy.device
    
    def __init__(self, units_list, act_fn, use_sigmoid=False):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.use_sigmoid = use_sigmoid
        self.lins = nn.ModuleList([
            nn.Linear(self.units_list[i], self.units_list[i + 1]) 
            for i in range(self.depth)
        ])
    
    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            elif self.use_sigmoid:
                x = torch.sigmoid(x)
            # else: linear output (for residual logits)
        return x
        
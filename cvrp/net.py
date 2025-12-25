import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn

# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=1, edge_feats=1, units=32, act_fn='silu', agg_fn='mean'):
        """
        feats: node feature dim
        edge_feats: edge feature dim (default 1; will be 3 when using best_edge_attr + tau_edge_attr)
        """
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.edge_feats = edge_feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')

        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for _ in range(self.depth)])

        # PATCH: edge input dim is now edge_feats (was hardcoded 1)
        self.e_lin0 = nn.Linear(self.edge_feats, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for _ in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        # x: (N, feats)
        # edge_attr: (E, edge_feats)
        w = edge_attr

        x = self.v_lin0(x)
        x = self.act_fn(x)

        w = self.e_lin0(w)
        w = self.act_fn(w)

        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)

            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)

            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))

        return w


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList(
            [nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)]
        )

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x)  # last layer
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)


class Net(nn.Module):
    def __init__(self, value_head: bool = False, use_state_edge_features: bool = True):
        super().__init__()
        self.value_head = value_head
        self.use_state_edge_features = use_state_edge_features

        edge_feats = 3 if use_state_edge_features else 1
        self.emb_net = EmbNet(edge_feats=edge_feats)
        self.par_net_heu = ParNet()          # heuristic in (0,1) via sigmoid

        if self.value_head:
            self.par_net_val = ValNet()      # PATCH: linear value output


    def forward(self, pyg, return_value: bool = False):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr

        # PATCH: concatenate state-conditioned edge features if present
        if self.use_state_edge_features and hasattr(pyg, "best_edge_attr") and hasattr(pyg, "tau_edge_attr"):
            # edge_attr: (E,1), best_edge_attr: (E,1), tau_edge_attr: (E,1) => (E,3)
            edge_attr = torch.cat([edge_attr, pyg.best_edge_attr, pyg.tau_edge_attr], dim=-1)

        emb = self.emb_net(x, edge_index, edge_attr)   # (E, units)
        heu = self.par_net_heu(emb)                    # (E,)

        if return_value:
            if not self.value_head:
                raise ValueError("Net was created with value_head=False but return_value=True was requested.")
            # simple global pooling over edges to get a single embedding, then value
            emb_graph = emb.mean(dim=0, keepdim=True)   # (1, units)
            val = self.par_net_val(emb_graph).squeeze(-1)  # (1,) -> scalar-ish
            return heu, val

        return heu

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False

    @staticmethod
    def reshape(pyg, vector):
        """Turn phe/heu vector into matrix with zero padding."""
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix


class ValueMLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList(
            [nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)]
        )

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            # NOTE: last layer is linear (no sigmoid)
        return x


class ValNet(ValueMLP):
    """
    Value head: outputs a scalar (unbounded) value estimate.
    """
    def __init__(self, depth=3, units=32, act_fn='silu'):
        super().__init__([units] * depth + [1], act_fn)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)
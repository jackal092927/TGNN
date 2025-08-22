import math, os, json
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ Mamba Block ------------------------------

class MambaBlock(nn.Module):
    def __init__(self, hidden_dim, state_dim=16, expand_factor=2, dt_rank=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.expand_dim = int(hidden_dim*expand_factor)
        self.dt_rank = dt_rank or max(1, hidden_dim//16)
        self.input_proj = nn.Linear(hidden_dim, self.expand_dim*2)
        self.dt_proj = nn.Linear(self.dt_rank, self.expand_dim)
        self.A_log = nn.Parameter(torch.randn(self.expand_dim, state_dim))
        self.D = nn.Parameter(torch.randn(self.expand_dim))
        self.delta_proj = nn.Linear(self.expand_dim, self.dt_rank)
        self.B_proj = nn.Linear(self.expand_dim, state_dim)
        self.C_proj = nn.Linear(self.expand_dim, state_dim)
        self.output_proj = nn.Linear(self.expand_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self._init()

    def _init(self):
        nn.init.normal_(self.A_log, mean=0, std=0.1)
        with torch.no_grad():
            self.A_log.data = -torch.exp(self.A_log.data)
        for m in [self.dt_proj, self.B_proj, self.C_proj, self.delta_proj]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.D)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x_proj = self.input_proj(x)
        x, gate = x_proj.chunk(2, dim=-1)
        x = self.act(x)
        delta = F.softplus(self.dt_proj(self.delta_proj(x)))
        B, C = self.B_proj(x), self.C_proj(x)
        A = -torch.exp(self.A_log)
        y = self._scan(x, delta, A, B, C)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * self.act(gate)
        return self.output_proj(y) + res

    def _scan(self, x, delta, A, B, C):
        B_, T, D = x.shape
        Sd = A.shape[1]
        h = torch.zeros(B_, D, Sd, device=x.device)
        outs = []
        for t in range(T):
            x_t, d_t = x[:, t], delta[:, t]
            B_t, C_t = B[:, t], C[:, t]
            A_disc = torch.exp(d_t.unsqueeze(-1) * A.unsqueeze(0))
            h = A_disc * h + (d_t.unsqueeze(-1) * B_t.unsqueeze(1)) * x_t.unsqueeze(-1)
            y_t = torch.sum(C_t.unsqueeze(1) * h, dim=-1)
            outs.append(y_t)
        return torch.stack(outs, dim=1)

# ------------------------------ Utilities ------------------------------

def create_sincos_positional_embeddings(max_nodes, pos_dim):
    import math
    position = torch.arange(max_nodes).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, pos_dim, 2).float() * -(math.log(10000.0)/pos_dim))
    pe = torch.zeros(max_nodes, pos_dim)
    pe[:,0::2] = torch.sin(position*div); pe[:,1::2] = torch.cos(position*div)
    return pe

# ------------------------- Simple Spatial Layer -------------------------

class SpatialLayer(nn.Module):
    """Ungated spatial layer: linear message + adjacency aggregation."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.msg = nn.Linear(input_dim, hidden_dim)
        self.upd = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        m = self.msg(x)
        agg = adj @ m
        out = self.upd(torch.cat([x, agg], dim=-1))
        out = self.act(out)
        return self.norm(out)

# -------------------------- Late Edge Gate Head -------------------------

class LateEdgeGateHead(nn.Module):
    """Computes gate logits & probs from final node states; not used for aggregation.
    scores = Qh @ (Kh)^T / (sqrt(H)*tau) + bias + phi(edge_feats) (optional)
    """
    def __init__(self, hidden_dim, gate_temperature=1.0, edge_feat_dim=0):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.tensor(-1.0))
        self.edge_feat = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)) if edge_feat_dim>0 else None
        self.gate_temperature = gate_temperature

    def forward(self, h, edge_features: Optional[torch.Tensor] = None):
        # h: [N, H]; edge_features: [N, N, F] or None
        q, k = self.q(h), self.k(h)
        scores = (q @ k.T) / (math.sqrt(q.shape[-1]) * max(self.gate_temperature, 1e-6))
        if self.edge_feat is not None and edge_features is not None:
            scores = scores + self.edge_feat(edge_features).squeeze(-1)
        logits = scores + self.bias
        probs = torch.sigmoid(logits)
        return probs, logits

# ------------------------------ GraphMamba ------------------------------

class GraphMambaLastGate(nn.Module):
    """GraphMamba with ungated spatial layers, late gate head and gate-aware predictor."""
    def __init__(self, max_nodes, pos_dim=256, hidden_dim=64, gnn_layers=2,
                 mamba_state_dim=16, dropout=0.1,
                 gate_temperature=1.0, edge_feat_dim=0):
        super().__init__()
        self.max_nodes = max_nodes
        self.pos_embeddings = nn.Parameter(create_sincos_positional_embeddings(max_nodes, pos_dim), requires_grad=False)
        self.input = SpatialLayer(pos_dim, hidden_dim)
        self.layers = nn.ModuleList([SpatialLayer(hidden_dim, hidden_dim) for _ in range(gnn_layers-1)])
        self.mamba_encoder = MambaBlock(hidden_dim=hidden_dim, state_dim=mamba_state_dim)
        self.late_head = LateEdgeGateHead(hidden_dim, gate_temperature=gate_temperature, edge_feat_dim=edge_feat_dim)
        # Gate-aware predictor: uses node states, neighbors (gate-weighted), and the gate(u,v)
        in_dim = hidden_dim*6 + 1  # h_u, h_v, |hu-hv|, hu+hv, n_u, n_v, and scalar g_uv
        self.edge_predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden_dim//2, 1), nn.Sigmoid()
        )
        self.drop = nn.Dropout(dropout)

    # ---- sequence encoding ----
    def encode_graph(self, adj, edge_features: Optional[torch.Tensor]=None):
        N = adj.shape[0]
        x = self.input(self.pos_embeddings[:N], adj)
        for lyr in self.layers:
            x = self.drop(lyr(x, adj))
        return x  # [N,H]

    def forward_sequence(self, graph_sequence: List[torch.Tensor], edge_feature_seq: Optional[List[torch.Tensor]] = None):
        """Returns (node_states_per_t, gates_probs_per_t, gates_logits_per_t)."""
        Hs = []
        for t, A in enumerate(graph_sequence):
            ef = edge_feature_seq[t] if edge_feature_seq is not None else None
            h = self.encode_graph(A, ef)
            Hs.append(h)
        # Temporal encoder over time for each node independently
        seq = torch.stack(Hs, dim=0).transpose(0,1)  # [N,T,H]
        outs = []
        for i in range(seq.shape[0]):
            outs.append(self.mamba_encoder(seq[i:i+1]).squeeze(0))  # [T,H]
        outs = torch.stack(outs, dim=0).transpose(0,1)  # [T,N,H]
        # Late gates on temporally-encoded states for interpretability + prediction context
        Gs, Ls = [], []
        for t in range(outs.shape[0]):
            ef_t = None
            if edge_feature_seq is not None and len(edge_feature_seq) > t:
                ef_t = edge_feature_seq[t]
            g, l = self.late_head(outs[t], ef_t)
            Gs.append(g); Ls.append(l)
        return outs, Gs, Ls

    # ---- gate-aware prediction ----
    def predict_next_edges(self, node_states_t, gate_probs_t, edge_pairs, adj_matrix_t: Optional[torch.Tensor] = None):
        """node_states_t: [N,H] = temporal states at time t
           gate_probs_t: [N,N]
           edge_pairs: LongTensor [M,2]
           adj_matrix_t: [N,N] optional adjacency mask at time t (0/1) for neighbor summaries/scalar gate
        """
        H = node_states_t
        G = gate_probs_t
        if adj_matrix_t is not None:
            G_eff = G * adj_matrix_t
        else:
            G_eff = G
        src = H[edge_pairs[:,0]]
        dst = H[edge_pairs[:,1]]
        # gate-weighted neighbor summaries
        Nu = G_eff[edge_pairs[:,0]] @ H  # [M,H]
        Nv = G_eff[edge_pairs[:,1]] @ H  # [M,H]
        guv = G_eff[edge_pairs[:,0], edge_pairs[:,1]].unsqueeze(-1)  # [M,1]
        feat = torch.cat([src, dst, (src-dst).abs(), (src+dst), Nu, Nv, guv], dim=-1)
        return self.edge_predictor(feat).squeeze(-1)

    # ---- regularizers on late gates ----
    @staticmethod
    def sparsity_loss(gates: torch.Tensor, mask: Optional[torch.Tensor]=None):
        if mask is not None:
            return (gates * mask).sum() / (mask.sum() + 1e-8)
        return gates.mean()

    @staticmethod
    def temporal_tv_loss(g_t: torch.Tensor, g_prev: torch.Tensor, mask: Optional[torch.Tensor]=None):
        if mask is None:
            mask = torch.ones_like(g_t)
        return ((g_t - g_prev).abs() * mask).sum() / (mask.sum() + 1e-8)

    # ---- explanations ----
    @staticmethod
    def gradient_x_gate(yhat: torch.Tensor, gates_prob: torch.Tensor, retain_graph: bool=False):
        grad = torch.autograd.grad(yhat, gates_prob, retain_graph=retain_graph, create_graph=False)[0]
        attr = gates_prob.detach() * torch.relu(grad.detach())
        return attr  # [N,N]

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
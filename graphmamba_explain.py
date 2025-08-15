"""
GraphMamba (Self-Explaining): Graph Neural Network with Mamba State-Space Model
-------------------------------------------------------------------------------
Adds *edge gating* at the spatial GNN stage and exposes the gates as
edge importances. During training, you can add:
  - sparsity loss:    lambda_sparse * mean(gates_on_edges)
  - temporal smooth:  lambda_tv     * mean(|gates_t - gates_{t-1}| on common edges)
to obtain faithful, temporally coherent explanations.

Key additions vs. original:
- PositionalGNNLayer now supports per-edge gates g_ij in (0,1)
- GraphMamba.forward_sequence(..., return_gates=True) returns a list of gate matrices
- Utility helpers to compute sparsity/TV losses and to extract top-k edges

Author: chatgpt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import math
from typing import List, Tuple, Optional

# ------------------------------ Mamba Block ------------------------------

class MambaBlock(nn.Module):
    def __init__(self, hidden_dim, state_dim=16, dt_rank=None, expand_factor=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.dt_rank = dt_rank or max(1, hidden_dim // 16)
        self.expand_dim = int(hidden_dim * expand_factor)

        self.input_proj = nn.Linear(hidden_dim, self.expand_dim * 2)

        self.dt_proj = nn.Linear(self.dt_rank, self.expand_dim)
        self.A_log = nn.Parameter(torch.randn(self.expand_dim, self.state_dim))
        self.D = nn.Parameter(torch.randn(self.expand_dim))

        self.delta_proj = nn.Linear(self.expand_dim, self.dt_rank)
        self.B_proj = nn.Linear(self.expand_dim, self.state_dim)
        self.C_proj = nn.Linear(self.expand_dim, self.state_dim)

        self.output_proj = nn.Linear(self.expand_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.SiLU()
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.A_log, mean=0, std=0.1)
        with torch.no_grad():
            self.A_log.data = -torch.exp(self.A_log.data)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.zeros_(self.D)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x_proj = self.input_proj(x)
        x, gate = x_proj.chunk(2, dim=-1)
        x = self.activation(x)
        delta = self.delta_proj(x)
        delta = self.dt_proj(delta)
        delta = F.softplus(delta)
        B = self.B_proj(x)
        C = self.C_proj(x)
        A = -torch.exp(self.A_log)
        y = self._selective_scan(x, delta, A, B, C)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * self.activation(gate)
        output = self.output_proj(y)
        return output + residual

    def _selective_scan(self, x, delta, A, B, C):
        batch_size, seq_len, expand_dim = x.shape
        state_dim = A.shape[1]
        h = torch.zeros(batch_size, expand_dim, state_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            delta_t = delta[:, t]
            B_t = B[:, t]
            C_t = C[:, t]
            A_discrete = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            h = A_discrete * h + (delta_t.unsqueeze(-1) * B_t.unsqueeze(1)) * x_t.unsqueeze(-1)
            y_t = torch.sum(C_t.unsqueeze(1) * h, dim=-1)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


# -------------------------- Gated Positional GNN --------------------------

class PositionalGNNLayer(nn.Module):
    """
    Graph layer with optional *edge gates* g_ij in (0,1).
    Aggregation:
        m_i = sum_j  (adj_ij * g_ij) * msg(h_j)
    Gate computation (scaled dot product over transformed messages):
        g_ij = sigmoid( (Q msg(h_i)) @ (K msg(h_j))^T / (sqrt(d) * tau) + b )
    where tau is a temperature (>0; larger -> smoother gates).
    """
    def __init__(self, input_dim, hidden_dim, use_edge_gates: bool = True, gate_temperature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_edge_gates = use_edge_gates
        self.gate_temperature = gate_temperature

        # Message and update nets
        self.message_net = nn.Linear(input_dim, hidden_dim)
        self.update_net = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Gate projections (operate on message vectors)
        if self.use_edge_gates:
            self.gate_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.gate_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.gate_bias = nn.Parameter(torch.tensor(-1.0))  # encourage sparsity initially

        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor, return_gates: bool = False):
        """
        Args:
            node_features: [N, input_dim]
            adj_matrix:    [N, N] binary {0,1} (dense) or float weights
            return_gates:  if True, also returns the (masked) gate matrix

        Returns:
            updated_features: [N, hidden_dim]
            (optional) gates_masked: [N, N] with zeros off-edges and in (0,1) on edges
        """
        N = node_features.shape[0]
        messages = self.message_net(node_features)  # [N, H]

        if self.use_edge_gates:
            # Pairwise scores via scaled dot product of transformed messages
            q = self.gate_q(messages)                # [N, H]
            k = self.gate_k(messages)                # [N, H]
            scores = (q @ k.T) / (math.sqrt(q.shape[-1]) * max(self.gate_temperature, 1e-6))  # [N, N]
            gates = torch.sigmoid(scores + self.gate_bias)                                     # [N, N]
            # Mask to existing edges only
            gates_masked = gates * adj_matrix
            aggregated = gates_masked @ messages      # [N, H]
        else:
            gates_masked = adj_matrix  # conceptually "all ones" on edges
            aggregated = adj_matrix @ messages

        combined = torch.cat([node_features, aggregated], dim=1)
        updated = self.update_net(combined)
        updated = self.activation(updated)
        updated = self.norm(updated)

        if return_gates:
            return updated, gates_masked
        return updated


# ------------------------- Positional Embeddings --------------------------

def create_sincos_positional_embeddings(max_nodes, pos_dim):
    position = torch.arange(max_nodes).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * -(math.log(10000.0) / pos_dim))
    pos_embeddings = torch.zeros(max_nodes, pos_dim)
    pos_embeddings[:, 0::2] = torch.sin(position * div_term)
    pos_embeddings[:, 1::2] = torch.cos(position * div_term)
    return pos_embeddings


# ------------------------------- GraphMamba -------------------------------

class GraphMamba(nn.Module):
    def __init__(self, max_nodes, pos_dim=256, hidden_dim=64, gnn_layers=2,
                 mamba_state_dim=16, dropout=0.1, use_edge_gates: bool = True,
                 gate_temperature: float = 1.0):
        super().__init__()

        self.max_nodes = max_nodes
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        self.use_edge_gates = use_edge_gates
        self.gate_temperature = gate_temperature

        pos_embeddings = create_sincos_positional_embeddings(max_nodes, pos_dim)
        self.register_buffer('pos_embeddings', pos_embeddings)

        # First spatial layer (with gates)
        self.gnn_input = PositionalGNNLayer(pos_dim, hidden_dim,
                                            use_edge_gates=use_edge_gates,
                                            gate_temperature=gate_temperature)
        # Additional spatial layers (no gates by default to reduce cost)
        self.gnn_layers_list = nn.ModuleList([
            PositionalGNNLayer(hidden_dim, hidden_dim, use_edge_gates=False)
            for _ in range(gnn_layers - 1)
        ])

        self.mamba_encoder = MambaBlock(hidden_dim=hidden_dim, state_dim=mamba_state_dim)

        edge_input_dim = hidden_dim * 2
        self.edge_predictor = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def encode_graph(self, adj_matrix, return_gates: bool = False):
        num_nodes = adj_matrix.shape[0]
        node_pos_emb = self.pos_embeddings[:num_nodes]

        x, gates = (None, None)
        if self.use_edge_gates:
            x, gates = self.gnn_input(node_pos_emb, adj_matrix, return_gates=True)
        else:
            x = self.gnn_input(node_pos_emb, adj_matrix, return_gates=False)

        for gnn_layer in self.gnn_layers_list:
            x = gnn_layer(x, adj_matrix)
            x = self.dropout(x)

        if return_gates and self.use_edge_gates:
            return x, gates
        return x, None

    def forward_sequence(self, graph_sequence: List[torch.Tensor], return_gates: bool = False):
        sequence_embeddings = []
        gates_list = [] if return_gates and self.use_edge_gates else None

        for adj in graph_sequence:
            node_emb, gates = self.encode_graph(adj, return_gates=return_gates)
            sequence_embeddings.append(node_emb)
            if gates_list is not None:
                gates_list.append(gates)

        sequence_embeddings = torch.stack(sequence_embeddings, dim=0)  # [T, N, H]
        num_nodes = sequence_embeddings.shape[1]
        sequence_embeddings = sequence_embeddings.transpose(0, 1)      # [N, T, H]

        temporal_embeddings = []
        for node_idx in range(num_nodes):
            node_seq = sequence_embeddings[node_idx].unsqueeze(0)      # [1, T, H]
            node_temporal = self.mamba_encoder(node_seq)               # [1, T, H]
            temporal_embeddings.append(node_temporal.squeeze(0))       # [T, H]

        temporal_embeddings = torch.stack(temporal_embeddings, dim=0).transpose(0, 1)  # [T, N, H]
        if return_gates and self.use_edge_gates:
            return temporal_embeddings, gates_list
        return temporal_embeddings

    def predict_next_edges(self, current_embeddings, edge_pairs):
        src = current_embeddings[edge_pairs[:, 0]]
        dst = current_embeddings[edge_pairs[:, 1]]
        edge_sum = src + dst
        edge_diff = torch.abs(src - dst)
        edge_feat = torch.cat([edge_sum, edge_diff], dim=1)
        return self.edge_predictor(edge_feat).squeeze(-1)

    # ------------------------ Explanation Utilities ------------------------

    @staticmethod
    def sparsity_loss(gates_masked: torch.Tensor) -> torch.Tensor:
        """Mean gate value on existing edges -> encourages fewer active edges."""
        eps = 1e-8
        num_edges = (gates_masked > 0).float().sum()
        if num_edges.item() == 0:
            return gates_masked.sum() * 0.0
        return gates_masked.sum() / (num_edges + eps)

    @staticmethod
    def temporal_tv_loss(gates_t: torch.Tensor, gates_prev: torch.Tensor) -> torch.Tensor:
        """Mean |g_t - g_{t-1}| on edges present at both timesteps."""
        common_mask = ((gates_t > 0) & (gates_prev > 0)).float()
        if common_mask.sum().item() == 0:
            return (gates_t - gates_prev).abs().sum() * 0.0
        diffs = (gates_t - gates_prev).abs() * common_mask
        return diffs.sum() / (common_mask.sum() + 1e-8)

    @staticmethod
    def topk_edges_from_gates(gates_masked: torch.Tensor, k: int = 20) -> List[Tuple[int, int, float]]:
        """Return top-k (i,j,gate_ij) for i<j sorted by gate value."""
        N = gates_masked.shape[0]
        tri_mask = torch.triu(torch.ones(N, N, device=gates_masked.device), diagonal=1)
        vals = gates_masked * tri_mask
        flat_vals = vals.flatten()
        k = min(k, int((N * (N - 1)) // 2))
        if k <= 0:
            return []
        topk = torch.topk(flat_vals, k)
        indices = topk.indices
        scores = topk.values
        ij = [(int(idx // N), int(idx % N)) for idx in indices]
        return [(i, j, float(s)) for (i, j), s in zip(ij, scores.cpu())]


# --------------------------- Data + Evaluation ---------------------------

def create_graph_sequence(g_df, timestamps):
    max_node = max(g_df['u'].max(), g_df['i'].max()) + 1
    graph_sequence = []
    for ts in timestamps:
        edges_up_to_ts = g_df[g_df['ts'] <= ts]
        adj = torch.zeros(max_node, max_node)
        for _, row in edges_up_to_ts.iterrows():
            u, v = int(row['u']), int(row['i'])
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        graph_sequence.append(adj)
    return graph_sequence


def load_triadic_data(data_name):
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    with open(f'./processed/{data_name}/ml_{data_name}_gt_fixed.json', 'r') as f:
        ground_truth = json.load(f)
    ground_truth_by_ts = {}
    for ts_str, edges in ground_truth.items():
        ts = float(ts_str)
        edge_set = set()
        for edge in edges:
            u, v = edge
            edge_set.add((min(u, v), max(u, v)))
        ground_truth_by_ts[ts] = edge_set
    return g_df, ground_truth_by_ts


def evaluate_graphmamba_sequence(model, graph_sequence, ground_truth_by_ts, timestamps,
                                 device, logger, eval_timestamps=None):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        sequence_embeddings = model.forward_sequence(graph_sequence)
        for i in range(len(timestamps) - 1):
            next_ts = timestamps[i + 1]
            if eval_timestamps is not None and next_ts not in eval_timestamps:
                continue
            current_emb = sequence_embeddings[i]
            true_edges = ground_truth_by_ts.get(next_ts, set())
            if len(true_edges) == 0:
                continue
            num_nodes = current_emb.shape[0]
            all_pairs = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)]
            pos = [p for p in all_pairs if p in true_edges]
            neg = [p for p in all_pairs if p not in true_edges]
            if len(pos) == 0:
                continue
            num_neg = min(len(pos), len(neg))
            if num_neg == 0:
                continue
            import torch
            idx = torch.randperm(len(neg))[:num_neg]
            neg_s = [neg[j] for j in idx]
            eval_pairs = pos + neg_s
            labels = [1.0] * len(pos) + [0.0] * len(neg_s)
            edge_pairs = torch.tensor(eval_pairs, device=device)
            preds = model.predict_next_edges(current_emb, edge_pairs)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels)
    if len(all_predictions) == 0:
        return {"accuracy": 0.0, "auc": 0.5, "ap": 0.0}
    predictions_np = np.array(all_predictions)
    labels_np = np.array(all_labels)
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
    accuracy = accuracy_score(labels_np, predictions_np > 0.5)
    auc = roc_auc_score(labels_np, predictions_np)
    ap = average_precision_score(labels_np, predictions_np)
    return {"accuracy": accuracy, "auc": auc, "ap": ap, "num_samples": len(all_predictions)}


def train_graphmamba(data_name='triadic_perfect_long_dense', epochs=100, lr=0.001,
                     hidden_dim=64, pos_dim=256, mamba_state_dim=16,
                     lambda_sparse: float = 1e-4, lambda_tv: float = 1e-3,
                     gate_temperature: float = 1.0):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading {data_name} dataset...")
    g_df, ground_truth_by_ts = load_triadic_data(data_name)
    timestamps = sorted(g_df['ts'].unique())
    logger.info(f"Dataset: {len(timestamps)} timestamps, {len(g_df)} edges")

    graph_sequence = create_graph_sequence(g_df, timestamps)
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
    logger.info(f"Graph sequence: {len(graph_sequence)} graphs, max_nodes: {max_nodes}")

    train_ts = int(len(timestamps) * 0.7)
    val_ts = int(len(timestamps) * 0.15)
    test_ts = len(timestamps) - train_ts - val_ts

    train_timestamps = timestamps[:train_ts]
    val_timestamps = timestamps[train_ts:train_ts + val_ts]
    test_timestamps = timestamps[train_ts + val_ts:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = GraphMamba(max_nodes=max_nodes, pos_dim=pos_dim, hidden_dim=hidden_dim,
                       gnn_layers=2, mamba_state_dim=mamba_state_dim, dropout=0.1,
                       use_edge_gates=True, gate_temperature=gate_temperature).to(device)

    graph_sequence = [adj.to(device) for adj in graph_sequence]
    logger.info(f"Model parameters: pos_dim={pos_dim}, hidden_dim={hidden_dim}, mamba_state_dim={mamba_state_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    best_val_ap = 0.0
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(len(train_timestamps) - 1):
            next_ts = train_timestamps[i + 1]
            train_sequence = graph_sequence[:i + 2]
            # Get embeddings + gates up to current time i
            seq_emb, gates_list = model.forward_sequence(train_sequence, return_gates=True)
            current_emb = seq_emb[i]                  # [N, H]
            current_gates = gates_list[i]             # [N, N], masked on edges at time i
            prev_gates = gates_list[i - 1] if i > 0 else None

            true_edges = ground_truth_by_ts.get(next_ts, set())
            if len(true_edges) == 0:
                continue

            N = current_emb.shape[0]
            all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            pos = [p for p in all_pairs if p in true_edges]
            neg = [p for p in all_pairs if p not in true_edges]
            if len(pos) == 0:
                continue
            num_neg = min(len(pos), len(neg))
            if num_neg == 0:
                continue

            idx = torch.randperm(len(neg))[:num_neg]
            neg_s = [neg[j] for j in idx]

            train_pairs = pos + neg_s
            train_labels = torch.tensor([1.0] * len(pos) + [0.0] * len(neg_s), device=device)
            edge_pairs_tensor = torch.tensor(train_pairs, device=device)

            optimizer.zero_grad()
            preds = model.predict_next_edges(current_emb, edge_pairs_tensor)
            loss_pred = criterion(preds, train_labels)

            # Explanation losses
            loss_sparse = GraphMamba.sparsity_loss(current_gates)
            if prev_gates is not None:
                loss_tv = GraphMamba.temporal_tv_loss(current_gates, prev_gates)
            else:
                loss_tv = current_gates.sum() * 0.0

            loss = loss_pred + lambda_sparse * loss_sparse + lambda_tv * loss_tv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if epoch % 10 == 0 or epoch == epochs - 1:
            val_sequence = graph_sequence[:train_ts + val_ts + 1]
            val_metrics = evaluate_graphmamba_sequence(
                model, val_sequence, ground_truth_by_ts,
                timestamps[:train_ts + val_ts], device, logger,
                eval_timestamps=set(val_timestamps)
            )
            logger.info(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                        f"Val Acc={val_metrics['accuracy']:.4f}, "
                        f"Val AUC={val_metrics['auc']:.4f}, "
                        f"Val AP={val_metrics['ap']:.4f}")
            if val_metrics['ap'] > best_val_ap:
                best_val_ap = val_metrics['ap']
                best_metrics = val_metrics.copy()
                test_metrics = evaluate_graphmamba_sequence(
                    model, graph_sequence, ground_truth_by_ts,
                    timestamps, device, logger,
                    eval_timestamps=set(test_timestamps)
                )
                best_metrics.update({
                    'test_accuracy': test_metrics['accuracy'],
                    'test_auc': test_metrics['auc'],
                    'test_ap': test_metrics['ap']
                })
        else:
            logger.info(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}")

    logger.info("\n" + "=" * 50)
    logger.info("GRAPHMAMBA (Self-Explaining) FINAL RESULTS")
    logger.info("=" * 50)
    logger.info(f"Best Val AP: {best_val_ap:.4f}")
    if best_metrics is not None:
        logger.info(f"Test Accuracy: {best_metrics['test_accuracy']:.4f}")
        logger.info(f"Test AUC: {best_metrics['test_auc']:.4f}")
        logger.info(f"Test AP: {best_metrics['test_ap']:.4f}")
    logger.info("=" * 50)

    return model, best_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Self-Explaining GraphMamba')
    parser.add_argument('--data', type=str, default='triadic_perfect_long_dense', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--pos_dim', type=int, default=256, help='Positional embedding dimension')
    parser.add_argument('--mamba_state_dim', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--lambda_sparse', type=float, default=1e-4, help='Sparsity loss weight')
    parser.add_argument('--lambda_tv', type=float, default=1e-3, help='Temporal smoothness loss weight')
    parser.add_argument('--gate_temperature', type=float, default=1.0, help='Gate temperature (higher = smoother gates)')
    args = parser.parse_args()

    train_graphmamba(
        data_name=args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        pos_dim=args.pos_dim,
        mamba_state_dim=args.mamba_state_dim,
        lambda_sparse=args.lambda_sparse,
        lambda_tv=args.lambda_tv,
        gate_temperature=args.gate_temperature,
    )

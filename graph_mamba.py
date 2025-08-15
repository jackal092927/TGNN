"""
GraphMamba: Graph Neural Network with Mamba State-Space Model for Temporal Graph Modeling

This implementation combines:
1. Sin/Cos Positional Encodings (proven effective for node identity)
2. Graph Convolutional Layers (spatial structure encoding)  
3. Mamba State-Space Model (temporal dynamics modeling)
4. Edge Prediction (triadic closure prediction)

Key advantages over GraphRNN:
- Linear complexity O(n) vs quadratic for attention
- Better long-range dependency modeling
- More efficient parallelizable training
- Selective attention mechanism
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


class MambaBlock(nn.Module):
    """
    Core Mamba State-Space Model Block
    
    Based on the Mamba architecture with selective state-space mechanisms.
    Provides linear-time complexity for sequence modeling with selective attention.
    """
    
    def __init__(self, hidden_dim, state_dim=16, dt_rank=None, expand_factor=2):
        super(MambaBlock, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.dt_rank = dt_rank or max(1, hidden_dim // 16)
        self.expand_dim = int(hidden_dim * expand_factor)
        
        # Input projection
        self.input_proj = nn.Linear(hidden_dim, self.expand_dim * 2)
        
        # Selective mechanism parameters
        self.dt_proj = nn.Linear(self.dt_rank, self.expand_dim)
        self.A_log = nn.Parameter(torch.randn(self.expand_dim, self.state_dim))
        self.D = nn.Parameter(torch.randn(self.expand_dim))
        
        # State space parameters
        self.delta_proj = nn.Linear(self.expand_dim, self.dt_rank)
        self.B_proj = nn.Linear(self.expand_dim, self.state_dim)
        self.C_proj = nn.Linear(self.expand_dim, self.state_dim)
        
        # Output projection
        self.output_proj = nn.Linear(self.expand_dim, hidden_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Activation
        self.activation = nn.SiLU()  # Swish activation used in Mamba
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters following Mamba paper"""
        # Initialize A matrix (should be negative for stability)
        nn.init.normal_(self.A_log, mean=0, std=0.1)
        with torch.no_grad():
            self.A_log.data = -torch.exp(self.A_log.data)
        
        # Initialize other parameters
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.zeros_(self.D)
    
    def forward(self, x):
        """
        Forward pass of Mamba block
        
        Args:
            x: [batch_size, seq_len, hidden_dim]
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Residual connection
        residual = x
        x = self.norm(x)
        
        # Input projection and split
        x_proj = self.input_proj(x)  # [batch, seq_len, expand_dim * 2]
        x, gate = x_proj.chunk(2, dim=-1)  # Each: [batch, seq_len, expand_dim]
        
        # Apply activation
        x = self.activation(x)
        
        # Selective mechanism
        delta = self.delta_proj(x)  # [batch, seq_len, dt_rank]
        delta = self.dt_proj(delta)  # [batch, seq_len, expand_dim]
        delta = F.softplus(delta)  # Ensure positive
        
        B = self.B_proj(x)  # [batch, seq_len, state_dim]
        C = self.C_proj(x)  # [batch, seq_len, state_dim]
        
        # State-space computation
        A = -torch.exp(self.A_log)  # [expand_dim, state_dim]
        
        # Selective scan (simplified version)
        y = self._selective_scan(x, delta, A, B, C)
        
        # Skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gate mechanism
        y = y * self.activation(gate)
        
        # Output projection
        output = self.output_proj(y)
        
        # Residual connection
        return output + residual
    
    def _selective_scan(self, x, delta, A, B, C):
        """
        Simplified selective scan operation
        
        This is a simplified version of the selective scan.
        In practice, this would use more optimized CUDA kernels.
        """
        batch_size, seq_len, expand_dim = x.shape
        state_dim = A.shape[1]
        
        # Initialize hidden state
        h = torch.zeros(batch_size, expand_dim, state_dim, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Get current inputs
            x_t = x[:, t]  # [batch, expand_dim]
            delta_t = delta[:, t]  # [batch, expand_dim]
            B_t = B[:, t]  # [batch, state_dim]
            C_t = C[:, t]  # [batch, state_dim]
            
            # Discretize A matrix
            A_discrete = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # [batch, expand_dim, state_dim]
            
            # Update state: h = A * h + B * x
            h = A_discrete * h + (delta_t.unsqueeze(-1) * B_t.unsqueeze(1)) * x_t.unsqueeze(-1)
            
            # Output: y = C * h
            y_t = torch.sum(C_t.unsqueeze(1) * h, dim=-1)  # [batch, expand_dim]
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # [batch, seq_len, expand_dim]


class PositionalGNNLayer(nn.Module):
    """Graph Neural Network layer that preserves positional information"""
    
    def __init__(self, input_dim, hidden_dim):
        super(PositionalGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Message computation
        self.message_net = nn.Linear(input_dim, hidden_dim)
        
        # Update computation  
        self.update_net = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_features, adj_matrix):
        """
        Args:
            node_features: [num_nodes, input_dim] 
            adj_matrix: [num_nodes, num_nodes]
        Returns:
            updated_features: [num_nodes, hidden_dim]
        """
        # Compute messages
        messages = self.message_net(node_features)  # [num_nodes, hidden_dim]
        
        # GIN-style aggregation: simple sum (no degree normalization)
        aggregated_messages = torch.mm(adj_matrix, messages)  # [num_nodes, hidden_dim]
        
        # Update node features
        combined = torch.cat([node_features, aggregated_messages], dim=1)
        updated_features = self.update_net(combined)
        updated_features = self.activation(updated_features)
        updated_features = self.norm(updated_features)
        
        return updated_features


def create_sincos_positional_embeddings(max_nodes, pos_dim):
    """
    Create sin/cos positional embeddings for nodes
    
    Args:
        max_nodes: Maximum number of nodes
        pos_dim: Positional embedding dimension
        
    Returns:
        pos_embeddings: [max_nodes, pos_dim]
    """
    position = torch.arange(max_nodes).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * 
                        -(math.log(10000.0) / pos_dim))
    
    pos_embeddings = torch.zeros(max_nodes, pos_dim)
    pos_embeddings[:, 0::2] = torch.sin(position * div_term)
    pos_embeddings[:, 1::2] = torch.cos(position * div_term)
    
    return pos_embeddings


class GraphMamba(nn.Module):
    """
    GraphMamba: Combines positional encodings, GNN, and Mamba for temporal graph modeling
    """
    
    def __init__(self, max_nodes, pos_dim=256, hidden_dim=64, gnn_layers=2, 
                 mamba_state_dim=16, dropout=0.1):
        super(GraphMamba, self).__init__()
        
        self.max_nodes = max_nodes
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        
        # Create positional embeddings (fixed, not learnable)
        pos_embeddings = create_sincos_positional_embeddings(max_nodes, pos_dim)
        self.register_buffer('pos_embeddings', pos_embeddings)
        
        # Graph layers
        self.gnn_input = PositionalGNNLayer(pos_dim, hidden_dim)
        self.gnn_layers_list = nn.ModuleList([
            PositionalGNNLayer(hidden_dim, hidden_dim) 
            for _ in range(gnn_layers - 1)
        ])
        
        # Mamba temporal encoder (replaces LSTM)
        self.mamba_encoder = MambaBlock(
            hidden_dim=hidden_dim,
            state_dim=mamba_state_dim
        )
        
        # Edge predictor with symmetric features
        edge_input_dim = hidden_dim * 2  # src_emb + dst_emb, abs(src_emb - dst_emb)
        self.edge_predictor = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Reduced dropout as requested
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def encode_graph(self, adj_matrix):
        """
        Encode a single graph using GNN with positional embeddings
        
        Args:
            adj_matrix: [num_nodes, num_nodes]
            
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        num_nodes = adj_matrix.shape[0]
        
        # Get positional embeddings for current nodes
        node_pos_emb = self.pos_embeddings[:num_nodes]  # [num_nodes, pos_dim]
        
        # Apply GNN layers
        node_embeddings = self.gnn_input(node_pos_emb, adj_matrix)
        
        for gnn_layer in self.gnn_layers_list:
            node_embeddings = gnn_layer(node_embeddings, adj_matrix)
            node_embeddings = self.dropout(node_embeddings)
        
        return node_embeddings
    
    def forward_sequence(self, graph_sequence):
        """
        Forward pass through the entire graph sequence
        
        Args:
            graph_sequence: List of adjacency matrices, one per timestamp
            
        Returns:
            sequence_embeddings: [seq_len, num_nodes, hidden_dim]
        """
        sequence_embeddings = []
        
        for adj_matrix in graph_sequence:
            node_embeddings = self.encode_graph(adj_matrix)
            sequence_embeddings.append(node_embeddings)
        
        # Stack into sequence: [seq_len, num_nodes, hidden_dim]
        sequence_embeddings = torch.stack(sequence_embeddings, dim=0)
        
        # Reshape for Mamba: [num_nodes, seq_len, hidden_dim]
        num_nodes = sequence_embeddings.shape[1]
        sequence_embeddings = sequence_embeddings.transpose(0, 1)
        
        # Apply Mamba to each node's temporal sequence
        temporal_embeddings = []
        for node_idx in range(num_nodes):
            node_sequence = sequence_embeddings[node_idx].unsqueeze(0)  # [1, seq_len, hidden_dim]
            node_temporal = self.mamba_encoder(node_sequence)  # [1, seq_len, hidden_dim]
            temporal_embeddings.append(node_temporal.squeeze(0))  # [seq_len, hidden_dim]
        
        # Stack back: [num_nodes, seq_len, hidden_dim] -> [seq_len, num_nodes, hidden_dim]
        temporal_embeddings = torch.stack(temporal_embeddings, dim=0).transpose(0, 1)
        
        return temporal_embeddings
    
    def predict_next_edges(self, current_embeddings, edge_pairs):
        """
        Predict probabilities for edge pairs given current node embeddings
        
        Args:
            current_embeddings: [num_nodes, hidden_dim]
            edge_pairs: [num_pairs, 2] - pairs of node indices
            
        Returns:
            predictions: [num_pairs] - edge probabilities
        """
        # Get node embeddings for edge pairs
        src_emb = current_embeddings[edge_pairs[:, 0]]  # [num_pairs, hidden_dim]
        dst_emb = current_embeddings[edge_pairs[:, 1]]  # [num_pairs, hidden_dim]
        
        # Symmetric edge features for undirected graphs
        edge_sum = src_emb + dst_emb                    # [num_pairs, hidden_dim] - symmetric sum
        edge_diff = torch.abs(src_emb - dst_emb)        # [num_pairs, hidden_dim] - symmetric difference
        
        edge_features = torch.cat([edge_sum, edge_diff], dim=1)  # [num_pairs, hidden_dim * 2]
        
        # Predict edge probabilities
        predictions = self.edge_predictor(edge_features).squeeze(-1)  # [num_pairs]
        
        return predictions


def create_graph_sequence(g_df, timestamps):
    """Create sequence of adjacency matrices from graph dataframe"""
    max_node = max(g_df['u'].max(), g_df['i'].max()) + 1
    graph_sequence = []
    
    for ts in timestamps:
        # Get edges up to current timestamp
        edges_up_to_ts = g_df[g_df['ts'] <= ts]
        
        # Create adjacency matrix
        adj_matrix = torch.zeros(max_node, max_node)
        
        for _, row in edges_up_to_ts.iterrows():
            u, v = int(row['u']), int(row['i'])
            adj_matrix[u, v] = 1.0
            adj_matrix[v, u] = 1.0  # Undirected graph
        
        graph_sequence.append(adj_matrix)
    
    return graph_sequence


def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    # Load graph data
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    
    # Load ground truth
    with open(f'./processed/{data_name}/ml_{data_name}_gt_fixed.json', 'r') as f:
        ground_truth = json.load(f)
    
    # Convert ground truth to proper format
    ground_truth_by_ts = {}
    for ts_str, edges in ground_truth.items():
        ts = float(ts_str)
        edge_set = set()
        for edge in edges:
            u, v = edge
            edge_set.add((min(u, v), max(u, v)))
        ground_truth_by_ts[ts] = edge_set
    
    return g_df, ground_truth_by_ts


def evaluate_graphmamba_sequence(model, graph_sequence, ground_truth_by_ts, 
                                timestamps, device, logger, eval_timestamps=None):
    """Evaluate GraphMamba on temporal sequence with balanced sampling"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        # Get temporal embeddings for entire sequence
        sequence_embeddings = model.forward_sequence(graph_sequence)
        
        for i in range(len(timestamps) - 1):
            current_ts = timestamps[i]
            next_ts = timestamps[i + 1]
            
            # Skip if not in evaluation timestamps
            if eval_timestamps is not None and next_ts not in eval_timestamps:
                continue
            
            # Get current node embeddings
            current_embeddings = sequence_embeddings[i]  # [num_nodes, hidden_dim]
            
            # Get ground truth edges for next timestamp
            true_edges = ground_truth_by_ts.get(next_ts, set())
            
            if len(true_edges) == 0:
                continue
            
            # Generate all possible node pairs
            num_nodes = current_embeddings.shape[0]
            all_pairs = []
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    all_pairs.append((u, v))
            
            if len(all_pairs) == 0:
                continue
            
            # Separate positive and negative pairs
            positive_pairs = []
            negative_pairs = []
            
            for pair in all_pairs:
                if pair in true_edges:
                    positive_pairs.append(pair)
                else:
                    negative_pairs.append(pair)
            
            if len(positive_pairs) == 0:
                continue
            
            
            # Balanced sampling (1:1 ratio)
            num_negatives = min(len(positive_pairs), len(negative_pairs))
            if num_negatives == 0:
                continue
            
            sampled_negative_indices = torch.randperm(len(negative_pairs))[:num_negatives]
            sampled_negative_pairs = [negative_pairs[idx] for idx in sampled_negative_indices]
            
            # Combine samples
            eval_pairs = positive_pairs + sampled_negative_pairs
            eval_labels = [1.0] * len(positive_pairs) + [0.0] * len(sampled_negative_pairs)
            
            if len(eval_pairs) == 0:
                continue
            
            # Convert to tensors
            edge_pairs_tensor = torch.tensor(eval_pairs, device=device)
            
            # Get predictions
            predictions = model.predict_next_edges(current_embeddings, edge_pairs_tensor)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(eval_labels)
    
    if len(all_predictions) == 0:
        return {"accuracy": 0.0, "auc": 0.5, "ap": 0.0}
    
    # Calculate metrics
    predictions_np = np.array(all_predictions)
    labels_np = np.array(all_labels)
    
    accuracy = accuracy_score(labels_np, predictions_np > 0.5)
    auc = roc_auc_score(labels_np, predictions_np)
    ap = average_precision_score(labels_np, predictions_np)
    
    return {
        "accuracy": accuracy,
        "auc": auc, 
        "ap": ap,
        "num_samples": len(all_predictions)
    }


def train_graphmamba(data_name='triadic_perfect_long_dense', epochs=100, lr=0.001, 
                    hidden_dim=64, pos_dim=256, mamba_state_dim=16):
    """Train GraphMamba model"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading {data_name} dataset...")
    g_df, ground_truth_by_ts = load_triadic_data(data_name)
    
    # Get timestamps and create sequence
    timestamps = sorted(g_df['ts'].unique())
    logger.info(f"Dataset: {len(timestamps)} timestamps, {len(g_df)} edges")
    
    # Create graph sequence
    graph_sequence = create_graph_sequence(g_df, timestamps)
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
    
    logger.info(f"Graph sequence: {len(graph_sequence)} graphs, max_nodes: {max_nodes}")
    
    # Temporal split
    train_ts = int(len(timestamps) * 0.7)
    val_ts = int(len(timestamps) * 0.15)
    test_ts = len(timestamps) - train_ts - val_ts
    
    train_timestamps = timestamps[:train_ts]
    val_timestamps = timestamps[train_ts:train_ts + val_ts]
    test_timestamps = timestamps[train_ts + val_ts:]
    
    logger.info(f"Split: {len(train_timestamps)} train, {len(val_timestamps)} val, {len(test_timestamps)} test")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = GraphMamba(
        max_nodes=max_nodes,
        pos_dim=pos_dim,
        hidden_dim=hidden_dim,
        gnn_layers=2,
        mamba_state_dim=mamba_state_dim,
        dropout=0.1
    ).to(device)
    
    # Move graph sequence to device
    graph_sequence = [adj.to(device) for adj in graph_sequence]
    
    logger.info(f"Model parameters: pos_dim={pos_dim}, hidden_dim={hidden_dim}, mamba_state_dim={mamba_state_dim}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_ap = 0.0
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training on transitions within training period
        for i in range(len(train_timestamps) - 1):
            current_ts = train_timestamps[i]
            next_ts = train_timestamps[i + 1]
            
            # Recompute forward pass for each transition to avoid graph reuse
            train_sequence = graph_sequence[:i + 2]  # Up to next timestamp
            sequence_embeddings = model.forward_sequence(train_sequence)
            
            # Get current embeddings
            current_embeddings = sequence_embeddings[i]
            
            # Get ground truth
            true_edges = ground_truth_by_ts.get(next_ts, set())
            if len(true_edges) == 0:
                continue
            
            # Generate training pairs
            num_nodes = current_embeddings.shape[0]
            all_pairs = []
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    all_pairs.append((u, v))
            
            # Separate positive/negative
            positive_pairs = [pair for pair in all_pairs if pair in true_edges]
            negative_pairs = [pair for pair in all_pairs if pair not in true_edges]
            
            if len(positive_pairs) == 0:
                continue
            
            # Balanced sampling for training (1:1)
            num_negatives = min(len(positive_pairs), len(negative_pairs))
            if num_negatives == 0:
                continue
            
            sampled_negative_indices = torch.randperm(len(negative_pairs))[:num_negatives]
            sampled_negatives = [negative_pairs[idx] for idx in sampled_negative_indices]
            
            # Combine and create batch
            train_pairs = positive_pairs + sampled_negatives
            train_labels = torch.tensor([1.0] * len(positive_pairs) + [0.0] * len(sampled_negatives), 
                                      device=device)
            
            edge_pairs_tensor = torch.tensor(train_pairs, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model.predict_next_edges(current_embeddings, edge_pairs_tensor)
            loss = criterion(predictions, train_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        
        # Validation
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
                
                # Test evaluation
                test_metrics = evaluate_graphmamba_sequence(
                    model, graph_sequence, ground_truth_by_ts, timestamps, device, logger,
                    eval_timestamps=set(test_timestamps)
                )
                best_metrics.update({
                    'test_accuracy': test_metrics['accuracy'],
                    'test_auc': test_metrics['auc'], 
                    'test_ap': test_metrics['ap']
                })
        else:
            logger.info(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}")
    
    # Final results
    logger.info("\n" + "="*50)
    logger.info("GRAPHMAMBA FINAL RESULTS")
    logger.info("="*50)
    logger.info(f"Best Val AP: {best_val_ap:.4f}")
    logger.info(f"Test Accuracy: {best_metrics['test_accuracy']:.4f}")
    logger.info(f"Test AUC: {best_metrics['test_auc']:.4f}")
    logger.info(f"Test AP: {best_metrics['test_ap']:.4f}")
    logger.info("="*50)
    
    return model, best_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GraphMamba for Triadic Closure')
    parser.add_argument('--data', type=str, default='triadic_perfect_long_dense',
                       help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--pos_dim', type=int, default=256,
                       help='Positional embedding dimension')
    parser.add_argument('--mamba_state_dim', type=int, default=16,
                       help='Mamba state dimension')
    
    args = parser.parse_args()
    
    train_graphmamba(
        data_name=args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        pos_dim=args.pos_dim,
        mamba_state_dim=args.mamba_state_dim
    )

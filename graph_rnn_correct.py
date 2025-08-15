"""
Correct GraphRNN Implementation
- Train on ALL edge predictions at each timestamp
- Use teacher forcing during training
- Autoregressive generation during evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import networkx as nx


class GraphStateEncoder(nn.Module):
    """Encode current graph state into a fixed-size representation"""
    
    def __init__(self, node_feat_dim, hidden_dim, max_nodes):
        super(GraphStateEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # Node feature encoder
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        
        # Graph structure encoder (adjacency matrix based)
        self.adj_encoder = nn.Sequential(
            nn.Linear(max_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combine node features and structure
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: [num_nodes, node_feat_dim]
            adjacency_matrix: [max_nodes, max_nodes]
        """
        device = node_features.device
        
        # Encode node features
        node_emb = self.node_encoder(node_features)  # [num_nodes, hidden_dim]
        node_global = torch.mean(node_emb, dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Encode adjacency structure
        adj_degrees = adjacency_matrix.sum(dim=1)  # [max_nodes] - degree vector
        adj_emb = self.adj_encoder(adj_degrees.unsqueeze(0))  # [1, hidden_dim]
        
        # Combine representations
        combined = torch.cat([node_global, adj_emb], dim=1)  # [1, hidden_dim * 2]
        graph_state = self.combiner(combined).squeeze(0)  # [hidden_dim]
        
        return graph_state


class EdgePredictor(nn.Module):
    """Predict edge probability for all node pairs"""
    
    def __init__(self, hidden_dim, max_nodes):
        super(EdgePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # Node embeddings (learnable)
        self.node_embeddings = nn.Embedding(max_nodes, hidden_dim, padding_idx=0)
        
        # Edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # rnn_state + src_emb + dst_emb
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, rnn_hidden, edge_pairs):
        """
        Args:
            rnn_hidden: [hidden_dim] - RNN hidden state
            edge_pairs: [num_pairs, 2] - pairs of (src, dst) node indices
        
        Returns:
            logits: [num_pairs] - edge existence logits
        """
        device = rnn_hidden.device
        num_pairs = edge_pairs.shape[0]
        
        if num_pairs == 0:
            return torch.tensor([], device=device)
        
        # Get node embeddings
        node_embs = self.node_embeddings.weight  # [max_nodes, hidden_dim]
        
        src_nodes = edge_pairs[:, 0]  # [num_pairs]
        dst_nodes = edge_pairs[:, 1]  # [num_pairs]
        
        src_embs = node_embs[src_nodes]  # [num_pairs, hidden_dim]
        dst_embs = node_embs[dst_nodes]  # [num_pairs, hidden_dim]
        
        # Expand RNN hidden state
        rnn_expanded = rnn_hidden.unsqueeze(0).expand(num_pairs, -1)  # [num_pairs, hidden_dim]
        
        # Combine features
        combined = torch.cat([rnn_expanded, src_embs, dst_embs], dim=1)  # [num_pairs, hidden_dim * 3]
        
        # Predict edge probabilities
        logits = self.edge_predictor(combined).squeeze(1)  # [num_pairs]
        
        return logits


class GraphRNN_Correct(nn.Module):
    """
    Correct GraphRNN implementation:
    - Predicts ALL edges at each timestamp
    - Uses teacher forcing during training
    - Autoregressive generation during evaluation
    """
    
    def __init__(self, node_feat_dim, hidden_dim=128, max_nodes=100, rnn_layers=2):
        super(GraphRNN_Correct, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.rnn_layers = rnn_layers
        
        # Components
        self.graph_encoder = GraphStateEncoder(node_feat_dim, hidden_dim, max_nodes)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=rnn_layers, batch_first=True, dropout=0.1)
        self.edge_predictor = EdgePredictor(hidden_dim, max_nodes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def create_adjacency_matrix(self, edge_list, num_nodes):
        """Create adjacency matrix from edge list"""
        device = next(self.parameters()).device
        adj = torch.zeros(self.max_nodes, self.max_nodes, device=device)
        
        if len(edge_list) > 0:
            edge_list = torch.tensor(edge_list, dtype=torch.long, device=device)
            src, dst = edge_list[:, 0], edge_list[:, 1]
            
            # Filter valid edges
            valid_mask = (src < num_nodes) & (dst < num_nodes)
            src, dst = src[valid_mask], dst[valid_mask]
            
            if len(src) > 0:
                adj[src, dst] = 1.0
                adj[dst, src] = 1.0  # Undirected
        
        return adj
    
    def get_candidate_edges(self, current_edges, num_nodes):
        """Get all candidate edges (not currently existing)"""
        existing = set()
        for src, dst in current_edges:
            existing.add((min(src, dst), max(src, dst)))
        
        candidates = []
        for src in range(num_nodes):
            for dst in range(src + 1, num_nodes):
                if (src, dst) not in existing:
                    candidates.append([src, dst])
        
        return candidates
    
    def forward_step(self, rnn_hidden, node_features, current_edges, num_nodes):
        """
        Single forward step: predict edges for next timestamp
        
        Args:
            rnn_hidden: current RNN hidden state
            node_features: [num_nodes, node_feat_dim]
            current_edges: list of current edges [[src, dst], ...]
            num_nodes: number of nodes in graph
            
        Returns:
            edge_logits: predictions for all candidate edges
            candidate_edges: corresponding edge pairs
            new_rnn_hidden: updated RNN hidden state
        """
        device = next(self.parameters()).device
        
        # Create adjacency matrix
        adj_matrix = self.create_adjacency_matrix(current_edges, num_nodes)
        
        # Encode current graph state
        graph_state = self.graph_encoder(node_features, adj_matrix)  # [hidden_dim]
        
        # Update RNN
        graph_state_input = graph_state.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        rnn_output, new_rnn_hidden = self.rnn(graph_state_input, rnn_hidden)
        rnn_state = rnn_output.squeeze(0).squeeze(0)  # [hidden_dim]
        
        # Get candidate edges
        candidate_edges = self.get_candidate_edges(current_edges, num_nodes)
        
        if len(candidate_edges) == 0:
            return torch.tensor([], device=device), [], new_rnn_hidden
        
        # Predict edge probabilities
        edge_pairs = torch.tensor(candidate_edges, dtype=torch.long, device=device)
        edge_logits = self.edge_predictor(rnn_state, edge_pairs)
        
        return edge_logits, candidate_edges, new_rnn_hidden
    
    def forward_sequence(self, node_features, edge_sequence, num_nodes, teacher_forcing=True):
        """
        Process entire sequence of graphs
        
        Args:
            node_features: [num_nodes, node_feat_dim]
            edge_sequence: list of edge lists for each timestamp
            num_nodes: number of nodes
            teacher_forcing: use ground truth edges during training
            
        Returns:
            all_predictions: list of (edge_logits, candidate_edges) for each timestamp
        """
        device = next(self.parameters()).device
        
        # Initialize RNN hidden state
        h_0 = torch.zeros(self.rnn_layers, 1, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.rnn_layers, 1, self.hidden_dim, device=device)
        rnn_hidden = (h_0, c_0)
        
        all_predictions = []
        current_edges = []  # Start with empty graph
        
        for t in range(len(edge_sequence) - 1):  # Predict t+1 from t
            # Predict next edges
            edge_logits, candidate_edges, rnn_hidden = self.forward_step(
                rnn_hidden, node_features, current_edges, num_nodes
            )
            
            all_predictions.append((edge_logits, candidate_edges))
            
            if teacher_forcing:
                # Use ground truth edges for next step
                current_edges = edge_sequence[t + 1].copy()
            else:
                # Use predicted edges for next step
                if len(edge_logits) > 0:
                    probs = torch.sigmoid(edge_logits)
                    predicted_mask = probs > 0.5
                    
                    predicted_edges = []
                    for i, is_edge in enumerate(predicted_mask):
                        if is_edge:
                            predicted_edges.append(candidate_edges[i])
                    
                    current_edges = current_edges + predicted_edges
        
        return all_predictions
    
    def generate_sequence(self, node_features, initial_edges, num_steps, num_nodes):
        """
        Generate sequence autoregressively (for evaluation)
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Initialize RNN hidden state
        h_0 = torch.zeros(self.rnn_layers, 1, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.rnn_layers, 1, self.hidden_dim, device=device)
        rnn_hidden = (h_0, c_0)
        
        generated_sequence = [initial_edges.copy()]
        current_edges = initial_edges.copy()
        
        with torch.no_grad():
            for t in range(num_steps):
                # Predict next edges
                edge_logits, candidate_edges, rnn_hidden = self.forward_step(
                    rnn_hidden, node_features, current_edges, num_nodes
                )
                
                # Select edges based on predictions
                if len(edge_logits) > 0:
                    probs = torch.sigmoid(edge_logits)
                    
                    # Select top-k edges or use threshold
                    top_k = min(5, len(probs))  # Add at most 5 edges per step
                    top_indices = torch.topk(probs, top_k)[1]
                    
                    new_edges = []
                    for idx in top_indices:
                        if probs[idx] > 0.5:  # Threshold
                            new_edges.append(candidate_edges[idx.item()])
                    
                    current_edges.extend(new_edges)
                
                generated_sequence.append(current_edges.copy())
        
        return generated_sequence


def create_edge_sequence_from_data(g_df, max_timestamp):
    """Convert dataframe to sequence of edge lists"""
    edge_sequence = []
    
    for t in range(max_timestamp + 1):
        edges_at_t = g_df[g_df.ts <= t]  # Cumulative edges
        edge_list = []
        
        for _, row in edges_at_t.iterrows():
            edge_list.append([row.u, row.i])
        
        edge_sequence.append(edge_list)
    
    return edge_sequence


if __name__ == "__main__":
    # Test the correct implementation
    print("Testing Correct GraphRNN...")
    
    node_feat_dim = 100
    max_nodes = 50
    hidden_dim = 64
    
    model = GraphRNN_Correct(node_feat_dim, hidden_dim, max_nodes)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy data
    dummy_node_feat = torch.randn(30, node_feat_dim)
    dummy_edge_sequence = [
        [],  # t=0: empty
        [[0, 1], [1, 2]],  # t=1: 2 edges
        [[0, 1], [1, 2], [2, 0]],  # t=2: 3 edges
    ]
    
    predictions = model.forward_sequence(dummy_node_feat, dummy_edge_sequence, 30, teacher_forcing=True)
    print(f"Generated {len(predictions)} predictions")
    
    for i, (logits, candidates) in enumerate(predictions):
        print(f"Step {i}: {len(logits)} candidate edges, {len(candidates)} candidates")
    
    print("Correct GraphRNN test completed!")

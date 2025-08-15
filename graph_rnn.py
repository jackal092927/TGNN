"""
GraphRNN: A simple RNN-based model for temporal graph prediction
Specifically designed for triadic closure prediction on dynamic graphs.

Key Ideas:
1. Use RNN to model temporal dependencies in graph evolution
2. Simple architecture focused on learning triadic closure patterns
3. Direct edge prediction based on graph state history
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import networkx as nx


class GraphStateEncoder(nn.Module):
    """Encode current graph state into a fixed-size representation"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, max_nodes):
        super(GraphStateEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # Node and edge feature encoders
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        
        # Graph structure encoder (adjacency matrix based)
        self.adj_encoder = nn.Sequential(
            nn.Linear(max_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combine all graph information
        self.graph_combiner = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_features, edge_list, edge_features, num_nodes):
        """
        Args:
            node_features: [num_nodes, node_feat_dim]
            edge_list: [num_edges, 2] 
            edge_features: [num_edges, edge_feat_dim]
            num_nodes: actual number of nodes
        """
        device = node_features.device
        batch_size = 1  # Single graph
        
        # Encode node features
        node_emb = self.node_encoder(node_features)  # [num_nodes, hidden_dim]
        node_global = torch.mean(node_emb, dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Encode edge features
        if edge_features.shape[0] > 0:
            edge_emb = self.edge_encoder(edge_features)  # [num_edges, hidden_dim]
            edge_global = torch.mean(edge_emb, dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            edge_global = torch.zeros(1, self.hidden_dim).to(device)
        
        # Create adjacency representation
        adj_matrix = torch.zeros(self.max_nodes, self.max_nodes).to(device)
        if edge_list.shape[0] > 0:
            src, dst = edge_list[:, 0], edge_list[:, 1]
            # Only use valid edges (within num_nodes)
            valid_mask = (src < num_nodes) & (dst < num_nodes)
            src, dst = src[valid_mask], dst[valid_mask]
            if len(src) > 0:
                adj_matrix[src, dst] = 1.0
                adj_matrix[dst, src] = 1.0  # Undirected
        
        # Encode adjacency structure
        adj_flat = adj_matrix.sum(dim=1)  # [max_nodes] - degree vector
        adj_emb = self.adj_encoder(adj_flat.unsqueeze(0))  # [1, hidden_dim]
        
        # Combine all representations
        graph_repr = torch.cat([node_global, edge_global, adj_emb], dim=1)  # [1, hidden_dim * 3]
        graph_state = self.graph_combiner(graph_repr)  # [1, hidden_dim]
        
        return graph_state.squeeze(0)  # [hidden_dim]


class TriadicPredictor(nn.Module):
    """Predict next edge based on current graph state and RNN hidden state"""
    
    def __init__(self, hidden_dim, max_nodes):
        super(TriadicPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # Node pair scorer
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge existence predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # rnn_state + src_emb + dst_emb
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary prediction
        )
        
    def forward(self, rnn_hidden, node_embeddings, src_node, dst_node):
        """
        Args:
            rnn_hidden: [hidden_dim] - RNN hidden state
            node_embeddings: [max_nodes, hidden_dim] - node representations
            src_node: scalar - source node id
            dst_node: scalar - destination node id
        """
        # Get node embeddings
        src_emb = node_embeddings[src_node]  # [hidden_dim]
        dst_emb = node_embeddings[dst_node]  # [hidden_dim]
        
        # Combine RNN state with node pair
        combined = torch.cat([rnn_hidden, src_emb, dst_emb], dim=0)  # [hidden_dim * 3]
        
        # Predict edge probability
        logit = self.edge_predictor(combined)  # [1]
        prob = torch.sigmoid(logit)
        
        return prob.squeeze(0), logit.squeeze(0)  # scalar


class GraphRNN(nn.Module):
    """
    GraphRNN: Simple RNN-based model for temporal graph prediction
    
    Architecture:
    1. Encode each graph state in the sequence
    2. Use RNN to model temporal dependencies  
    3. Predict next edges based on RNN hidden state
    """
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=128, max_nodes=100, rnn_layers=2):
        super(GraphRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.rnn_layers = rnn_layers
        
        # Components
        self.graph_encoder = GraphStateEncoder(node_feat_dim, edge_feat_dim, hidden_dim, max_nodes)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=rnn_layers, batch_first=True, dropout=0.1)
        self.triadic_predictor = TriadicPredictor(hidden_dim, max_nodes)
        
        # Node embeddings (learnable)
        self.node_embeddings = nn.Embedding(max_nodes, hidden_dim, padding_idx=0)
        
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
    
    def encode_sequence(self, sequence_data):
        """
        Encode a sequence of graph states
        
        Args:
            sequence_data: list of (node_features, edge_list, edge_features, timestamp)
        Returns:
            graph_states: [seq_len, hidden_dim]
        """
        graph_states = []
        
        for node_feat, edge_list, edge_feat, ts in sequence_data:
            # Convert to tensors if needed
            if isinstance(node_feat, np.ndarray):
                node_feat = torch.tensor(node_feat, dtype=torch.float32)
            if isinstance(edge_list, np.ndarray):
                edge_list = torch.tensor(edge_list, dtype=torch.long)
            if isinstance(edge_feat, np.ndarray):
                edge_feat = torch.tensor(edge_feat, dtype=torch.float32)
            
            # Move to same device as model
            device = next(self.parameters()).device
            node_feat = node_feat.to(device)
            edge_list = edge_list.to(device)
            edge_feat = edge_feat.to(device)
            
            # Encode graph state
            num_nodes = node_feat.shape[0]
            graph_state = self.graph_encoder(node_feat, edge_list, edge_feat, num_nodes)
            graph_states.append(graph_state)
        
        if len(graph_states) == 0:
            return None
            
        # Stack into tensor
        graph_states_tensor = torch.stack(graph_states)  # [seq_len, hidden_dim]
        return graph_states_tensor
    
    def forward(self, sequence_data, target_src=None, target_dst=None):
        """
        Forward pass
        
        Args:
            sequence_data: list of (node_features, edge_list, edge_features, timestamp)
            target_src: target source node (for training)
            target_dst: target destination node (for training)
        """
        device = next(self.parameters()).device
        
        # Encode sequence
        graph_states = self.encode_sequence(sequence_data)
        if graph_states is None:
            return None, None
        
        # Pass through RNN
        graph_states = graph_states.unsqueeze(0)  # [1, seq_len, hidden_dim]
        rnn_out, (hidden, cell) = self.rnn(graph_states)  # [1, seq_len, hidden_dim]
        
        # Use final hidden state for prediction
        final_hidden = rnn_out[0, -1, :]  # [hidden_dim]
        
        if target_src is not None and target_dst is not None:
            # Training mode: predict specific edge
            node_embs = self.node_embeddings.weight  # [max_nodes, hidden_dim]
            prob, logit = self.triadic_predictor(final_hidden, node_embs, target_src, target_dst)
            return prob, logit
        else:
            # Inference mode: return hidden state for candidate scoring
            return final_hidden, None
    
    def predict_edge_candidates(self, sequence_data, candidate_pairs):
        """
        Score multiple candidate edges
        
        Args:
            sequence_data: list of graph states
            candidate_pairs: list of (src, dst) tuples
        Returns:
            scores: list of probabilities
        """
        device = next(self.parameters()).device
        
        # Get RNN hidden state
        final_hidden, _ = self.forward(sequence_data)
        if final_hidden is None:
            return [0.5] * len(candidate_pairs)
        
        node_embs = self.node_embeddings.weight
        scores = []
        
        for src, dst in candidate_pairs:
            prob, _ = self.triadic_predictor(final_hidden, node_embs, src, dst)
            scores.append(prob.item())
        
        return scores


def create_graph_sequence_from_data(g_df, max_timestamp, n_feat, e_feat):
    """
    Convert dataframe to sequence of graph states
    
    Args:
        g_df: DataFrame with columns [u, i, ts, label, idx]
        max_timestamp: maximum timestamp to include
        n_feat: node features [num_nodes, feat_dim]
        e_feat: edge features [num_edges, feat_dim]
    
    Returns:
        sequence: list of (node_features, edge_list, edge_features, timestamp)
    """
    sequence = []
    
    for ts in range(int(max_timestamp) + 1):
        # Get edges up to current timestamp
        edges_up_to_ts = g_df[g_df.ts <= ts]
        
        if len(edges_up_to_ts) == 0:
            continue
            
        # Extract edge information
        src_nodes = edges_up_to_ts.u.values
        dst_nodes = edges_up_to_ts.i.values
        edge_indices = edges_up_to_ts.idx.values
        
        # Create edge list
        edge_list = np.column_stack([src_nodes, dst_nodes])
        
        # Get edge features for these edges
        edge_features = e_feat[edge_indices]
        
        # Node features (all nodes)
        node_features = n_feat
        
        sequence.append((node_features, edge_list, edge_features, ts))
    
    return sequence


if __name__ == "__main__":
    # Test the model
    print("Testing GraphRNN...")
    
    # Create dummy data
    node_feat_dim, edge_feat_dim = 100, 100
    max_nodes = 50
    hidden_dim = 64
    
    model = GraphRNN(node_feat_dim, edge_feat_dim, hidden_dim, max_nodes)
    print(f"GraphRNN created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_node_feat = torch.randn(30, node_feat_dim)
    dummy_edge_list = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
    dummy_edge_feat = torch.randn(3, edge_feat_dim)
    
    sequence_data = [(dummy_node_feat, dummy_edge_list, dummy_edge_feat, 0)]
    
    prob, logit = model(sequence_data, target_src=0, target_dst=5)
    print(f"Test prediction: prob={prob:.4f}, logit={logit:.4f}")
    
    print("GraphRNN test completed!")

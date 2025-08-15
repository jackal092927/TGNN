"""
GIN-based Edge Convolution GNN for Triadic Closure Prediction

Uses Graph Isomorphism Network (GIN) as backbone for edge convolution.
GIN has maximum expressive power among GNNs and can distinguish different graph structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import networkx as nx


class EdgeGraph:
    """
    Convert node graph to edge graph (line graph) where:
    - Each edge becomes a node
    - Two edge-nodes are connected if they share a vertex in original graph
    """
    
    def __init__(self):
        self.edge_to_idx = {}  # {(u,v): edge_idx}
        self.idx_to_edge = {}  # {edge_idx: (u,v)}
        self.edge_adjacency = []  # Adjacency matrix for edge graph
        
    def build_from_edges(self, edge_list):
        """
        Build edge graph from list of edges
        
        Args:
            edge_list: list of (src, dst) tuples
        """
        # Reset
        self.edge_to_idx = {}
        self.idx_to_edge = {}
        
        # Create edge-to-index mapping
        for i, (u, v) in enumerate(edge_list):
            edge_key = (min(u, v), max(u, v))  # Normalize edge direction
            if edge_key not in self.edge_to_idx:
                idx = len(self.edge_to_idx)
                self.edge_to_idx[edge_key] = idx
                self.idx_to_edge[idx] = edge_key
        
        num_edges = len(self.edge_to_idx)
        
        # Build edge adjacency matrix
        self.edge_adjacency = torch.zeros(num_edges, num_edges, dtype=torch.float32)
        
        # Group edges by vertices
        vertex_to_edges = defaultdict(list)
        for edge_key, edge_idx in self.edge_to_idx.items():
            u, v = edge_key
            vertex_to_edges[u].append(edge_idx)
            vertex_to_edges[v].append(edge_idx)
        
        # Connect edges that share vertices
        for vertex, edge_indices in vertex_to_edges.items():
            for i, edge_i in enumerate(edge_indices):
                for j, edge_j in enumerate(edge_indices):
                    if i != j:  # Don't connect edge to itself
                        self.edge_adjacency[edge_i, edge_j] = 1.0
        
        return num_edges
    
    def get_triadic_candidates(self, edge_list, all_nodes):
        """
        Find all potential triadic closures given current edges
        
        Args:
            edge_list: current edges [(u,v), ...]
            all_nodes: set of all nodes in graph
            
        Returns:
            candidates: list of (src, dst, shared_neighbor) for potential closures
        """
        # Build adjacency for current edges
        adj = defaultdict(set)
        existing_edges = set()
        
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            existing_edges.add((min(u, v), max(u, v)))
        
        candidates = []
        
        # Find all potential triadic closures
        for node in all_nodes:
            neighbors = list(adj[node])
            
            # For each pair of neighbors, they could form a triadic closure
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    u, v = neighbors[i], neighbors[j]
                    edge_key = (min(u, v), max(u, v))
                    
                    # If u-v edge doesn't exist, it's a candidate
                    if edge_key not in existing_edges:
                        candidates.append((u, v, node))  # shared_neighbor = node
        
        return candidates


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network (GIN) Layer
    
    GIN has maximum expressive power and can distinguish different graph structures.
    Perfect for triadic closure pattern recognition.
    """
    
    def __init__(self, in_dim, out_dim, eps=0.0):
        super(GINLayer, self).__init__()
        self.eps = eps
        
        # MLP for GIN (using LayerNorm instead of BatchNorm for stability)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        
    def forward(self, edge_features, edge_adjacency):
        """
        Args:
            edge_features: [num_edges, in_dim]
            edge_adjacency: [num_edges, num_edges] - adjacency matrix for edge graph
        """
        num_edges = edge_features.shape[0]
        device = edge_features.device
        
        if num_edges == 0:
            return edge_features
        
        # Move adjacency to same device
        edge_adjacency = edge_adjacency.to(device)
        
        # GIN aggregation: (1 + eps) * x + sum(neighbors)
        # Aggregate neighbor features
        neighbor_sum = torch.mm(edge_adjacency, edge_features)  # [num_edges, in_dim]
        
        # GIN update: (1 + eps) * self + aggregated_neighbors
        updated = (1 + self.eps) * edge_features + neighbor_sum
        
        # Apply MLP
        output = self.mlp(updated)
        
        return output


class GINEdgeEncoder(nn.Module):
    """
    Multi-layer GIN encoder for edge representations
    """
    
    def __init__(self, edge_feat_dim, hidden_dim, num_layers=3, dropout=0.1):
        super(GINEdgeEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(edge_feat_dim, hidden_dim)
        
        # GIN layers
        self.gin_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gin_layers.append(GINLayer(hidden_dim, hidden_dim))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, edge_features, edge_adjacency):
        """
        Args:
            edge_features: [num_edges, edge_feat_dim]
            edge_adjacency: [num_edges, num_edges]
        """
        # Input projection
        x = self.input_proj(edge_features)  # [num_edges, hidden_dim]
        x = F.relu(x)
        
        # Apply GIN layers with residual connections
        for gin_layer in self.gin_layers:
            residual = x
            x = gin_layer(x, edge_adjacency)
            x = self.dropout_layer(x)
            
            # Residual connection (if dimensions match)
            if residual.shape == x.shape:
                x = x + residual
        
        return x


class TriadicPredictor(nn.Module):
    """
    Predict triadic closure probability based on GIN edge representations
    
    Uses attention mechanism to focus on relevant edge relationships
    """
    
    def __init__(self, edge_dim, hidden_dim):
        super(TriadicPredictor, self).__init__()
        
        # Attention mechanism for parent edges
        self.attention = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Final prediction MLP (using LayerNorm for stability)
        self.predictor = nn.Sequential(
            nn.Linear(edge_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, parent_edge1_feat, parent_edge2_feat):
        """
        Predict triadic closure based on two parent edges with attention
        
        Args:
            parent_edge1_feat: [edge_dim] - first parent edge
            parent_edge2_feat: [edge_dim] - second parent edge
        """
        # Stack parent edges
        parent_edges = torch.stack([parent_edge1_feat, parent_edge2_feat])  # [2, edge_dim]
        
        # Compute attention weights
        attn_weights = self.attention(parent_edges)  # [2, 1]
        attn_weights = F.softmax(attn_weights, dim=0)  # [2, 1]
        
        # Weighted combination of parent edges
        weighted_parent1 = attn_weights[0] * parent_edge1_feat
        weighted_parent2 = attn_weights[1] * parent_edge2_feat
        
        # Combine features
        combined = torch.cat([weighted_parent1, weighted_parent2])  # [edge_dim * 2]
        
        # Predict
        logit = self.predictor(combined.unsqueeze(0)).squeeze(0)  # Add/remove batch dim
        prob = torch.sigmoid(logit)
        
        return prob.squeeze(), logit.squeeze()


class GINEdgeConvGNN(nn.Module):
    """
    GIN-based Edge Convolution GNN for Triadic Closure Prediction
    
    Architecture:
    1. Convert node graph to edge graph (line graph)
    2. Apply GIN layers on edge graph
    3. Use attention-based triadic predictor
    """
    
    def __init__(self, edge_feat_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super(GINEdgeConvGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GIN edge encoder
        self.edge_encoder = GINEdgeEncoder(edge_feat_dim, hidden_dim, num_layers, dropout)
        
        # Triadic predictor with attention
        self.triadic_predictor = TriadicPredictor(hidden_dim, hidden_dim)
        
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
    
    def forward(self, edge_list, edge_features, triadic_candidates):
        """
        Forward pass
        
        Args:
            edge_list: list of (u, v) current edges
            edge_features: [num_edges, edge_feat_dim] - features for current edges
            triadic_candidates: list of (src, dst, shared_neighbor) potential closures
            
        Returns:
            predictions: [num_candidates] - probabilities for each candidate
            logits: [num_candidates] - logits for each candidate
        """
        device = next(self.parameters()).device
        
        if len(edge_list) == 0:
            # No edges, return empty predictions
            return torch.tensor([], device=device), torch.tensor([], device=device)
        
        # Build edge graph
        edge_graph = EdgeGraph()
        num_edges = edge_graph.build_from_edges(edge_list)
        
        if num_edges != edge_features.shape[0]:
            # Mismatch in edge count, handle gracefully
            min_edges = min(num_edges, edge_features.shape[0])
            edge_features = edge_features[:min_edges]
            num_edges = min_edges
        
        if num_edges == 0:
            return torch.tensor([], device=device), torch.tensor([], device=device)
        
        # Encode edge features with GIN
        edge_features = edge_features.to(device)
        edge_emb = self.edge_encoder(edge_features, edge_graph.edge_adjacency)  # [num_edges, hidden_dim]
        
        # Predict triadic closures
        predictions = []
        logits = []
        
        for src, dst, shared_neighbor in triadic_candidates:
            # Find parent edges: shared_neighbor-src and shared_neighbor-dst
            parent_edge1 = (min(shared_neighbor, src), max(shared_neighbor, src))
            parent_edge2 = (min(shared_neighbor, dst), max(shared_neighbor, dst))
            
            # Get edge indices
            parent_idx1 = edge_graph.edge_to_idx.get(parent_edge1, None)
            parent_idx2 = edge_graph.edge_to_idx.get(parent_edge2, None)
            
            if parent_idx1 is not None and parent_idx2 is not None:
                # Both parent edges exist, predict triadic closure
                parent_feat1 = edge_emb[parent_idx1]  # [hidden_dim]
                parent_feat2 = edge_emb[parent_idx2]  # [hidden_dim]
                
                prob, logit = self.triadic_predictor(parent_feat1, parent_feat2)
                predictions.append(prob)
                logits.append(logit)
            else:
                # Parent edges don't exist, very low probability
                predictions.append(torch.tensor(0.05, device=device))
                logits.append(torch.tensor(-3.0, device=device))  # logit for prob=0.05
        
        if len(predictions) == 0:
            return torch.tensor([], device=device), torch.tensor([], device=device)
        
        predictions = torch.stack(predictions)
        logits = torch.stack(logits)
        
        return predictions, logits


def create_edge_features_from_indices(edge_indices, all_edge_features):
    """
    Extract edge features for given edge indices
    
    Args:
        edge_indices: list of edge indices
        all_edge_features: [num_total_edges, feat_dim] - all edge features
    
    Returns:
        edge_features: [num_edges, feat_dim] - features for selected edges
    """
    if len(edge_indices) == 0:
        return torch.zeros(0, all_edge_features.shape[1])
    
    edge_features = all_edge_features[edge_indices]
    return edge_features


if __name__ == "__main__":
    # Test the model
    print("Testing GIN-based Edge Convolution GNN...")
    
    edge_feat_dim = 100
    hidden_dim = 64
    
    model = GINEdgeConvGNN(edge_feat_dim, hidden_dim, num_layers=3)
    print(f"GINEdgeConvGNN created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy data
    edge_list = [(0, 1), (1, 2), (2, 3), (0, 3)]  # A square
    edge_features = torch.randn(4, edge_feat_dim)
    triadic_candidates = [(0, 2, 1), (1, 3, 2)]  # Two potential closures
    
    predictions, logits = model(edge_list, edge_features, triadic_candidates)
    print(f"Test predictions: {predictions}")
    print(f"Test logits: {logits}")
    
    print("GIN-based EdgeConvGNN test completed!")

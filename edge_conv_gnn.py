"""
Edge Convolution GNN for Triadic Closure Prediction

Key Ideas:
1. Model edges as nodes in a higher-order graph (line graph)
2. Use edge-edge relationships to predict triadic closures
3. Focus on local graph topology rather than long temporal sequences
4. Direct triadic pattern: if edges A-B and B-C exist → A-C likely
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
        self.edge_adjacency = []  # Adjacency list for edge graph
        
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
        
        # Build edge adjacency (edges are adjacent if they share a vertex)
        self.edge_adjacency = [[] for _ in range(num_edges)]
        
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
                        if edge_j not in self.edge_adjacency[edge_i]:
                            self.edge_adjacency[edge_i].append(edge_j)
        
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


class EdgeConvLayer(nn.Module):
    """
    Edge Convolution Layer
    Aggregates information from neighboring edges
    """
    
    def __init__(self, in_dim, out_dim):
        super(EdgeConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Message computation
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),  # self + neighbor
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Update computation
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),  # self + aggregated
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, edge_features, edge_adjacency):
        """
        Args:
            edge_features: [num_edges, in_dim]
            edge_adjacency: list of lists - adjacency for edge graph
        """
        num_edges = edge_features.shape[0]
        device = edge_features.device
        
        # Compute messages
        messages = []
        
        for edge_i in range(num_edges):
            edge_i_feat = edge_features[edge_i]  # [in_dim]
            neighbors = edge_adjacency[edge_i]
            
            if len(neighbors) == 0:
                # No neighbors, use self-message
                message = self.message_net(torch.cat([edge_i_feat, edge_i_feat]))
            else:
                # Aggregate messages from neighbors
                neighbor_messages = []
                
                for edge_j in neighbors:
                    edge_j_feat = edge_features[edge_j]  # [in_dim]
                    combined = torch.cat([edge_i_feat, edge_j_feat])  # [in_dim * 2]
                    message = self.message_net(combined)  # [out_dim]
                    neighbor_messages.append(message)
                
                # Average neighbor messages
                message = torch.stack(neighbor_messages).mean(dim=0)  # [out_dim]
            
            messages.append(message)
        
        messages = torch.stack(messages)  # [num_edges, out_dim]
        
        # Update edge features
        updated_features = []
        
        for edge_i in range(num_edges):
            edge_i_feat = edge_features[edge_i]  # [in_dim]
            message_i = messages[edge_i]  # [out_dim]
            
            combined = torch.cat([edge_i_feat, message_i])  # [in_dim + out_dim]
            updated_feat = self.update_net(combined)  # [out_dim]
            updated_features.append(updated_feat)
        
        updated_features = torch.stack(updated_features)  # [num_edges, out_dim]
        
        return updated_features


class TriadicPredictor(nn.Module):
    """
    Predict triadic closure probability based on edge representations
    """
    
    def __init__(self, edge_dim, hidden_dim):
        super(TriadicPredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(edge_dim * 2, hidden_dim),  # Two parent edges
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary prediction
        )
        
    def forward(self, parent_edge1_feat, parent_edge2_feat):
        """
        Predict triadic closure based on two parent edges
        
        Args:
            parent_edge1_feat: [edge_dim] - first parent edge
            parent_edge2_feat: [edge_dim] - second parent edge
        """
        combined = torch.cat([parent_edge1_feat, parent_edge2_feat])  # [edge_dim * 2]
        logit = self.predictor(combined)  # [1]
        prob = torch.sigmoid(logit)
        
        return prob.squeeze(0), logit.squeeze(0)


class EdgeConvGNN(nn.Module):
    """
    Edge Convolution GNN for Triadic Closure Prediction
    
    Architecture:
    1. Convert node graph to edge graph
    2. Apply edge convolution layers
    3. Predict triadic closures based on edge pairs
    """
    
    def __init__(self, edge_feat_dim, hidden_dim=128, num_layers=3):
        super(EdgeConvGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(EdgeConvLayer(hidden_dim, hidden_dim))
        
        # Triadic predictor
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
        
        # Encode edge features
        edge_features = edge_features.to(device)
        edge_emb = self.edge_encoder(edge_features)  # [num_edges, hidden_dim]
        
        # Apply edge convolution layers
        for conv_layer in self.conv_layers:
            edge_emb = conv_layer(edge_emb, edge_graph.edge_adjacency)
            edge_emb = F.relu(edge_emb)  # Apply activation after each layer
        
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
                # Parent edges don't exist, low probability
                predictions.append(torch.tensor(0.1, device=device))
                logits.append(torch.tensor(-2.0, device=device))  # logit for prob=0.1
        
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
    print("Testing Edge Convolution GNN...")
    
    edge_feat_dim = 100
    hidden_dim = 64
    
    model = EdgeConvGNN(edge_feat_dim, hidden_dim, num_layers=3)
    print(f"EdgeConvGNN created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy data
    edge_list = [(0, 1), (1, 2), (2, 3), (0, 3)]  # A square
    edge_features = torch.randn(4, edge_feat_dim)
    triadic_candidates = [(0, 2, 1), (1, 3, 2)]  # Two potential closures
    
    predictions, logits = model(edge_list, edge_features, triadic_candidates)
    print(f"Test predictions: {predictions}")
    print(f"Test logits: {logits}")
    
    print("EdgeConvGNN test completed!")

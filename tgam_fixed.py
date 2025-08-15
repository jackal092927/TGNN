"""
Corrected Temporal Graph Autoregressive Model (TGAM) for Link Prediction

This version properly aligns with the link prediction task like TGIB:
- Given edge history up to time k
- Predict probability that edge (u,v) exists at time k+1
- Binary classification task (real vs fake edges)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GraphStateEncoder(nn.Module):
    """Encodes current graph state using message passing"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers=2):
        super(GraphStateEncoder, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embedding layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge embedding layers
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # node + edge + neighbor
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Aggregation
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_features, edge_list, edge_features, max_nodes):
        """
        Args:
            node_features: [num_nodes, node_feat_dim]
            edge_list: [num_edges, 2] - source and target nodes
            edge_features: [num_edges, edge_feat_dim]
            max_nodes: maximum number of nodes to consider
        """
        device = node_features.device
        
        # Initialize node embeddings for all possible nodes
        if len(node_features) < max_nodes:
            # Pad with zeros for unseen nodes
            padding = torch.zeros(max_nodes - len(node_features), self.node_feat_dim).to(device)
            padded_node_features = torch.cat([node_features, padding], dim=0)
        else:
            padded_node_features = node_features[:max_nodes]
            
        node_emb = self.node_encoder(padded_node_features)  # [max_nodes, hidden_dim]
        
        if len(edge_list) == 0:
            return node_emb
            
        # Edge embeddings
        edge_emb = self.edge_encoder(edge_features)  # [num_edges, hidden_dim]
        
        # Message passing
        for layer in self.message_layers:
            new_node_emb = node_emb.clone()
            
            for i in range(len(edge_list)):
                src, dst = edge_list[i]
                src, dst = int(src), int(dst)
                
                if src >= max_nodes or dst >= max_nodes:
                    continue
                
                # Message from src to dst
                message_input = torch.cat([
                    node_emb[src], 
                    edge_emb[i], 
                    node_emb[dst]
                ], dim=0)
                
                message = layer(message_input.unsqueeze(0)).squeeze(0)
                new_node_emb[dst] = new_node_emb[dst] + message
                
                # Message from dst to src (undirected)
                message_input = torch.cat([
                    node_emb[dst], 
                    edge_emb[i], 
                    node_emb[src]
                ], dim=0)
                
                message = layer(message_input.unsqueeze(0)).squeeze(0)
                new_node_emb[src] = new_node_emb[src] + message
            
            node_emb = self.aggregation(new_node_emb)
            
        return node_emb


class TemporalSequenceEncoder(nn.Module):
    """Encodes temporal sequence of edges"""
    
    def __init__(self, edge_feat_dim, hidden_dim, num_heads=8, num_layers=4):
        super(TemporalSequenceEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Edge and time encoding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim + 1, hidden_dim),  # edge_feat + time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, edge_features, timestamps):
        """
        Args:
            edge_features: [seq_len, edge_feat_dim]
            timestamps: [seq_len]
        """
        if len(edge_features) == 0:
            return torch.zeros(1, self.hidden_dim).to(edge_features.device)
            
        # Combine edge features with time
        time_expanded = timestamps.unsqueeze(-1)  # [seq_len, 1]
        edge_time_feat = torch.cat([edge_features, time_expanded], dim=-1)  # [seq_len, edge_feat_dim + 1]
        
        # Encode edges
        encoded_edges = self.edge_encoder(edge_time_feat)  # [seq_len, hidden_dim]
        
        # Add positional encoding
        encoded_edges = self.pos_encoding(encoded_edges.unsqueeze(0)).squeeze(0)
        
        # Transformer encoding
        sequence_emb = self.transformer(encoded_edges.unsqueeze(0)).squeeze(0)  # [seq_len, hidden_dim]
        
        # Return final state
        return sequence_emb[-1:] if len(sequence_emb.shape) > 1 else sequence_emb.unsqueeze(0)


class LinkPredictor(nn.Module):
    """Predicts link existence probability"""
    
    def __init__(self, hidden_dim):
        super(LinkPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Link prediction layers
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # sequence + src_emb + dst_emb
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequence_context, src_emb, dst_emb):
        """
        Args:
            sequence_context: [1, hidden_dim] - temporal sequence context
            src_emb: [hidden_dim] - source node embedding  
            dst_emb: [hidden_dim] - destination node embedding
        """
        # Combine all contexts
        combined = torch.cat([
            sequence_context.squeeze(0), 
            src_emb, 
            dst_emb
        ], dim=0)
        
        # Predict link probability
        prob = self.link_predictor(combined.unsqueeze(0)).squeeze(0)
        return prob


class TGAM_LinkPrediction(nn.Module):
    """TGAM for Link Prediction Task"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=128, 
                 max_nodes=1000, num_graph_layers=2, num_temporal_layers=4):
        super(TGAM_LinkPrediction, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # Components
        self.graph_encoder = GraphStateEncoder(
            node_feat_dim, edge_feat_dim, hidden_dim, num_graph_layers
        )
        
        self.temporal_encoder = TemporalSequenceEncoder(
            edge_feat_dim, hidden_dim, num_heads=8, num_layers=num_temporal_layers
        )
        
        self.link_predictor = LinkPredictor(hidden_dim)
        
    def forward(self, src_l, dst_l, candidate_dst, ts_l, e_idx_l, node_features, edge_features):
        """
        Args:
            src_l: [seq_len] - source nodes sequence
            dst_l: [seq_len] - destination nodes sequence  
            candidate_dst: destination node to evaluate
            ts_l: [seq_len] - timestamps
            e_idx_l: [seq_len] - edge indices
            node_features: [num_nodes, node_feat_dim]
            edge_features: [num_edges, edge_feat_dim]
        """
        device = next(self.parameters()).device
        
        # Convert to tensors
        if not isinstance(src_l, torch.Tensor):
            src_l = torch.tensor(src_l, dtype=torch.long).to(device)
        if not isinstance(dst_l, torch.Tensor):
            dst_l = torch.tensor(dst_l, dtype=torch.long).to(device)
        if not isinstance(ts_l, torch.Tensor):
            ts_l = torch.tensor(ts_l, dtype=torch.float32).to(device)
        if not isinstance(e_idx_l, torch.Tensor):
            e_idx_l = torch.tensor(e_idx_l, dtype=torch.long).to(device)
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float32).to(device)
        if not isinstance(edge_features, torch.Tensor):
            edge_features = torch.tensor(edge_features, dtype=torch.float32).to(device)
            
        # Create edge list for graph encoder
        edge_list = torch.stack([src_l, dst_l], dim=1)  # [seq_len, 2]
        
        # Get edge features for this sequence
        seq_edge_features = edge_features[e_idx_l]  # [seq_len, edge_feat_dim]
        
        # Encode current graph state
        node_embeddings = self.graph_encoder(
            node_features, edge_list, seq_edge_features, self.max_nodes
        )  # [max_nodes, hidden_dim]
        
        # Encode temporal sequence
        sequence_context = self.temporal_encoder(seq_edge_features, ts_l)  # [1, hidden_dim]
        
        # Get embeddings for src and candidate dst
        src_node = src_l[-1].item()  # Last source node
        src_emb = node_embeddings[src_node] if src_node < self.max_nodes else torch.zeros(self.hidden_dim).to(device)
        dst_emb = node_embeddings[candidate_dst] if candidate_dst < self.max_nodes else torch.zeros(self.hidden_dim).to(device)
        
        # Predict link probability
        link_prob = self.link_predictor(sequence_context, src_emb, dst_emb)
        
        return link_prob 
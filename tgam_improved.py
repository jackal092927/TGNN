"""
Improved TGAM Model with Better Initialization and Architecture

Key improvements:
1. Proper weight initialization (Xavier/He initialization)
2. LeakyReLU instead of ReLU to prevent dying neurons
3. Batch normalization for better gradient flow
4. Residual connections where appropriate
5. Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


def init_weights(m):
    """Proper weight initialization"""
    if isinstance(m, nn.Linear):
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.TransformerEncoderLayer):
        # He initialization for transformer layers (they use ReLU internally)
        for param in m.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


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


class ImprovedGraphStateEncoder(nn.Module):
    """Improved graph state encoder with better activations and normalization"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers=2):
        super(ImprovedGraphStateEncoder, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embedding with batch norm and leaky relu
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Edge embedding with batch norm and leaky relu
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Message passing layers with residual connections
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm works with single samples
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Aggregation with residual connection
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm works with single samples
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Apply proper initialization
        self.apply(init_weights)
        
    def forward(self, node_features, edge_list, edge_features, max_nodes):
        """Forward pass with improved architecture"""
        device = node_features.device
        
        # Initialize node embeddings for all possible nodes
        if len(node_features) < max_nodes:
            padding = torch.zeros(max_nodes - len(node_features), self.node_feat_dim).to(device)
            padded_node_features = torch.cat([node_features, padding], dim=0)
        else:
            padded_node_features = node_features[:max_nodes]
        
        # Encode node features
        node_emb = self.node_encoder(padded_node_features)
        
        if len(edge_list) == 0:
            return node_emb
            
        # Encode edge features
        edge_emb = self.edge_encoder(edge_features)
        
        # Message passing with residual connections
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
                new_node_emb[dst] = new_node_emb[dst] + message  # Residual connection
                
                # Message from dst to src (undirected)
                message_input = torch.cat([
                    node_emb[dst], 
                    edge_emb[i], 
                    node_emb[src]
                ], dim=0)
                
                message = layer(message_input.unsqueeze(0)).squeeze(0)
                new_node_emb[src] = new_node_emb[src] + message  # Residual connection
            
            # Aggregation with residual connection
            aggregated = self.aggregation(new_node_emb)
            node_emb = node_emb + aggregated  # Residual connection
            
        return node_emb


class ImprovedTemporalSequenceEncoder(nn.Module):
    """Improved temporal encoder with better initialization"""
    
    def __init__(self, edge_feat_dim, hidden_dim, num_heads=8, num_layers=4):
        super(ImprovedTemporalSequenceEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Edge and time encoding with proper initialization
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'  # GELU instead of ReLU to prevent dying neurons
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Apply proper initialization
        self.apply(init_weights)
        
    def forward(self, edge_features, timestamps):
        """Forward pass with improved architecture"""
        if len(edge_features) == 0:
            return torch.zeros(1, self.hidden_dim).to(edge_features.device)
            
        # Combine edge features with time
        time_expanded = timestamps.unsqueeze(-1)
        edge_time_feat = torch.cat([edge_features, time_expanded], dim=-1)
        
        # Encode edges
        encoded_edges = self.edge_encoder(edge_time_feat)
        
        # Add positional encoding
        encoded_edges = self.pos_encoding(encoded_edges.unsqueeze(0)).squeeze(0)
        
        # Transformer encoding
        sequence_emb = self.transformer(encoded_edges.unsqueeze(0)).squeeze(0)
        
        # Return final state
        return sequence_emb[-1:] if len(sequence_emb.shape) > 1 else sequence_emb.unsqueeze(0)


class ImprovedLinkPredictor(nn.Module):
    """Improved link predictor with better architecture and initialization"""
    
    def __init__(self, hidden_dim):
        super(ImprovedLinkPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Improved link prediction with residual connections and better activations
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Apply proper initialization
        self.apply(init_weights)
        
    def forward(self, sequence_context, src_emb, dst_emb):
        """Forward pass with improved architecture"""
        # Combine all contexts
        combined = torch.cat([
            sequence_context.squeeze(0), 
            src_emb, 
            dst_emb
        ], dim=0)
        
        # Predict link probability
        prob = self.link_predictor(combined.unsqueeze(0)).squeeze(0)
        return prob


class TGAM_LinkPrediction_Improved(nn.Module):
    """Improved TGAM with better initialization and architecture"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=128, 
                 max_nodes=1000, num_graph_layers=2, num_temporal_layers=4):
        super(TGAM_LinkPrediction_Improved, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # Improved components
        self.graph_encoder = ImprovedGraphStateEncoder(
            node_feat_dim, edge_feat_dim, hidden_dim, num_graph_layers
        )
        
        self.temporal_encoder = ImprovedTemporalSequenceEncoder(
            edge_feat_dim, hidden_dim, num_heads=8, num_layers=num_temporal_layers
        )
        
        self.link_predictor = ImprovedLinkPredictor(hidden_dim)
        
        print(f"Improved TGAM initialized with:")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Graph layers: {num_graph_layers}")
        print(f"  Temporal layers: {num_temporal_layers}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters())}")
        
    def forward(self, src_l, dst_l, candidate_dst, ts_l, e_idx_l, node_features, edge_features):
        """Forward pass with improved model"""
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
            
        # Create edge list
        edge_list = torch.stack([src_l, dst_l], dim=1)
        
        # Get edge features for this sequence
        seq_edge_features = edge_features[e_idx_l]
        
        # Encode current graph state
        node_embeddings = self.graph_encoder(
            node_features, edge_list, seq_edge_features, self.max_nodes
        )
        
        # Encode temporal sequence
        sequence_context = self.temporal_encoder(seq_edge_features, ts_l)
        
        # Get embeddings for src and candidate dst
        src_node = src_l[-1].item()
        src_emb = node_embeddings[src_node] if src_node < self.max_nodes else torch.zeros(self.hidden_dim).to(device)
        dst_emb = node_embeddings[candidate_dst] if candidate_dst < self.max_nodes else torch.zeros(self.hidden_dim).to(device)
        
        # Predict link probability
        link_prob = self.link_predictor(sequence_context, src_emb, dst_emb)
        
        return link_prob 
"""
Temporal Graph Autoregressive Model (TGAM)

An autoregressive approach to temporal graph modeling that predicts the next edge 
given the current graph state and temporal history. This model is designed to work 
better with synthetic datasets by learning the underlying generative process.
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
        
    def forward(self, node_features, edge_list, edge_features, num_nodes):
        """
        Args:
            node_features: [num_nodes, node_feat_dim]
            edge_list: [num_edges, 2] - source and target nodes
            edge_features: [num_edges, edge_feat_dim]
            num_nodes: total number of nodes
        """
        device = node_features.device
        
        # Initialize node embeddings
        node_emb = self.node_encoder(node_features)  # [num_nodes, hidden_dim]
        
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
    """Encodes temporal sequence of graph states"""
    
    def __init__(self, hidden_dim, num_heads=8, num_layers=6):
        super(TemporalSequenceEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Time encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, graph_states, timestamps):
        """
        Args:
            graph_states: [seq_len, num_nodes, hidden_dim]
            timestamps: [seq_len]
        """
        seq_len, num_nodes, hidden_dim = graph_states.shape
        
        # Aggregate nodes for each timestep (mean pooling)
        timestep_emb = graph_states.mean(dim=1)  # [seq_len, hidden_dim]
        
        # Add time encoding
        time_emb = self.time_encoder(timestamps.unsqueeze(-1))  # [seq_len, hidden_dim]
        timestep_emb = timestep_emb + time_emb
        
        # Add positional encoding
        timestep_emb = self.pos_encoding(timestep_emb.unsqueeze(0)).squeeze(0)
        
        # Transformer encoding
        sequence_emb = self.transformer(timestep_emb.unsqueeze(0)).squeeze(0)
        
        return sequence_emb


class EdgePredictor(nn.Module):
    """Predicts next edge given current context"""
    
    def __init__(self, hidden_dim, max_nodes):
        super(EdgePredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # Context processor
        self.context_processor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # current + sequence + time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Source node predictor
        self.src_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes)
        )
        
        # Destination node predictor (conditioned on source)
        self.dst_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # context + src_emb
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes)
        )
        
    def forward(self, current_graph_state, sequence_context, time_context, node_embeddings):
        """
        Args:
            current_graph_state: [hidden_dim] - aggregated current graph state
            sequence_context: [hidden_dim] - from temporal sequence encoder
            time_context: [hidden_dim] - time embedding
            node_embeddings: [num_nodes, hidden_dim] - current node embeddings
        """
        # Combine contexts
        context = torch.cat([current_graph_state, sequence_context, time_context], dim=0)
        context = self.context_processor(context.unsqueeze(0)).squeeze(0)
        
        # Predict source node
        src_logits = self.src_predictor(context)
        src_probs = F.softmax(src_logits, dim=0)
        
        # Sample or take argmax for source
        src_node = torch.multinomial(src_probs, 1).item()
        src_emb = node_embeddings[src_node] if src_node < len(node_embeddings) else torch.zeros_like(context)
        
        # Predict destination node (conditioned on source)
        dst_input = torch.cat([context, src_emb], dim=0)
        dst_logits = self.dst_predictor(dst_input.unsqueeze(0)).squeeze(0)
        
        return src_logits, dst_logits, src_node


class TGAM(nn.Module):
    """Temporal Graph Autoregressive Model"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=128, 
                 max_nodes=1000, num_graph_layers=2, num_temporal_layers=6):
        super(TGAM, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # Components
        self.graph_encoder = GraphStateEncoder(
            node_feat_dim, edge_feat_dim, hidden_dim, num_graph_layers
        )
        
        self.temporal_encoder = TemporalSequenceEncoder(
            hidden_dim, num_heads=8, num_layers=num_temporal_layers
        )
        
        self.edge_predictor = EdgePredictor(hidden_dim, max_nodes)
        
        # Time encoder
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, sequence_data, target_time, predict_next=True):
        """
        Args:
            sequence_data: list of (node_features, edge_list, edge_features, timestamp)
            target_time: time for next prediction
            predict_next: whether to predict next edge
        """
        device = next(self.parameters()).device
        
        # Encode each graph state in sequence
        graph_states = []
        timestamps = []
        
        for node_feat, edge_list, edge_feat, ts in sequence_data:
            if isinstance(node_feat, np.ndarray):
                node_feat = torch.tensor(node_feat, dtype=torch.float32).to(device)
            if isinstance(edge_list, np.ndarray):
                edge_list = torch.tensor(edge_list, dtype=torch.long).to(device)
            if isinstance(edge_feat, np.ndarray):
                edge_feat = torch.tensor(edge_feat, dtype=torch.float32).to(device)
            
            # Encode current graph state
            node_emb = self.graph_encoder(node_feat, edge_list, edge_feat, len(node_feat))
            graph_states.append(node_emb)
            timestamps.append(ts)
        
        if len(graph_states) == 0:
            return None, None, None
            
        # Pad to same size
        max_nodes_in_seq = max(gs.shape[0] for gs in graph_states)
        padded_states = []
        for gs in graph_states:
            if gs.shape[0] < max_nodes_in_seq:
                padding = torch.zeros(max_nodes_in_seq - gs.shape[0], self.hidden_dim).to(device)
                gs = torch.cat([gs, padding], dim=0)
            padded_states.append(gs)
        
        # Stack into tensor
        graph_states_tensor = torch.stack(padded_states)  # [seq_len, max_nodes, hidden_dim]
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32).to(device)
        
        # Encode temporal sequence
        sequence_context = self.temporal_encoder(graph_states_tensor, timestamps_tensor)
        
        if not predict_next:
            return sequence_context, None, None
        
        # Get current state
        current_state = graph_states_tensor[-1].mean(dim=0)  # Aggregate nodes
        current_sequence_context = sequence_context[-1]
        
        # Encode target time
        time_context = self.time_encoder(torch.tensor([target_time], dtype=torch.float32).to(device)).squeeze(0)
        
        # Predict next edge
        src_logits, dst_logits, src_node = self.edge_predictor(
            current_state, current_sequence_context, time_context, graph_states[-1]
        )
        
        return src_logits, dst_logits
    
    def generate_sequence(self, initial_graph, start_time, num_steps, time_step=1.0):
        """Generate a sequence of edges autoregressively"""
        
        device = next(self.parameters()).device
        
        # Initialize sequence
        sequence = [initial_graph]
        current_time = start_time
        
        generated_edges = []
        
        for step in range(num_steps):
            current_time += time_step
            
            # Predict next edge
            src_logits, dst_logits = self.forward(
                sequence, current_time, predict_next=True
            )
            
            if src_logits is None:
                break
                
            # Sample next edge
            src_probs = F.softmax(src_logits, dim=0)
            dst_probs = F.softmax(dst_logits, dim=0)
            
            src_node = torch.multinomial(src_probs, 1).item()
            dst_node = torch.multinomial(dst_probs, 1).item()
            
            # Add edge to current graph
            node_feat, edge_list, edge_feat, _ = sequence[-1]
            
            # Update edge list
            new_edge = np.array([[src_node, dst_node]])
            new_edge_list = np.vstack([edge_list, new_edge]) if len(edge_list) > 0 else new_edge
            
            # Use dummy edge features (since we don't predict them)
            dummy_edge_feat = np.zeros((1, edge_feat.shape[1])) if len(edge_feat) > 0 else np.zeros((1, 1))
            new_edge_feat_full = np.vstack([edge_feat, dummy_edge_feat]) if len(edge_feat) > 0 else dummy_edge_feat
            
            # Create new graph state
            new_graph = (node_feat, new_edge_list, new_edge_feat_full, current_time)
            sequence.append(new_graph)
            
            generated_edges.append((src_node, dst_node, current_time))
            
        return generated_edges, sequence


def compute_loss(model, batch_data, criterion):
    """Compute training loss for a batch"""
    
    total_loss = 0
    total_samples = 0
    
    for sequence in batch_data:
        if len(sequence) < 2:
            continue
            
        # Use all but last as input, predict last edge
        input_sequence = sequence[:-1]
        target_edge_data = sequence[-1]
        
        target_node_feat, target_edge_list, target_edge_feat, target_time = target_edge_data
        
        if len(target_edge_list) == 0:
            continue
            
        # Get the last edge as target
        last_edge = target_edge_list[-1]
        src_target, dst_target = last_edge[0], last_edge[1]
        
        # Predict
        src_logits, dst_logits = model(input_sequence, target_time)
        
        if src_logits is None:
            continue
            
        # Compute loss
        src_loss = criterion(src_logits.unsqueeze(0), torch.tensor([src_target], dtype=torch.long).to(src_logits.device))
        dst_loss = criterion(dst_logits.unsqueeze(0), torch.tensor([dst_target], dtype=torch.long).to(dst_logits.device))
        
        loss = src_loss + dst_loss
        total_loss += loss
        total_samples += 1
    
    return total_loss / max(total_samples, 1) 
"""
Example: Proper Autoregressive Loss with Teacher Forcing for Temporal Graph Prediction
"""

import torch
import torch.nn as nn
import numpy as np
from itertools import combinations

def autoregressive_loss_with_teacher_forcing(model, node_features, edge_features, 
                                            edge_sequence, timestamp_sequence, device):
    """
    Proper autoregressive loss with teacher forcing for temporal graph prediction
    
    Args:
        model: TGAM model
        node_features: [num_nodes, node_feat_dim] 
        edge_features: [num_edges, edge_feat_dim]
        edge_sequence: List of (src, dst, edge_idx) tuples in temporal order
        timestamp_sequence: List of timestamps corresponding to edges
        device: torch device
    
    Returns:
        total_loss: Autoregressive loss with teacher forcing
    """
    
    # Group edges by timestamp
    edges_by_timestamp = {}
    for i, (src, dst, e_idx) in enumerate(edge_sequence):
        ts = timestamp_sequence[i]
        if ts not in edges_by_timestamp:
            edges_by_timestamp[ts] = []
        edges_by_timestamp[ts].append((src, dst, e_idx))
    
    timestamps = sorted(edges_by_timestamp.keys())
    total_loss = 0.0
    num_predictions = 0
    
    criterion = nn.BCELoss()
    
    # Autoregressive prediction with teacher forcing
    for t_idx, current_timestamp in enumerate(timestamps[1:], 1):  # Skip timestamp 0 (initial state)
        
        # TEACHER FORCING: Use ground truth edges up to previous timestamp
        history_edges = []
        history_timestamps = []
        history_edge_indices = []
        
        for prev_ts in timestamps[:t_idx]:
            for src, dst, e_idx in edges_by_timestamp[prev_ts]:
                history_edges.append((src, dst))
                history_timestamps.append(prev_ts)
                history_edge_indices.append(e_idx)
        
        if len(history_edges) == 0:
            continue
            
        # Convert to arrays for model input
        history_src = np.array([edge[0] for edge in history_edges])
        history_dst = np.array([edge[1] for edge in history_edges])
        history_ts = np.array(history_timestamps)
        history_e_idx = np.array(history_edge_indices)
        
        # Ground truth edges at current timestamp
        current_edges = edges_by_timestamp[current_timestamp]
        current_nodes = set()
        for src, dst, _ in current_edges:
            current_nodes.update([src, dst])
        
        # Generate ALL possible edges between active nodes at current timestamp
        possible_edges = list(combinations(current_nodes, 2))
        if len(possible_edges) == 0:
            continue
        
        # For each possible edge, predict if it should exist
        edge_predictions = []
        edge_labels = []
        
        for candidate_src, candidate_dst in possible_edges:
            # Check if this edge actually exists in ground truth
            edge_exists = any(
                (src == candidate_src and dst == candidate_dst) or 
                (src == candidate_dst and dst == candidate_src)
                for src, dst, _ in current_edges
            )
            
            # Predict probability of this edge existing
            prob = model(
                history_src, history_dst, candidate_dst,  # Use candidate_dst as target
                history_ts, history_e_idx,
                node_features, edge_features
            )
            
            edge_predictions.append(prob)
            edge_labels.append(1.0 if edge_exists else 0.0)
        
        if len(edge_predictions) > 0:
            # Compute loss for all edge predictions at this timestamp
            predictions_tensor = torch.cat(edge_predictions)
            labels_tensor = torch.tensor(edge_labels, device=device, dtype=torch.float32)
            
            timestamp_loss = criterion(predictions_tensor, labels_tensor)
            total_loss += timestamp_loss
            num_predictions += len(edge_predictions)
    
    # Average loss across all predictions
    if num_predictions > 0:
        return total_loss / len(timestamps[1:])  # Average per timestamp
    else:
        return torch.tensor(0.0, device=device)

def simplified_autoregressive_loss(model, src_sequence, dst_sequence, ts_sequence, 
                                  e_idx_sequence, node_features, edge_features, device):
    """
    Simplified autoregressive loss: predict next edge given sequence
    (More practical for our current TGAM architecture)
    """
    
    unique_timestamps = np.unique(ts_sequence)
    total_loss = 0.0
    criterion = nn.BCELoss()
    
    for t_idx in range(1, len(unique_timestamps)):
        current_ts = unique_timestamps[t_idx]
        
        # Find edges up to previous timestamp (teacher forcing)
        history_mask = ts_sequence < current_ts
        if not np.any(history_mask):
            continue
            
        history_src = src_sequence[history_mask]
        history_dst = dst_sequence[history_mask]
        history_ts = ts_sequence[history_mask]
        history_e_idx = e_idx_sequence[history_mask]
        
        # Find edges at current timestamp (prediction targets)
        current_mask = ts_sequence == current_ts
        current_edges = list(zip(src_sequence[current_mask], dst_sequence[current_mask]))
        
        if len(current_edges) == 0:
            continue
        
        # For each edge at current timestamp, predict its probability
        timestamp_loss = 0.0
        num_current_edges = 0
        
        for target_src, target_dst in current_edges:
            # Positive prediction (real edge)
            pos_prob = model(
                np.concatenate([history_src, [target_src]]),
                np.concatenate([history_dst, [target_dst]]), 
                target_dst,
                np.concatenate([history_ts, [current_ts]]),
                np.concatenate([history_e_idx, [e_idx_sequence[ts_sequence == current_ts][0]]]),
                node_features, edge_features
            )
            
            # Negative prediction (random edge that doesn't exist at this timestamp)
            # Sample a random node that's not target_dst
            all_nodes = set(np.concatenate([history_src, history_dst, [target_src, target_dst]]))
            fake_nodes = list(all_nodes - {target_dst})
            if len(fake_nodes) > 0:
                fake_dst = np.random.choice(fake_nodes)
                
                neg_prob = model(
                    np.concatenate([history_src, [target_src]]),
                    np.concatenate([history_dst, [target_dst]]),  # Keep history same
                    fake_dst,  # But predict fake destination
                    np.concatenate([history_ts, [current_ts]]),
                    np.concatenate([history_e_idx, [e_idx_sequence[ts_sequence == current_ts][0]]]),
                    node_features, edge_features
                )
                
                # Compute loss for this edge
                pos_label = torch.tensor([1.0], device=device)
                neg_label = torch.tensor([0.0], device=device)
                
                edge_loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
                timestamp_loss += edge_loss
                num_current_edges += 1
        
        if num_current_edges > 0:
            total_loss += timestamp_loss / num_current_edges
    
    return total_loss / (len(unique_timestamps) - 1) if len(unique_timestamps) > 1 else torch.tensor(0.0, device=device)

# Example usage in training loop:
def train_with_autoregressive_loss():
    """
    Example training loop with proper autoregressive loss
    """
    
    # ... model setup ...
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Method 1: Full autoregressive loss (computationally expensive)
            # loss = autoregressive_loss_with_teacher_forcing(
            #     model, node_features, edge_features, 
            #     edge_sequence, timestamp_sequence, device
            # )
            
            # Method 2: Simplified autoregressive loss (more practical)
            loss = simplified_autoregressive_loss(
                model, src_sequence, dst_sequence, ts_sequence,
                e_idx_sequence, node_features, edge_features, device
            )
            
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    print("This file demonstrates proper autoregressive loss for temporal graph prediction")
    print("Key differences from current implementation:")
    print("1. Teacher forcing: Use ground truth history at each timestamp")
    print("2. Multi-edge prediction: Predict all edges at each timestamp")
    print("3. Temporal consistency: Maintain proper temporal ordering") 
"""
SIMPLE IMPROVED GraphRNN Training Script
Uses the existing forward_sequence method with improvements:
1. Increased training scale (300+ epochs, lr=0.0001)
2. Gradient clipping (max_norm=1.0)  
3. Increased model capacity (hidden_dim=256, rnn_layers=3)
4. Better optimization and regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import json
import logging
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from graph_rnn_correct import GraphRNN_Correct


def setup_logging(data_name):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'log/graphrnn_simple_improved_{data_name}_{int(time.time())}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    # Load ground truth
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, e_feat, n_feat, ground_truth


def create_graph_sequence(g_df):
    """Create sequence of edge lists per timestamp"""
    max_timestamp = int(g_df['ts'].max())
    graph_sequence = []
    
    for t in range(max_timestamp + 1):
        edges_at_t = g_df[g_df['ts'] == t]
        edge_list = []
        
        for _, row in edges_at_t.iterrows():
            edge_list.append([int(row.u), int(row.i)])
        
        graph_sequence.append(edge_list)
    
    return graph_sequence


def evaluate_sequence_predictions_balanced(predictions, ground_truth_sequence, logger, balance_ratio=1.0):
    """
    Evaluate predictions from forward_sequence with BALANCED SAMPLING
    
    Args:
        predictions: List of (edge_logits, candidate_edges) from model
        ground_truth_sequence: Ground truth edges for each step
        logger: Logger instance
        balance_ratio: Ratio of negative to positive samples (1.0 = equal)
    """
    all_true_labels = []
    all_pred_scores = []
    total_positives = 0
    total_negatives = 0
    
    for step_idx, (edge_logits, candidate_edges) in enumerate(predictions):
        if step_idx >= len(ground_truth_sequence):
            continue
            
        gt_edges = set()
        for edge in ground_truth_sequence[step_idx]:
            gt_edges.add((edge[0], edge[1]))
            gt_edges.add((edge[1], edge[0]))  # Add reverse
        
        if len(candidate_edges) == 0:
            continue
            
        # Convert logits to probabilities
        probs = torch.sigmoid(edge_logits).cpu().numpy()
        
        # Separate positive and negative candidates
        pos_indices = []
        neg_indices = []
        
        for i, (u, v) in enumerate(candidate_edges):
            if (u, v) in gt_edges:
                pos_indices.append(i)
            else:
                neg_indices.append(i)
        
        # BALANCED SAMPLING - Critical fix!
        if len(pos_indices) > 0:
            # Sample negatives according to balance_ratio
            num_neg_samples = min(len(neg_indices), int(len(pos_indices) * balance_ratio))
            
            if num_neg_samples > 0:
                # Randomly sample negatives
                neg_sample_indices = np.random.choice(neg_indices, num_neg_samples, replace=False)
                
                # Add positive samples
                for i in pos_indices:
                    all_true_labels.append(1)
                    all_pred_scores.append(probs[i])
                    total_positives += 1
                
                # Add sampled negative samples
                for i in neg_sample_indices:
                    all_true_labels.append(0)
                    all_pred_scores.append(probs[i])
                    total_negatives += 1
    
    if len(all_true_labels) == 0:
        return {'accuracy': 0.0, 'auc': 0.5, 'ap': 0.0}
    
    # Convert to numpy arrays
    y_true = np.array(all_true_labels)
    y_scores = np.array(all_pred_scores)
    
    # Calculate metrics
    y_pred = (y_scores > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    else:
        auc = 0.5
        ap = 0.0
    
    # Log sampling info
    pos_ratio = total_positives / (total_positives + total_negatives) if (total_positives + total_negatives) > 0 else 0
    logger.info(f"BALANCED Evaluation: {total_positives} pos, {total_negatives} neg ({pos_ratio:.1%} positive)")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'ap': ap,
        'total_samples': len(all_true_labels),
        'positive_ratio': pos_ratio
    }


def train_simple_improved_graphrnn(data_name, epochs=300, lr=0.0001, hidden_dim=256, rnn_layers=3, gradient_clip=1.0):
    """
    Train improved GraphRNN using forward_sequence method
    """
    
    logger = setup_logging(data_name)
    logger.info(f"🚀 Training SIMPLE IMPROVED GraphRNN on {data_name}")
    logger.info(f"💪 IMPROVEMENTS: epochs={epochs}, lr={lr}, hidden_dim={hidden_dim}, layers={rnn_layers}")
    logger.info(f"🔧 FIXES: gradient_clip={gradient_clip}")
    
    # Load data
    g_df, e_feat, n_feat, ground_truth = load_triadic_data(data_name)
    logger.info(f"Dataset: {len(g_df)} edges, {len(set(g_df['u'].tolist() + g_df['i'].tolist()))} nodes")
    
    # Create graph sequence
    graph_sequence = create_graph_sequence(g_df)
    logger.info(f"Created sequence with {len(graph_sequence)} timestamps")
    
    # Create node features
    unique_nodes = sorted(set(g_df['u'].tolist() + g_df['i'].tolist()))
    max_node_id = max(unique_nodes)
    
    if n_feat.shape[0] > max_node_id:
        node_features = n_feat[:max_node_id + 1]
    else:
        node_features = np.random.randn(max_node_id + 1, n_feat.shape[1])
        node_features[:n_feat.shape[0]] = n_feat
    
    logger.info(f"Node features shape: {node_features.shape}")
    
    # Split data temporally
    total_steps = len(graph_sequence)
    train_steps = int(total_steps * 0.6)  # 60% for training
    val_steps = int(total_steps * 0.2)    # 20% for validation
    test_steps = total_steps - train_steps - val_steps  # 20% for testing
    
    logger.info(f"Train: {train_steps} steps (ts 0-{train_steps-1})")
    logger.info(f"Val: {val_steps} steps (ts {train_steps}-{train_steps+val_steps-1})")  
    logger.info(f"Test: {test_steps} steps (ts {train_steps+val_steps}-{total_steps-1})")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize IMPROVED model
    model = GraphRNN_Correct(
        node_feat_dim=node_features.shape[1],
        hidden_dim=hidden_dim,  # INCREASED
        max_nodes=node_features.shape[0],
        rnn_layers=rnn_layers   # INCREASED
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {total_params:,} parameters")
    
    # Setup optimizer with REDUCED learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, verbose=True
    )
    
    # Training variables
    best_val_ap = 0.0
    best_model_path = f'models/graphrnn_simple_improved_{data_name}_best.pt'
    patience_counter = 0
    max_patience = 50
    
    # Convert to torch tensors
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32, device=device)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Training sequence (first train_steps)
        train_sequence = graph_sequence[:train_steps]
        
        # Forward pass
        predictions = model.forward_sequence(
            node_features_tensor, 
            train_sequence, 
            node_features.shape[0],
            teacher_forcing=True
        )
        
        # Calculate loss
        total_loss = 0.0
        num_batches = 0
        
        for step_idx, (edge_logits, candidate_edges) in enumerate(predictions):
            if len(candidate_edges) == 0:
                continue
                
            # Ground truth for this step
            gt_edges = set()
            if step_idx + 1 < len(train_sequence):  # Next step ground truth
                for edge in train_sequence[step_idx + 1]:
                    gt_edges.add((edge[0], edge[1]))
                    gt_edges.add((edge[1], edge[0]))
            
            # Create targets
            targets = []
            valid_logits = []
            
            for i, (u, v) in enumerate(candidate_edges):
                if (u, v) in gt_edges:
                    targets.append(1.0)
                else:
                    targets.append(0.0)
                valid_logits.append(edge_logits[i])
            
            if len(valid_logits) > 0:
                targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
                logits_tensor = torch.stack(valid_logits)
                
                loss = nn.functional.binary_cross_entropy_with_logits(logits_tensor, targets_tensor)
                total_loss += loss
                num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            
            # Backward pass
            optimizer.zero_grad()
            avg_loss.backward()
            
            # GRADIENT CLIPPING - Critical fix!
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            loss_value = avg_loss.item()
        else:
            loss_value = 0.0
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            
            with torch.no_grad():
                # Validation sequence
                val_sequence = graph_sequence[train_steps:train_steps+val_steps]
                
                # Use training sequence to build up hidden state, then predict validation
                full_sequence = graph_sequence[:train_steps+val_steps]
                val_predictions = model.forward_sequence(
                    node_features_tensor,
                    full_sequence,
                    node_features.shape[0],
                    teacher_forcing=False
                )
                
                # Only evaluate validation part
                val_pred_subset = val_predictions[train_steps:]
                val_gt_subset = val_sequence
                
                val_metrics = evaluate_sequence_predictions_balanced(val_pred_subset, val_gt_subset, logger, balance_ratio=1.0)
            
            logger.info(f"Epoch {epoch}: Loss={loss_value:.4f}, "
                       f"Val Acc={val_metrics['accuracy']:.4f}, "
                       f"Val AUC={val_metrics['auc']:.4f}, "
                       f"Val AP={val_metrics['ap']:.4f}")
            
            # Save best model
            if val_metrics['ap'] > best_val_ap:
                best_val_ap = val_metrics['ap']
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"🏆 New best model saved with Val AP: {best_val_ap:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            scheduler.step(val_metrics['ap'])
            
            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={max_patience})")
                break
        
        else:
            logger.info(f"Epoch {epoch}: Loss={loss_value:.4f}")
    
    # Final test evaluation
    logger.info("🎯 Final test evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    with torch.no_grad():
        # Full sequence for test
        test_predictions = model.forward_sequence(
            node_features_tensor,
            graph_sequence,
            node_features.shape[0],
            teacher_forcing=False
        )
        
        # Only evaluate test part
        test_pred_subset = test_predictions[train_steps+val_steps:]
        test_gt_subset = graph_sequence[train_steps+val_steps:]
        
        test_metrics = evaluate_sequence_predictions_balanced(test_pred_subset, test_gt_subset, logger, balance_ratio=1.0)
    
    logger.info("🎯 Final Test Results:")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  Test AP: {test_metrics['ap']:.4f}")
    logger.info(f"  Best Val AP: {best_val_ap:.4f}")
    
    print("=" * 60)
    print("FINAL RESULTS (SIMPLE IMPROVED GraphRNN)")
    print("=" * 60)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC:      {test_metrics['auc']:.4f}")
    print(f"Test AP:       {test_metrics['ap']:.4f}")
    print(f"Best Val AP:   {best_val_ap:.4f}")
    print("=" * 60)
    
    return test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='triadic_perfect_long_dense')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--rnn_layers', type=int, default=3)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    
    args = parser.parse_args()
    
    print("🚀 SIMPLE IMPROVED GraphRNN Training")
    print(f"Implementing key fixes for {args.data}")
    
    train_simple_improved_graphrnn(
        data_name=args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        rnn_layers=args.rnn_layers,
        gradient_clip=args.gradient_clip
    )

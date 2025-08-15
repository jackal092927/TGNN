"""
Correct GraphRNN Training Script
- Train on ALL edge predictions at each timestamp
- Use teacher forcing during training
- Proper autoregressive evaluation
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
from graph_rnn_correct import GraphRNN_Correct, create_edge_sequence_from_data


def setup_logging(data_name):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'log/graphrnn_correct_{data_name}_{int(time.time())}.log'),
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


def evaluate_predictions_balanced(predictions, ground_truth_sequence, logger, balance_ratio=1.0):
    """
    Evaluate predictions against ground truth with balanced sampling
    
    Args:
        predictions: list of (edge_logits, candidate_edges) for each timestamp
        ground_truth_sequence: list of edge lists for each timestamp
        balance_ratio: ratio of negative to positive samples (1.0 = equal)
    """
    all_labels = []
    all_probs = []
    
    for t, (edge_logits, candidate_edges) in enumerate(predictions):
        if t + 1 >= len(ground_truth_sequence):
            break
            
        # Ground truth edges at t+1
        gt_edges_t1 = set()
        for edge in ground_truth_sequence[t + 1]:
            gt_edges_t1.add((min(edge[0], edge[1]), max(edge[0], edge[1])))
        
        # Ground truth edges at t (to exclude existing edges)
        gt_edges_t = set()
        for edge in ground_truth_sequence[t]:
            gt_edges_t.add((min(edge[0], edge[1]), max(edge[0], edge[1])))
        
        # New edges at t+1
        new_edges = gt_edges_t1 - gt_edges_t
        
        if len(edge_logits) == 0 or len(new_edges) == 0:
            continue
            
        # Convert logits to probabilities
        probs = torch.sigmoid(edge_logits).cpu().numpy()
        
        # Separate positive and negative samples
        positive_indices = []
        negative_indices = []
        
        for i, (src, dst) in enumerate(candidate_edges):
            edge_key = (min(src, dst), max(src, dst))
            
            if edge_key in new_edges:
                positive_indices.append(i)
            else:
                negative_indices.append(i)
        
        # Balance sampling
        num_positives = len(positive_indices)
        num_negatives_to_sample = int(num_positives * balance_ratio)
        
        if num_negatives_to_sample > 0 and len(negative_indices) > 0:
            # Randomly sample negatives
            if num_negatives_to_sample >= len(negative_indices):
                sampled_negative_indices = negative_indices
            else:
                sampled_negative_indices = np.random.choice(
                    negative_indices, num_negatives_to_sample, replace=False
                ).tolist()
            
            # Add balanced samples
            for idx in positive_indices:
                all_labels.append(1)
                all_probs.append(probs[idx])
                
            for idx in sampled_negative_indices:
                all_labels.append(0)
                all_probs.append(probs[idx])
    
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Accuracy (threshold = 0.5)
    predictions_binary = (all_probs > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, predictions_binary)
    
    # AUC and AP
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
    else:
        auc = 0.5
        ap = np.mean(all_labels)
    
    logger.info(f"Balanced Evaluation: Acc={accuracy:.4f}, AUC={auc:.4f}, AP={ap:.4f}")
    logger.info(f"Positive samples: {np.sum(all_labels)}/{len(all_labels)} ({np.mean(all_labels)*100:.1f}%)")
    
    return accuracy, auc, ap


def train_graphrnn_correct(data_name='triadic_medium', num_epochs=100, learning_rate=0.001, 
                          hidden_dim=128, rnn_layers=2):
    """
    Train GraphRNN with correct methodology
    """
    logger = setup_logging(data_name)
    logger.info(f"Training Correct GraphRNN on {data_name}")
    
    # Load data
    g_df, e_feat, n_feat, ground_truth = load_triadic_data(data_name)
    logger.info(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    
    # Create edge sequence
    max_timestamp = int(g_df.ts.max())
    num_nodes = len(n_feat)
    
    edge_sequence = create_edge_sequence_from_data(g_df, max_timestamp)
    logger.info(f"Created sequence with {len(edge_sequence)} timestamps")
    
    # Split sequence temporally
    train_end = int(max_timestamp * 0.6)  # 60% for training
    val_end = int(max_timestamp * 0.8)    # 20% for validation
    
    train_sequence = edge_sequence[:train_end + 1]
    val_sequence = edge_sequence[train_end:val_end + 1]
    test_sequence = edge_sequence[val_end:]
    
    logger.info(f"Train: {len(train_sequence)} steps (ts 0-{train_end})")
    logger.info(f"Val: {len(val_sequence)} steps (ts {train_end}-{val_end})")
    logger.info(f"Test: {len(test_sequence)} steps (ts {val_end}-{max_timestamp})")
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = GraphRNN_Correct(
        node_feat_dim=n_feat.shape[1],
        hidden_dim=hidden_dim,
        max_nodes=num_nodes,
        rnn_layers=rnn_layers
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Convert node features to tensor
    node_features = torch.tensor(n_feat, dtype=torch.float32, device=device)
    
    # Training loop
    best_val_ap = 0.0
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass with teacher forcing
        predictions = model.forward_sequence(node_features, train_sequence, num_nodes, teacher_forcing=True)
        
        # Calculate loss with balanced sampling
        total_loss = 0.0
        total_samples = 0
        
        for t, (edge_logits, candidate_edges) in enumerate(predictions):
            if t + 1 >= len(train_sequence) or len(edge_logits) == 0:
                continue
            
            # Ground truth edges at t+1
            gt_edges_t1 = set()
            for edge in train_sequence[t + 1]:
                gt_edges_t1.add((min(edge[0], edge[1]), max(edge[0], edge[1])))
            
            # Ground truth edges at t
            gt_edges_t = set()
            for edge in train_sequence[t]:
                gt_edges_t.add((min(edge[0], edge[1]), max(edge[0], edge[1])))
            
            # New edges at t+1
            new_edges = gt_edges_t1 - gt_edges_t
            
            if len(new_edges) == 0:
                continue
            
            # Separate positive and negative samples
            positive_indices = []
            negative_indices = []
            
            for i, (src, dst) in enumerate(candidate_edges):
                edge_key = (min(src, dst), max(src, dst))
                if edge_key in new_edges:
                    positive_indices.append(i)
                else:
                    negative_indices.append(i)
            
            # Balance sampling for training (1:2 ratio - more negatives for training)
            num_positives = len(positive_indices)
            num_negatives_to_sample = min(num_positives * 2, len(negative_indices))
            
            if num_positives > 0 and num_negatives_to_sample > 0:
                # Sample negatives
                sampled_negative_indices = np.random.choice(
                    negative_indices, num_negatives_to_sample, replace=False
                ).tolist()
                
                # Create balanced labels and logits
                selected_indices = positive_indices + sampled_negative_indices
                selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
                selected_logits = edge_logits[selected_indices_tensor]
                
                labels = [1.0] * len(positive_indices) + [0.0] * len(sampled_negative_indices)
                labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
                
                loss = criterion(selected_logits, labels_tensor)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
        
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            train_losses.append(avg_loss)
            
            # Backward pass
            optimizer.zero_grad()
            
            # Recalculate for gradients with balanced sampling
            predictions = model.forward_sequence(node_features, train_sequence, num_nodes, teacher_forcing=True)
            batch_loss = 0.0
            
            for t, (edge_logits, candidate_edges) in enumerate(predictions):
                if t + 1 >= len(train_sequence) or len(edge_logits) == 0:
                    continue
                
                # Ground truth edges at t+1
                gt_edges_t1 = set()
                for edge in train_sequence[t + 1]:
                    gt_edges_t1.add((min(edge[0], edge[1]), max(edge[0], edge[1])))
                
                # Ground truth edges at t
                gt_edges_t = set()
                for edge in train_sequence[t]:
                    gt_edges_t.add((min(edge[0], edge[1]), max(edge[0], edge[1])))
                
                # New edges at t+1
                new_edges = gt_edges_t1 - gt_edges_t
                
                if len(new_edges) == 0:
                    continue
                
                # Separate positive and negative samples
                positive_indices = []
                negative_indices = []
                
                for i, (src, dst) in enumerate(candidate_edges):
                    edge_key = (min(src, dst), max(src, dst))
                    if edge_key in new_edges:
                        positive_indices.append(i)
                    else:
                        negative_indices.append(i)
                
                # Balance sampling for training
                num_positives = len(positive_indices)
                num_negatives_to_sample = min(num_positives * 2, len(negative_indices))
                
                if num_positives > 0 and num_negatives_to_sample > 0:
                    # Sample negatives
                    sampled_negative_indices = np.random.choice(
                        negative_indices, num_negatives_to_sample, replace=False
                    ).tolist()
                    
                    # Create balanced labels and logits
                    selected_indices = positive_indices + sampled_negative_indices
                    selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
                    selected_logits = edge_logits[selected_indices_tensor]
                    
                    labels = [1.0] * len(positive_indices) + [0.0] * len(sampled_negative_indices)
                    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
                    
                    loss = criterion(selected_logits, labels_tensor)
                    batch_loss += loss
            
            if batch_loss > 0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        scheduler.step()
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model.forward_sequence(node_features, val_sequence, num_nodes, teacher_forcing=True)
                val_acc, val_auc, val_ap = evaluate_predictions_balanced(val_predictions, val_sequence, logger, balance_ratio=1.0)
                
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}")
                
                # Save best model
                if val_ap > best_val_ap:
                    best_val_ap = val_ap
                    torch.save(model.state_dict(), f'models/graphrnn_correct_{data_name}_best.pt')
                    logger.info(f"New best model saved with Val AP: {val_ap:.4f}")
        else:
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")
    
    # Final test evaluation
    logger.info("Final test evaluation...")
    
    try:
        model.load_state_dict(torch.load(f'models/graphrnn_correct_{data_name}_best.pt'))
        logger.info("Loaded best model for testing")
    except:
        logger.warning("Could not load best model, using current model")
    
    model.eval()
    with torch.no_grad():
        test_predictions = model.forward_sequence(node_features, test_sequence, num_nodes, teacher_forcing=True)
        test_acc, test_auc, test_ap = evaluate_predictions_balanced(test_predictions, test_sequence, logger, balance_ratio=1.0)
        
        logger.info(f"Final Test Results:")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        logger.info(f"  Test AUC: {test_auc:.4f}")
        logger.info(f"  Test AP: {test_ap:.4f}")
        logger.info(f"  Best Val AP: {best_val_ap:.4f}")
    
    return {
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'best_val_ap': best_val_ap,
        'train_losses': train_losses
    }


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Train Correct GraphRNN')
    parser.add_argument('--data', type=str, default='triadic_medium')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('log', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    print(f"Training Correct GraphRNN on {args.data}")
    print(f"Methodology: Predict ALL edges at each timestamp with teacher forcing")
    
    # Train model
    results = train_graphrnn_correct(
        data_name=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        rnn_layers=args.rnn_layers
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS (Correct GraphRNN)")
    print("="*60)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC:      {results['test_auc']:.4f}")
    print(f"Test AP:       {results['test_ap']:.4f}")
    print(f"Best Val AP:   {results['best_val_ap']:.4f}")
    print("="*60)

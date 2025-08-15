"""
Training script for GraphRNN on triadic closure datasets
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
from graph_rnn import GraphRNN, create_graph_sequence_from_data
from utils import RandEdgeSampler


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'log/graphrnn_triadic_{int(time.time())}.log'),
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


def create_negative_samples(g_df, num_nodes, timestamp, num_negatives=1, existing_edges=None):
    """
    Create negative edge samples for a given timestamp
    
    Args:
        g_df: DataFrame with edge data
        num_nodes: total number of nodes
        timestamp: current timestamp
        num_negatives: number of negative samples per positive
        existing_edges: set of existing edges to avoid
    """
    if existing_edges is None:
        # Get all edges up to current timestamp
        edges_up_to_ts = g_df[g_df.ts <= timestamp]
        existing_edges = set()
        for _, row in edges_up_to_ts.iterrows():
            existing_edges.add((row.u, row.i))
            existing_edges.add((row.i, row.u))  # Undirected
    
    negative_samples = []
    attempts = 0
    max_attempts = num_negatives * 100
    
    while len(negative_samples) < num_negatives and attempts < max_attempts:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        if src != dst and (src, dst) not in existing_edges:
            negative_samples.append((src, dst))
            existing_edges.add((src, dst))
            existing_edges.add((dst, src))
        
        attempts += 1
    
    return negative_samples


def evaluate_model(model, sequence_data, positive_edges, negative_edges, device):
    """
    Evaluate model on positive and negative edge samples
    
    Args:
        model: GraphRNN model
        sequence_data: sequence of graph states
        positive_edges: list of (src, dst) positive edges
        negative_edges: list of (src, dst) negative edges
        device: torch device
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        # Score positive edges
        for src, dst in positive_edges:
            prob, _ = model(sequence_data, target_src=src, target_dst=dst)
            if prob is not None:
                all_probs.append(prob.item())
                all_labels.append(1)
        
        # Score negative edges  
        for src, dst in negative_edges:
            prob, _ = model(sequence_data, target_src=src, target_dst=dst)
            if prob is not None:
                all_probs.append(prob.item())
                all_labels.append(0)
    
    if len(all_probs) == 0:
        return 0.5, 0.5, 0.5
    
    # Calculate metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Accuracy (threshold = 0.5)
    predictions = (all_probs > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    
    # AUC and AP
    if len(np.unique(all_labels)) > 1:  # Need both classes
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
    else:
        auc = 0.5
        ap = np.mean(all_labels)
    
    return accuracy, auc, ap


def train_graphrnn(data_name='triadic_medium', num_epochs=50, learning_rate=0.001, 
                   hidden_dim=128, rnn_layers=2, batch_size=32):
    """
    Train GraphRNN on triadic closure dataset
    """
    logger = setup_logging()
    logger.info(f"Training GraphRNN on {data_name}")
    
    # Load data
    g_df, e_feat, n_feat, ground_truth = load_triadic_data(data_name)
    logger.info(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes, {e_feat.shape[1]} edge features")
    
    # Dataset info
    max_timestamp = g_df.ts.max()
    num_nodes = len(n_feat)
    logger.info(f"Max timestamp: {max_timestamp}, Nodes: {num_nodes}")
    
    # Split data temporally
    val_time = np.quantile(g_df.ts, 0.7)
    test_time = np.quantile(g_df.ts, 0.85)
    
    train_data = g_df[g_df.ts <= val_time]
    val_data = g_df[(g_df.ts > val_time) & (g_df.ts <= test_time)]
    test_data = g_df[g_df.ts > test_time]
    
    logger.info(f"Train edges: {len(train_data)}, Val edges: {len(val_data)}, Test edges: {len(test_data)}")
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = GraphRNN(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1], 
        hidden_dim=hidden_dim,
        max_nodes=num_nodes,
        rnn_layers=rnn_layers
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    best_val_ap = 0.0
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        # Get training edges (excluding initial state at timestamp 0)
        training_edges = train_data[train_data.ts > 0]
        
        if len(training_edges) == 0:
            logger.warning("No training edges found!")
            continue
        
        # Process each training edge
        for idx, row in training_edges.iterrows():
            src, dst, ts = row.u, row.i, row.ts
            
            # Create sequence up to timestamp ts-1
            sequence_data = create_graph_sequence_from_data(g_df, ts-1, n_feat, e_feat)
            
            if len(sequence_data) == 0:
                continue
            
            # Positive sample
            optimizer.zero_grad()
            
            prob_pos, logit_pos = model(sequence_data, target_src=src, target_dst=dst)
            if prob_pos is None:
                continue
                
            # Create negative samples
            existing_edges = set()
            edges_up_to_ts = g_df[g_df.ts <= ts-1]  # Don't include current edge
            for _, edge_row in edges_up_to_ts.iterrows():
                existing_edges.add((edge_row.u, edge_row.i))
                existing_edges.add((edge_row.i, edge_row.u))
            
            neg_samples = create_negative_samples(g_df, num_nodes, ts-1, num_negatives=2, 
                                                existing_edges=existing_edges)
            
            # Train on positive + negatives
            all_logits = [logit_pos]
            all_labels = [1.0]
            
            for neg_src, neg_dst in neg_samples:
                prob_neg, logit_neg = model(sequence_data, target_src=neg_src, target_dst=neg_dst)
                if prob_neg is not None:
                    all_logits.append(logit_neg)
                    all_labels.append(0.0)
            
            if len(all_logits) > 1:
                logits_tensor = torch.stack(all_logits)
                labels_tensor = torch.tensor(all_labels, device=device)
                
                loss = criterion(logits_tensor, labels_tensor)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_losses.append(loss.item())
        
        scheduler.step()
        
        if len(epoch_losses) > 0:
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            # Evaluation every 5 epochs
            if epoch % 5 == 0:
                # Validation
                val_edges = [(row.u, row.i) for _, row in val_data.iterrows()]
                val_sequence = create_graph_sequence_from_data(g_df, int(val_time), n_feat, e_feat)
                
                if len(val_edges) > 0 and len(val_sequence) > 0:
                    # Create negative samples for validation
                    existing_edges = set()
                    edges_up_to_val = g_df[g_df.ts <= val_time]
                    for _, edge_row in edges_up_to_val.iterrows():
                        existing_edges.add((edge_row.u, edge_row.i))
                        existing_edges.add((edge_row.i, edge_row.u))
                    
                    val_neg_edges = create_negative_samples(g_df, num_nodes, val_time, 
                                                          len(val_edges), existing_edges)
                    
                    val_acc, val_auc, val_ap = evaluate_model(model, val_sequence, val_edges, 
                                                            val_neg_edges, device)
                    
                    logger.info(f"Epoch {epoch}: Loss: {avg_loss:.4f}, "
                              f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
                    
                    # Save best model
                    if val_ap > best_val_ap:
                        best_val_ap = val_ap
                        torch.save(model.state_dict(), f'models/graphrnn_{data_name}_best.pt')
                        logger.info(f"New best model saved with Val AP: {val_ap:.4f}")
                else:
                    logger.info(f"Epoch {epoch}: Loss: {avg_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: Loss: {avg_loss:.4f}")
        else:
            logger.warning(f"Epoch {epoch}: No valid training samples")
    
    # Final test evaluation
    if len(test_data) > 0:
        logger.info("Final test evaluation...")
        
        # Load best model
        try:
            model.load_state_dict(torch.load(f'models/graphrnn_{data_name}_best.pt'))
            logger.info("Loaded best model for testing")
        except:
            logger.warning("Could not load best model, using current model")
        
        test_edges = [(row.u, row.i) for _, row in test_data.iterrows()]
        test_sequence = create_graph_sequence_from_data(g_df, int(test_time), n_feat, e_feat)
        
        # Create negative samples for test
        existing_edges = set()
        edges_up_to_test = g_df[g_df.ts <= test_time]
        for _, edge_row in edges_up_to_test.iterrows():
            existing_edges.add((edge_row.u, edge_row.i))
            existing_edges.add((edge_row.i, edge_row.u))
        
        test_neg_edges = create_negative_samples(g_df, num_nodes, test_time, 
                                               len(test_edges), existing_edges)
        
        test_acc, test_auc, test_ap = evaluate_model(model, test_sequence, test_edges, 
                                                   test_neg_edges, device)
        
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
    else:
        logger.warning("No test data available")
        return {
            'test_accuracy': 0.0,
            'test_auc': 0.0,
            'test_ap': 0.0, 
            'best_val_ap': best_val_ap,
            'train_losses': train_losses
        }


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Train GraphRNN on triadic closure datasets')
    parser.add_argument('--data', type=str, default='triadic_medium', 
                       help='Dataset name (default: triadic_medium)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension (default: 128)')
    parser.add_argument('--rnn_layers', type=int, default=2,
                       help='Number of RNN layers (default: 2)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device id (default: 0)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('log', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    print(f"Training GraphRNN on {args.data}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Hidden: {args.hidden_dim}")
    
    # Train model
    results = train_graphrnn(
        data_name=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        rnn_layers=args.rnn_layers
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC:      {results['test_auc']:.4f}")
    print(f"Test AP:       {results['test_ap']:.4f}")
    print(f"Best Val AP:   {results['best_val_ap']:.4f}")
    print("="*60)

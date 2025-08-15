"""
Spectral Triadic GNN - Solving the Node Identity Problem

USER'S BRILLIANT INSIGHT:
Standard GNNs can't detect triadic closures because they lose node identity
during message passing. When nodes u and v both receive messages from node w,
they don't know it's the SAME node w!

SOLUTION:
Use spectral embeddings as positional encodings to give each node a unique,
learnable identity that persists through message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import pandas as pd
import json
import logging
import time
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from collections import defaultdict

def setup_logging(data_name):
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'log/spectral_gnn_{data_name}_{int(time.time())}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def compute_spectral_embeddings(adjacency_matrix, k=8):
    """
    Compute spectral positional encodings using graph Laplacian eigendecomposition
    
    This gives each node a unique k-dimensional 'coordinate' that encodes
    its structural position in the graph.
    
    Args:
        adjacency_matrix: numpy array of shape (n_nodes, n_nodes)
        k: number of eigenvectors to use as positional encoding
        
    Returns:
        numpy array of shape (n_nodes, k) with spectral coordinates
    """
    n_nodes = adjacency_matrix.shape[0]
    
    # Handle edge cases
    if n_nodes <= k + 1:
        print(f"   ⚠️  Graph too small ({n_nodes} nodes) for {k} eigenvectors, using random embeddings")
        return np.random.randn(n_nodes, k) * 0.1
    
    # Convert to sparse matrix
    adj_sparse = csr_matrix(adjacency_matrix)
    
    # Compute degree matrix
    degrees = np.array(adj_sparse.sum(axis=1)).flatten()
    
    # Handle isolated nodes
    degrees = np.maximum(degrees, 1e-12)
    
    # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    deg_sqrt_inv = np.power(degrees, -0.5)
    deg_sqrt_inv_matrix = csr_matrix((deg_sqrt_inv, (range(n_nodes), range(n_nodes))), 
                                    shape=(n_nodes, n_nodes))
    
    normalized_adj = deg_sqrt_inv_matrix @ adj_sparse @ deg_sqrt_inv_matrix
    laplacian = csr_matrix(np.eye(n_nodes)) - normalized_adj
    
    try:
        # Compute k+1 smallest eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigsh(laplacian, k=k+1, which='SM', maxiter=1000)
        
        # Remove the first eigenvector (constant vector with eigenvalue ~0)
        spectral_embeddings = eigenvectors[:, 1:k+1]
        
        print(f"   ✅ Computed spectral embeddings: {spectral_embeddings.shape}")
        print(f"   Eigenvalues: {eigenvalues[1:k+1]}")
        
        return spectral_embeddings.astype(np.float32)
        
    except Exception as e:
        print(f"   ⚠️  Eigendecomposition failed: {e}")
        print(f"   Using random embeddings as fallback")
        return np.random.randn(n_nodes, k).astype(np.float32) * 0.1

class SpectralGNNLayer(nn.Module):
    """
    GNN layer that preserves node identity through spectral embeddings
    
    KEY INSIGHT: Include spectral coordinates in messages so that receiving
    nodes know the identity of the sender.
    """
    
    def __init__(self, hidden_dim, spectral_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spectral_dim = spectral_dim
        
        # Message function: combines node embedding + spectral identity
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + spectral_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # self + aggregated messages
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_embeddings, adj_matrix, spectral_embeddings):
        """
        Forward pass with spectral-aware message passing
        
        Args:
            node_embeddings: [n_nodes, hidden_dim] 
            adj_matrix: [n_nodes, n_nodes] adjacency matrix
            spectral_embeddings: [n_nodes, spectral_dim] positional encodings
            
        Returns:
            updated_embeddings: [n_nodes, hidden_dim]
        """
        n_nodes = node_embeddings.shape[0]
        device = node_embeddings.device
        
        # Create messages: combine node embedding + spectral identity
        # This is the KEY INSIGHT - messages carry node identity!
        messages = self.message_mlp(
            torch.cat([node_embeddings, spectral_embeddings], dim=1)
        )
        
        # Aggregate messages from neighbors
        adj_matrix = adj_matrix.to(device)
        aggregated_messages = torch.mm(adj_matrix, messages)  # Sum messages from neighbors
        
        # Normalize by degree (avoid division by zero)
        degrees = adj_matrix.sum(dim=1, keepdim=True)
        degrees = torch.clamp(degrees, min=1.0)  # Avoid division by zero
        aggregated_messages = aggregated_messages / degrees
        
        # Update node embeddings: combine self + aggregated messages
        updated = self.update_mlp(
            torch.cat([node_embeddings, aggregated_messages], dim=1)
        )
        
        # Residual connection + layer norm
        output = self.layer_norm(updated + node_embeddings)
        
        return output

class SpectralTriadicGNN(nn.Module):
    """
    Spectral GNN designed specifically for triadic closure detection
    
    ARCHITECTURE:
    1. Spectral positional encoding initialization
    2. Multi-layer spectral-aware message passing  
    3. Triadic-aware edge prediction using spectral information
    """
    
    def __init__(self, node_feat_dim, spectral_dim=8, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.spectral_dim = spectral_dim
        self.hidden_dim = hidden_dim
        
        # Project spectral embeddings to hidden dimension
        self.spectral_proj = nn.Linear(spectral_dim, hidden_dim)
        
        # Initial node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Spectral-aware GNN layers
        self.gnn_layers = nn.ModuleList([
            SpectralGNNLayer(hidden_dim, spectral_dim) 
            for _ in range(num_layers)
        ])
        
        # Edge predictor with spectral interaction features
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + spectral_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, node_features, adj_matrix, spectral_embeddings, edge_pairs=None):
        """
        Forward pass through spectral triadic GNN
        
        Args:
            node_features: [n_nodes, node_feat_dim] initial node features
            adj_matrix: [n_nodes, n_nodes] adjacency matrix  
            spectral_embeddings: [n_nodes, spectral_dim] positional encodings
            edge_pairs: [n_pairs, 2] pairs of nodes to predict edges for
            
        Returns:
            If edge_pairs provided: [n_pairs] edge prediction logits
            Else: [n_nodes, hidden_dim] final node embeddings
        """
        device = node_features.device
        spectral_embeddings = spectral_embeddings.to(device)
        
        # STEP 1: Initialize with spectral positional encoding
        spectral_encoded = self.spectral_proj(spectral_embeddings)
        initial_features = torch.cat([node_features, spectral_encoded], dim=1)
        node_embeddings = self.node_encoder(initial_features)
        
        # STEP 2: Multi-layer spectral-aware message passing
        for layer in self.gnn_layers:
            node_embeddings = layer(node_embeddings, adj_matrix, spectral_embeddings)
        
        # STEP 3: Edge prediction (if edge pairs provided)
        if edge_pairs is not None:
            return self.predict_edges(node_embeddings, spectral_embeddings, edge_pairs)
        else:
            return node_embeddings
    
    def predict_edges(self, node_embeddings, spectral_embeddings, edge_pairs):
        """
        Predict edges using spectral-enhanced features
        
        TRIADIC INSIGHT: Include spectral interaction features that can
        capture common neighbor patterns.
        """
        # Get embeddings for edge pairs
        src_embeddings = node_embeddings[edge_pairs[:, 0]]  # [n_pairs, hidden_dim]
        dst_embeddings = node_embeddings[edge_pairs[:, 1]]  # [n_pairs, hidden_dim]
        
        # Get spectral coordinates for edge pairs
        src_spectral = spectral_embeddings[edge_pairs[:, 0]]  # [n_pairs, spectral_dim]
        dst_spectral = spectral_embeddings[edge_pairs[:, 1]]  # [n_pairs, spectral_dim]
        
        # Combine all features for edge prediction
        edge_features = torch.cat([
            src_embeddings,  # Source node embedding
            dst_embeddings,  # Destination node embedding  
            src_spectral,    # Source spectral coordinates
            dst_spectral     # Destination spectral coordinates
        ], dim=1)
        
        # Predict edge probability
        edge_logits = self.edge_predictor(edge_features).squeeze(-1)
        
        return edge_logits

def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, e_feat, n_feat, ground_truth

def create_adjacency_matrix(edges, num_nodes):
    """Create adjacency matrix from edge list"""
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in edges:
        adj_matrix[u, v] = 1.0
        adj_matrix[v, u] = 1.0  # Undirected graph
    return adj_matrix

def evaluate_spectral_gnn(model, node_features, adj_matrix, spectral_embeddings, 
                         ground_truth_edges, all_nodes, device, logger, balance_ratio=1.0):
    """Evaluate spectral GNN with balanced sampling"""
    
    model.eval()
    
    # Generate all possible edge pairs
    edge_pairs = []
    labels = []
    
    gt_edge_set = set()
    for u, v in ground_truth_edges:
        gt_edge_set.add((u, v))
        gt_edge_set.add((v, u))
    
    # Collect positive and negative pairs
    pos_pairs = []
    neg_pairs = []
    
    for u in all_nodes:
        for v in all_nodes:
            if u >= v:  # Avoid duplicates
                continue
                
            if (u, v) in gt_edge_set or (v, u) in gt_edge_set:
                pos_pairs.append([u, v])
            else:
                neg_pairs.append([u, v])
    
    # Balanced sampling
    num_neg_samples = min(len(neg_pairs), int(len(pos_pairs) * balance_ratio))
    if num_neg_samples > 0:
        neg_sample_indices = np.random.choice(len(neg_pairs), num_neg_samples, replace=False)
        sampled_neg_pairs = [neg_pairs[i] for i in neg_sample_indices]
    else:
        sampled_neg_pairs = []
    
    # Combine positive and negative pairs
    all_pairs = pos_pairs + sampled_neg_pairs
    all_labels = [1] * len(pos_pairs) + [0] * len(sampled_neg_pairs)
    
    if len(all_pairs) == 0:
        return {'accuracy': 0.0, 'auc': 0.5, 'ap': 0.0}
    
    # Convert to tensors
    edge_pairs_tensor = torch.tensor(all_pairs, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32, device=device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(node_features, adj_matrix, spectral_embeddings, edge_pairs_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    
    # Calculate metrics
    accuracy = accuracy_score(labels_tensor.cpu().numpy(), preds.cpu().numpy())
    
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(labels_tensor.cpu().numpy(), probs.cpu().numpy())
        ap = average_precision_score(labels_tensor.cpu().numpy(), probs.cpu().numpy())
    else:
        auc = 0.5
        ap = 0.0
    
    pos_ratio = len(pos_pairs) / len(all_pairs) if len(all_pairs) > 0 else 0
    logger.info(f"BALANCED Evaluation: {len(pos_pairs)} pos, {len(sampled_neg_pairs)} neg ({pos_ratio:.1%} positive)")
    
    return {
        'accuracy': accuracy,
        'auc': auc, 
        'ap': ap,
        'total_samples': len(all_pairs),
        'positive_ratio': pos_ratio
    }

def train_spectral_triadic_gnn(data_name, epochs=200, lr=0.001, hidden_dim=128, spectral_dim=8):
    """Train spectral triadic GNN on triadic closure dataset"""
    
    logger = setup_logging(data_name)
    logger.info(f"🚀 Training Spectral Triadic GNN on {data_name}")
    logger.info(f"💡 USER'S INSIGHT: Using spectral embeddings as positional encodings")
    logger.info(f"🎯 GOAL: Enable GNN to detect common neighbors via node identity")
    
    # Load data
    g_df, e_feat, n_feat, ground_truth = load_triadic_data(data_name)
    logger.info(f"Dataset: {len(g_df)} edges, {len(set(g_df['u'].tolist() + g_df['i'].tolist()))} nodes")
    
    # Get all nodes and create node mapping
    all_nodes = sorted(set(g_df['u'].tolist() + g_df['i'].tolist()))
    num_nodes = len(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    # Split data temporally
    timestamps = sorted(g_df['ts'].unique())
    total_ts = len(timestamps)
    train_ts = int(total_ts * 0.6)
    val_ts = int(total_ts * 0.2)
    
    train_timestamps = timestamps[:train_ts]
    val_timestamps = timestamps[train_ts:train_ts + val_ts]
    test_timestamps = timestamps[train_ts + val_ts:]
    
    logger.info(f"Train timestamps: {len(train_timestamps)} (ts {train_timestamps[0]}-{train_timestamps[-1]})")
    logger.info(f"Val timestamps: {len(val_timestamps)} (ts {val_timestamps[0]}-{val_timestamps[-1]})")
    logger.info(f"Test timestamps: {len(test_timestamps)} (ts {test_timestamps[0]}-{test_timestamps[-1]})")
    
    # Create training graph (all edges up to validation)
    train_edges = g_df[g_df['ts'] < val_timestamps[0]]
    edge_list = [(node_to_idx[int(row.u)], node_to_idx[int(row.i)]) for _, row in train_edges.iterrows()]
    
    # Create adjacency matrix for spectral embedding computation
    adj_matrix = create_adjacency_matrix(edge_list, num_nodes)
    logger.info(f"Training graph: {len(edge_list)} edges, density: {np.sum(adj_matrix)/(num_nodes*(num_nodes-1)):.4f}")
    
    # CRITICAL STEP: Compute spectral embeddings
    logger.info(f"🧮 Computing spectral positional encodings...")
    spectral_embeddings = compute_spectral_embeddings(adj_matrix, k=spectral_dim)
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = SpectralTriadicGNN(
        node_feat_dim=n_feat.shape[1],
        spectral_dim=spectral_dim,
        hidden_dim=hidden_dim,
        num_layers=3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {total_params:,} parameters")
    
    # Convert data to tensors
    node_features = torch.tensor(n_feat[:num_nodes], dtype=torch.float32, device=device)
    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=device)
    spectral_embeddings_tensor = torch.tensor(spectral_embeddings, dtype=torch.float32, device=device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    
    # Training variables
    best_val_ap = 0.0
    best_model_path = f'models/spectral_gnn_{data_name}_best.pt'
    patience_counter = 0
    max_patience = 40
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Train on each training timestamp
        for ts in train_timestamps:
            # Get ground truth edges for this timestamp
            edges_at_ts = g_df[g_df['ts'] == ts]
            if len(edges_at_ts) == 0:
                continue
                
            gt_edges = [(node_to_idx[int(row.u)], node_to_idx[int(row.i)]) for _, row in edges_at_ts.iterrows()]
            
            # Create training pairs with balanced sampling
            pos_pairs = gt_edges
            
            # Sample negative pairs
            neg_pairs = []
            num_neg_needed = len(pos_pairs) * 2  # 1:2 ratio
            
            attempts = 0
            while len(neg_pairs) < num_neg_needed and attempts < num_neg_needed * 10:
                u = np.random.choice(all_nodes)
                v = np.random.choice(all_nodes)
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                
                if u_idx != v_idx and (u_idx, v_idx) not in gt_edges and (v_idx, u_idx) not in gt_edges:
                    if adj_matrix[u_idx, v_idx] == 0:  # Not already connected
                        neg_pairs.append((u_idx, v_idx))
                
                attempts += 1
            
            if len(neg_pairs) == 0:
                continue
                
            # Combine positive and negative pairs
            all_pairs = pos_pairs + neg_pairs
            all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
            
            # Convert to tensors
            edge_pairs_tensor = torch.tensor(all_pairs, dtype=torch.long, device=device)
            labels_tensor = torch.tensor(all_labels, dtype=torch.float32, device=device)
            
            # Forward pass
            logits = model(node_features, adj_matrix_tensor, spectral_embeddings_tensor, edge_pairs_tensor)
            loss = criterion(logits, labels_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(1, num_batches)
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            # Validation evaluation on first validation timestamp
            val_ts = val_timestamps[0]
            val_edges = g_df[g_df['ts'] == val_ts]
            val_gt_edges = [(node_to_idx[int(row.u)], node_to_idx[int(row.i)]) for _, row in val_edges.iterrows()]
            
            val_metrics = evaluate_spectral_gnn(
                model, node_features, adj_matrix_tensor, spectral_embeddings_tensor,
                val_gt_edges, all_nodes, device, logger, balance_ratio=1.0
            )
            
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
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
            
            scheduler.step(val_metrics['ap'])
            
            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        else:
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")
    
    # Final test evaluation
    logger.info("🎯 Final test evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    
    # Test on first test timestamp
    test_ts = test_timestamps[0]
    test_edges = g_df[g_df['ts'] == test_ts]
    test_gt_edges = [(node_to_idx[int(row.u)], node_to_idx[int(row.i)]) for _, row in test_edges.iterrows()]
    
    test_metrics = evaluate_spectral_gnn(
        model, node_features, adj_matrix_tensor, spectral_embeddings_tensor,
        test_gt_edges, all_nodes, device, logger, balance_ratio=1.0
    )
    
    logger.info("🎯 Final Test Results:")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  Test AP: {test_metrics['ap']:.4f}")
    logger.info(f"  Best Val AP: {best_val_ap:.4f}")
    
    print("=" * 60)
    print("FINAL RESULTS (Spectral Triadic GNN)")
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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--spectral_dim', type=int, default=8)
    
    args = parser.parse_args()
    
    print("🚀 SPECTRAL TRIADIC GNN")
    print("💡 Implementing user's insight: spectral embeddings as node identity")
    
    train_spectral_triadic_gnn(
        data_name=args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        spectral_dim=args.spectral_dim
    )

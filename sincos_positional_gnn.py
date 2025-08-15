"""
Sin/Cos Positional Embedding GNN for Triadic Closure

USER'S IMPROVED INSIGHT:
Instead of complex spectral embeddings, use simple sin/cos positional encodings
like in NLP (Transformer) frameworks. Much more stable and practical!

APPROACH:
1. Fixed-dimensional sin/cos positional embedding for each node
2. 3-layer MLP to process positional embeddings before feeding to GNN
3. Preserve node identity during message passing

This avoids eigendecomposition issues while keeping the core insight intact.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import logging
import time
import math
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

def setup_logging(data_name):
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'log/sincos_gnn_{data_name}_{int(time.time())}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_sincos_positional_embeddings(num_nodes, pos_dim=128):
    """
    Create sin/cos positional embeddings like in Transformer architectures
    
    USER'S INSIGHT: Use fixed sin/cos patterns instead of spectral embeddings
    - More stable than eigendecomposition
    - Well-established in NLP
    - Each node gets unique positional signature
    
    Args:
        num_nodes: Number of nodes in the graph
        pos_dim: Dimension of positional embeddings (should be even)
        
    Returns:
        torch.Tensor: [num_nodes, pos_dim] positional embeddings
    """
    
    if pos_dim % 2 != 0:
        pos_dim += 1  # Make even for sin/cos pairs
    
    # Create position indices for each node
    positions = torch.arange(0, num_nodes, dtype=torch.float32).unsqueeze(1)  # [num_nodes, 1]
    
    # Create dimension indices
    div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * 
                        -(math.log(10000.0) / pos_dim))  # [pos_dim/2]
    
    # Initialize positional embeddings
    pos_emb = torch.zeros(num_nodes, pos_dim)
    
    # Apply sin to even indices, cos to odd indices
    pos_emb[:, 0::2] = torch.sin(positions * div_term)  # Even: sin
    pos_emb[:, 1::2] = torch.cos(positions * div_term)  # Odd: cos
    
    print(f"   ✅ Created sin/cos positional embeddings: {pos_emb.shape}")
    print(f"   📊 Sample embeddings for first 3 nodes:")
    for i in range(min(3, num_nodes)):
        print(f"      Node {i}: [{pos_emb[i, :6].tolist()}, ...]")
    
    return pos_emb

class PositionalEncoder(nn.Module):
    """
    3-layer MLP to process positional embeddings
    
    USER'S SUGGESTION: Apply MLP after positional embedding before GNN
    This allows the model to learn how to use positional information effectively.
    """
    
    def __init__(self, pos_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, pos_embeddings):
        """
        Process raw positional embeddings through MLP
        
        Args:
            pos_embeddings: [num_nodes, pos_dim] raw sin/cos embeddings
            
        Returns:
            processed_pos: [num_nodes, output_dim] processed positional features
        """
        processed = self.pos_mlp(pos_embeddings)
        return self.layer_norm(processed)

class SinCosGNNLayer(nn.Module):
    """
    GNN layer with sin/cos positional encoding for node identity
    
    KEY INNOVATION: Messages carry both content AND processed positional information
    """
    
    def __init__(self, hidden_dim, pos_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        
        # Message function: combines node embedding + processed positional encoding
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_embeddings, adj_matrix, processed_pos_embeddings):
        """
        Forward pass with positional-aware message passing
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            adj_matrix: [num_nodes, num_nodes]
            processed_pos_embeddings: [num_nodes, pos_dim] from PositionalEncoder
            
        Returns:
            updated_embeddings: [num_nodes, hidden_dim]
        """
        device = node_embeddings.device
        adj_matrix = adj_matrix.to(device)
        processed_pos_embeddings = processed_pos_embeddings.to(device)
        
        # Create messages with positional identity
        # This is the KEY: messages now carry sender's positional signature!
        message_input = torch.cat([node_embeddings, processed_pos_embeddings], dim=1)
        messages = self.message_mlp(message_input)
        
        # Aggregate messages from neighbors
        aggregated_messages = torch.mm(adj_matrix, messages)
        
        # Normalize by degree
        degrees = adj_matrix.sum(dim=1, keepdim=True)
        degrees = torch.clamp(degrees, min=1.0)
        aggregated_messages = aggregated_messages / degrees
        
        # Update node embeddings
        update_input = torch.cat([node_embeddings, aggregated_messages], dim=1)
        updated = self.update_mlp(update_input)
        
        # Residual connection + layer norm
        return self.layer_norm(updated + node_embeddings)

class SinCosTriadicGNN(nn.Module):
    """
    Complete GNN with sin/cos positional encodings for triadic closure
    
    ARCHITECTURE (following user's suggestions):
    1. Sin/cos positional embeddings (fixed, no eigendecomposition)
    2. 3-layer MLP to process positional embeddings  
    3. Multi-layer GNN with positional-aware message passing
    4. Edge predictor using both content and positional information
    """
    
    def __init__(self, node_feat_dim, num_nodes, pos_dim=128, hidden_dim=64, num_layers=3):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        
        # 1. Create sin/cos positional embeddings (fixed, not learned)
        self.register_buffer('pos_embeddings', 
                           create_sincos_positional_embeddings(num_nodes, pos_dim))
        
        # 2. 3-layer MLP to process positional embeddings (USER'S SUGGESTION)
        self.pos_encoder = PositionalEncoder(pos_dim, hidden_dim, hidden_dim)
        
        # 3. Node feature encoder (combines original features + processed positions)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 4. Multi-layer GNN with positional awareness
        self.gnn_layers = nn.ModuleList([
            SinCosGNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 5. Edge predictor using only node embeddings (USER'S SUGGESTION: remove positional pairs)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Only 2*node_emb (no positional pairs)
            nn.ReLU(),
            nn.Dropout(0.05),  # USER'S SUGGESTION: decreased dropout rate
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),  # USER'S SUGGESTION: decreased dropout rate
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, node_features, adj_matrix, node_indices=None):
        """
        Forward pass through sin/cos positional GNN
        
        Args:
            node_features: [num_nodes, node_feat_dim]
            adj_matrix: [num_nodes, num_nodes]
            node_indices: [num_nodes] indices for positional lookup (optional)
            
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
            processed_pos: [num_nodes, hidden_dim] processed positional embeddings
        """
        device = node_features.device
        num_nodes = node_features.shape[0]
        
        # Get positional embeddings for current nodes
        if node_indices is None:
            node_indices = torch.arange(num_nodes, device=device)
        
        # Get raw sin/cos positional embeddings
        raw_pos_emb = self.pos_embeddings[node_indices]  # [num_nodes, pos_dim]
        
        # Process positional embeddings through 3-layer MLP
        processed_pos = self.pos_encoder(raw_pos_emb)  # [num_nodes, hidden_dim]
        
        # Initialize node embeddings (original features + processed positions)
        initial_features = torch.cat([node_features, processed_pos], dim=1)
        node_embeddings = self.node_encoder(initial_features)
        
        # Multi-layer message passing with positional awareness
        for layer in self.gnn_layers:
            node_embeddings = layer(node_embeddings, adj_matrix, processed_pos)
        
        return node_embeddings, processed_pos
    
    def predict_edges(self, node_embeddings, processed_pos, edge_pairs):
        """
        Predict edges using only node embeddings
        
        USER'S INSIGHT: Node embeddings already contain positional information 
        from message passing, so we don't need explicit positional pairs.
        
        TRIADIC DETECTION: The model can detect common neighbors because
        node embeddings contain positional signatures from message passing!
        """
        # Get embeddings for source and destination nodes
        src_emb = node_embeddings[edge_pairs[:, 0]]  # [num_pairs, hidden_dim]
        dst_emb = node_embeddings[edge_pairs[:, 1]]  # [num_pairs, hidden_dim]
        
        # USER'S SUGGESTION: Only use node embeddings (no explicit positional pairs)
        # The positional information is already encoded in node embeddings through
        # the positional-aware message passing process!
        edge_features = torch.cat([
            src_emb,    # Source node embedding (contains positional info)
            dst_emb     # Destination node embedding (contains positional info)
        ], dim=1)
        
        # Predict edge probability
        return self.edge_predictor(edge_features).squeeze(-1)

def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, e_feat, n_feat, ground_truth

def create_node_mapping(g_df):
    """Create mapping from original node IDs to consecutive indices"""
    all_nodes = sorted(set(g_df['u'].tolist() + g_df['i'].tolist()))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    return node_to_idx, all_nodes

def create_adjacency_matrix(edges, num_nodes):
    """Create adjacency matrix from edge list"""
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj_matrix[u, v] = 1.0
            adj_matrix[v, u] = 1.0
    return adj_matrix

def evaluate_sincos_gnn(model, node_features, adj_matrix, ground_truth_edges, 
                       num_nodes, device, logger, balance_ratio=1.0):
    """Evaluate with balanced sampling"""
    model.eval()
    
    with torch.no_grad():
        node_embeddings, processed_pos = model(node_features, adj_matrix)
    
    # Create ground truth set
    gt_edge_set = set()
    for u, v in ground_truth_edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            gt_edge_set.add((u, v))
            gt_edge_set.add((v, u))
    
    # Generate edge pairs (sample for efficiency)
    pos_pairs = []
    neg_pairs = []
    
    max_pairs = min(3000, num_nodes * (num_nodes - 1) // 2)
    sampled_pairs = []
    
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            sampled_pairs.append((u, v))
            if len(sampled_pairs) >= max_pairs:
                break
        if len(sampled_pairs) >= max_pairs:
            break
    
    # Classify pairs
    for u, v in sampled_pairs:
        if (u, v) in gt_edge_set or (v, u) in gt_edge_set:
            pos_pairs.append([u, v])
        else:
            neg_pairs.append([u, v])
    
    if len(pos_pairs) == 0:
        return {'accuracy': 0.0, 'auc': 0.5, 'ap': 0.0}
    
    # Balanced sampling
    num_neg_samples = min(len(neg_pairs), int(len(pos_pairs) * balance_ratio))
    if num_neg_samples > 0:
        neg_indices = np.random.choice(len(neg_pairs), num_neg_samples, replace=False)
        sampled_neg_pairs = [neg_pairs[i] for i in neg_indices]
    else:
        sampled_neg_pairs = []
    
    # Combine pairs
    all_pairs = pos_pairs + sampled_neg_pairs
    all_labels = [1] * len(pos_pairs) + [0] * len(sampled_neg_pairs)
    
    if len(all_pairs) == 0:
        return {'accuracy': 0.0, 'auc': 0.5, 'ap': 0.0}
    
    # Get predictions
    edge_pairs_tensor = torch.tensor(all_pairs, dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model.predict_edges(node_embeddings, processed_pos, edge_pairs_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    
    # Calculate metrics
    y_true = np.array(all_labels)
    accuracy = accuracy_score(y_true, preds)
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, probs)
        ap = average_precision_score(y_true, probs)
    else:
        auc = 0.5
        ap = 0.0
    
    pos_ratio = len(pos_pairs) / len(all_pairs)
    logger.info(f"BALANCED Evaluation: {len(pos_pairs)} pos, {len(sampled_neg_pairs)} neg ({pos_ratio:.1%} positive)")
    
    return {'accuracy': accuracy, 'auc': auc, 'ap': ap}

def train_sincos_triadic_gnn(data_name, epochs=150, lr=0.001, hidden_dim=64, pos_dim=128):
    """Train sin/cos positional GNN"""
    
    logger = setup_logging(data_name)
    logger.info(f"🚀 Training Sin/Cos Positional GNN on {data_name}")
    logger.info(f"💡 USER'S IMPROVED INSIGHT: Sin/cos positional embeddings + 3-layer MLP")
    logger.info(f"🎯 ADVANTAGES: Stable, no eigendecomposition, well-established in NLP")
    
    # Load data
    g_df, e_feat, n_feat, ground_truth = load_triadic_data(data_name)
    logger.info(f"Dataset: {len(g_df)} edges")
    
    # Create node mapping
    node_to_idx, all_nodes = create_node_mapping(g_df)
    num_nodes = len(all_nodes)
    logger.info(f"Nodes: {num_nodes}")
    
    # Split data temporally
    timestamps = sorted(g_df['ts'].unique())
    total_ts = len(timestamps)
    train_ts = int(total_ts * 0.6)
    val_ts = int(total_ts * 0.2)
    
    train_timestamps = timestamps[:train_ts]
    val_timestamps = timestamps[train_ts:train_ts + val_ts]
    test_timestamps = timestamps[train_ts + val_ts:]
    
    logger.info(f"Train: {len(train_timestamps)} timestamps")
    logger.info(f"Val: {len(val_timestamps)} timestamps") 
    logger.info(f"Test: {len(test_timestamps)} timestamps")
    
    # Create training graph
    train_edges = g_df[g_df['ts'] < val_timestamps[0]]
    edge_list = []
    for _, row in train_edges.iterrows():
        u_idx = node_to_idx[int(row.u)]
        v_idx = node_to_idx[int(row.i)]
        edge_list.append((u_idx, v_idx))
    
    adj_matrix = create_adjacency_matrix(edge_list, num_nodes)
    logger.info(f"Training graph: {len(edge_list)} edges")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create node features
    if n_feat.shape[0] < num_nodes:
        node_features = np.zeros((num_nodes, n_feat.shape[1]), dtype=np.float32)
        node_features[:n_feat.shape[0]] = n_feat
    else:
        node_features = n_feat[:num_nodes]
    
    # Initialize model with user's suggestions
    logger.info(f"🧮 Initializing sin/cos positional embeddings...")
    model = SinCosTriadicGNN(
        node_feat_dim=node_features.shape[1],
        num_nodes=num_nodes,
        pos_dim=pos_dim,           # USER'S SUGGESTION: 64 or 128
        hidden_dim=hidden_dim,
        num_layers=3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")
    
    # Convert to tensors
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32, device=device)
    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    
    best_val_ap = 0.0
    patience_counter = 0
    max_patience = 30
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Train on each timestamp
        for ts in train_timestamps:
            edges_at_ts = g_df[g_df['ts'] == ts]
            if len(edges_at_ts) == 0:
                continue
                
            # Get ground truth edges
            gt_edges = []
            for _, row in edges_at_ts.iterrows():
                u_idx = node_to_idx[int(row.u)]
                v_idx = node_to_idx[int(row.i)]
                gt_edges.append((u_idx, v_idx))
            
            # Create training pairs
            pos_pairs = gt_edges
            neg_pairs = []
            
            # Sample negative pairs
            num_neg_needed = len(pos_pairs) * 2
            attempts = 0
            while len(neg_pairs) < num_neg_needed and attempts < num_neg_needed * 10:
                u = np.random.randint(0, num_nodes)
                v = np.random.randint(0, num_nodes)
                
                if u != v and (u, v) not in gt_edges and (v, u) not in gt_edges:
                    if adj_matrix[u, v] == 0:
                        neg_pairs.append((u, v))
                
                attempts += 1
            
            if len(neg_pairs) == 0:
                continue
            
            # Combine pairs
            all_pairs = pos_pairs + neg_pairs
            all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
            
            # Convert to tensors
            edge_pairs_tensor = torch.tensor(all_pairs, dtype=torch.long, device=device)
            labels_tensor = torch.tensor(all_labels, dtype=torch.float32, device=device)
            
            # Forward pass
            node_embeddings, processed_pos = model(node_features_tensor, adj_matrix_tensor)
            logits = model.predict_edges(node_embeddings, processed_pos, edge_pairs_tensor)
            loss = criterion(logits, labels_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(1, num_batches)
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            val_ts = val_timestamps[0]
            val_edges = g_df[g_df['ts'] == val_ts]
            val_gt_edges = []
            for _, row in val_edges.iterrows():
                u_idx = node_to_idx[int(row.u)]
                v_idx = node_to_idx[int(row.i)]
                val_gt_edges.append((u_idx, v_idx))
            
            val_metrics = evaluate_sincos_gnn(
                model, node_features_tensor, adj_matrix_tensor,
                val_gt_edges, num_nodes, device, logger
            )
            
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
                       f"Val Acc={val_metrics['accuracy']:.4f}, "
                       f"Val AUC={val_metrics['auc']:.4f}, "
                       f"Val AP={val_metrics['ap']:.4f}")
            
            if val_metrics['ap'] > best_val_ap:
                best_val_ap = val_metrics['ap']
                patience_counter = 0
                logger.info(f"🏆 New best Val AP: {best_val_ap:.4f}")
            else:
                patience_counter += 1
            
            scheduler.step(val_metrics['ap'])
            
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        else:
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")
    
    # Final test evaluation
    logger.info("🎯 Final test evaluation...")
    test_ts = test_timestamps[0]
    test_edges = g_df[g_df['ts'] == test_ts]
    test_gt_edges = []
    for _, row in test_edges.iterrows():
        u_idx = node_to_idx[int(row.u)]
        v_idx = node_to_idx[int(row.i)]
        test_gt_edges.append((u_idx, v_idx))
    
    test_metrics = evaluate_sincos_gnn(
        model, node_features_tensor, adj_matrix_tensor,
        test_gt_edges, num_nodes, device, logger
    )
    
    logger.info("🎯 Final Test Results:")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  Test AP: {test_metrics['ap']:.4f}")
    
    print("=" * 70)
    print("FINAL RESULTS (Sin/Cos Positional GNN)")
    print("=" * 70)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC:      {test_metrics['auc']:.4f}")
    print(f"Test AP:       {test_metrics['ap']:.4f}")
    print(f"Best Val AP:   {best_val_ap:.4f}")
    print("=" * 70)
    print("💡 USER'S IMPROVED SOLUTION:")
    print("   ✅ Sin/cos positional embeddings (stable, no eigendecomposition)")
    print("   ✅ 3-layer MLP for positional processing")
    print("   ✅ Node identity preserved during message passing")
    print("=" * 70)
    
    return test_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='triadic_perfect_long_dense')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--pos_dim', type=int, default=128)
    
    args = parser.parse_args()
    
    print("🚀 SIN/COS POSITIONAL GNN")
    print("💡 User's improved insight: Stable positional embeddings + MLP processing")
    
    train_sincos_triadic_gnn(
        data_name=args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        pos_dim=args.pos_dim
    )

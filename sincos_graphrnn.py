"""
Sin/Cos Positional GraphRNN for Dynamic Triadic Closure

USER'S KEY INSIGHTS:
1. Remove MLP for positional encoding - use raw sin/cos embeddings directly
2. This is a DYNAMIC graph problem - use GraphRNN to predict next graph state
3. Combine positional encoding (for node identity) with sequential modeling (for temporal dynamics)

APPROACH:
- Sin/cos positional embeddings (no MLP processing)
- GraphRNN architecture for sequential graph state prediction
- Positional-aware message passing at each timestamp
- Predict edges for next timestamp given current graph + positional info
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
            logging.FileHandler(f'log/sincos_graphrnn_{data_name}_{int(time.time())}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_sincos_positional_embeddings(num_nodes, pos_dim=128):
    """
    Create sin/cos positional embeddings (USER'S SUGGESTION: no MLP processing)
    
    Args:
        num_nodes: Number of nodes in the graph
        pos_dim: Dimension of positional embeddings (should be even)
        
    Returns:
        torch.Tensor: [num_nodes, pos_dim] raw positional embeddings
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
    
    print(f"   ✅ Created raw sin/cos positional embeddings: {pos_emb.shape}")
    print(f"   📊 Sample embeddings for first 3 nodes:")
    for i in range(min(3, num_nodes)):
        print(f"      Node {i}: [{pos_emb[i, :6].tolist()}, ...]")
    
    return pos_emb

class PositionalGNNLayer(nn.Module):
    """
    GNN layer with raw sin/cos positional encoding for node identity
    
    USER'S INSIGHTS:
    - Use raw positional embeddings directly (no MLP processing)
    - GIN-style aggregation (no degree normalization) for reduced complexity
    """
    
    def __init__(self, hidden_dim, pos_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        
        # Message function: combines node embedding + RAW positional encoding
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_embeddings, adj_matrix, raw_pos_embeddings):
        """
        Forward pass with raw positional-aware message passing
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            adj_matrix: [num_nodes, num_nodes]
            raw_pos_embeddings: [num_nodes, pos_dim] RAW sin/cos embeddings
            
        Returns:
            updated_embeddings: [num_nodes, hidden_dim]
        """
        device = node_embeddings.device
        adj_matrix = adj_matrix.to(device)
        raw_pos_embeddings = raw_pos_embeddings.to(device)
        
        # Create messages with RAW positional identity (USER'S INSIGHT: no MLP)
        message_input = torch.cat([node_embeddings, raw_pos_embeddings], dim=1)
        messages = self.message_mlp(message_input)
        
        # GIN-style aggregation: simple sum (no degree normalization)
        aggregated_messages = torch.mm(adj_matrix, messages)
        
        # Update node embeddings
        update_input = torch.cat([node_embeddings, aggregated_messages], dim=1)
        updated = self.update_mlp(update_input)
        
        # Residual connection + layer norm
        return self.layer_norm(updated + node_embeddings)

class SinCosGraphRNN(nn.Module):
    """
    GraphRNN with Sin/Cos Positional Encodings for Dynamic Triadic Closure
    
    USER'S INSIGHTS:
    1. This is a DYNAMIC graph problem - predict next graph state
    2. Use raw sin/cos positional embeddings (no MLP)
    3. Combine node identity (positional) with temporal dynamics (RNN)
    
    ARCHITECTURE:
    1. Raw sin/cos positional embeddings for node identity
    2. GNN encoder with positional awareness
    3. LSTM for temporal sequence modeling
    4. Edge predictor for next timestamp
    """
    
    def __init__(self, node_feat_dim, num_nodes, pos_dim=64, hidden_dim=128, 
                 rnn_layers=2, gnn_layers=2):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        
        # 1. Raw sin/cos positional embeddings (USER'S INSIGHT: no MLP processing)
        self.register_buffer('pos_embeddings', 
                           create_sincos_positional_embeddings(num_nodes, pos_dim))
        
        # 2. Node feature encoder (combines original features + raw positions)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim + pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Multi-layer GNN with raw positional awareness
        self.gnn_layers = nn.ModuleList([
            PositionalGNNLayer(hidden_dim, pos_dim)
            for _ in range(gnn_layers)
        ])
        
        # 4. Graph-level representation (for RNN input)
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 5. LSTM for temporal sequence modeling (USER'S INSIGHT: dynamic graphs)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=0.1 if rnn_layers > 1 else 0
        )
        
        # 6. Edge predictor for next timestamp
        # USER'S INSIGHT: Symmetric features for undirected graphs
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),  # (src+dst) + |src-dst| + temporal_context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def encode_graph_state(self, node_features, adj_matrix):
        """
        Encode current graph state using positional GNN
        
        Args:
            node_features: [num_nodes, node_feat_dim]
            adj_matrix: [num_nodes, num_nodes]
            
        Returns:
            graph_embedding: [hidden_dim] graph-level representation
            node_embeddings: [num_nodes, hidden_dim] node embeddings
        """
        device = node_features.device
        num_nodes = node_features.shape[0]
        
        # Get raw positional embeddings
        raw_pos_emb = self.pos_embeddings[:num_nodes]  # [num_nodes, pos_dim]
        
        # Initialize node embeddings (original features + RAW positions)
        initial_features = torch.cat([node_features, raw_pos_emb], dim=1)
        node_embeddings = self.node_encoder(initial_features)
        
        # Multi-layer message passing with RAW positional awareness
        for layer in self.gnn_layers:
            node_embeddings = layer(node_embeddings, adj_matrix, raw_pos_emb)
        
        # Graph-level pooling (mean pooling)
        graph_embedding = self.graph_pooling(node_embeddings.mean(dim=0))  # [hidden_dim]
        
        return graph_embedding, node_embeddings
    
    def forward_sequence(self, graph_sequence):
        """
        Process sequence of graph states
        
        Args:
            graph_sequence: List of (node_features, adj_matrix) for each timestamp
            
        Returns:
            temporal_contexts: [seq_len, hidden_dim] temporal context for each timestamp
            all_node_embeddings: List of [num_nodes, hidden_dim] for each timestamp
        """
        graph_embeddings = []
        all_node_embeddings = []
        
        # Encode each graph state
        for node_features, adj_matrix in graph_sequence:
            graph_emb, node_emb = self.encode_graph_state(node_features, adj_matrix)
            graph_embeddings.append(graph_emb)
            all_node_embeddings.append(node_emb)
        
        # Stack for LSTM input
        graph_sequence_tensor = torch.stack(graph_embeddings, dim=0).unsqueeze(0)  # [1, seq_len, hidden_dim]
        
        # LSTM for temporal modeling
        lstm_out, _ = self.lstm(graph_sequence_tensor)  # [1, seq_len, hidden_dim]
        temporal_contexts = lstm_out.squeeze(0)  # [seq_len, hidden_dim]
        
        return temporal_contexts, all_node_embeddings
    
    def predict_next_edges(self, node_embeddings, temporal_context, edge_pairs):
        """
        Predict edges for next timestamp given current node embeddings and temporal context
        
        Args:
            node_embeddings: [num_nodes, hidden_dim] current node embeddings
            temporal_context: [hidden_dim] temporal context from LSTM
            edge_pairs: [num_pairs, 2] pairs to predict
            
        Returns:
            edge_logits: [num_pairs] edge prediction logits
        """
        # Get embeddings for edge pairs
        src_emb = node_embeddings[edge_pairs[:, 0]]  # [num_pairs, hidden_dim]
        dst_emb = node_embeddings[edge_pairs[:, 1]]  # [num_pairs, hidden_dim]
        
        # Expand temporal context for all pairs
        num_pairs = edge_pairs.shape[0]
        temporal_expanded = temporal_context.unsqueeze(0).expand(num_pairs, -1)  # [num_pairs, hidden_dim]
        
        # USER'S INSIGHT: Symmetric edge features for undirected graphs
        edge_sum = src_emb + dst_emb                    # [num_pairs, hidden_dim] - symmetric sum
        edge_diff = torch.abs(src_emb - dst_emb)        # [num_pairs, hidden_dim] - symmetric difference
        
        edge_features = torch.cat([
            edge_sum,             # Symmetric sum of node embeddings
            edge_diff,            # Symmetric absolute difference
            temporal_expanded     # Temporal context from LSTM
        ], dim=1)
        
        # Predict edge probability
        return self.edge_predictor(edge_features).squeeze(-1)

def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    # Load FIXED ground truth (converted from edge indices to timestamps)
    try:
        with open(f'./processed/{data_name}/ml_{data_name}_gt_fixed.json', 'r') as f:
            ground_truth = json.load(f)
        print("✅ Using FIXED ground truth with proper timestamps")
    except FileNotFoundError:
        print("❌ Fixed ground truth not found, creating it...")
        # Create fixed ground truth on the fly
        with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
            ground_truth_raw = json.load(f)
        
        ground_truth = {}
        for edge_idx_str, parent_indices in ground_truth_raw.items():
            edge_idx = int(edge_idx_str)
            if edge_idx < len(g_df):
                edge_row = g_df.iloc[edge_idx]
                timestamp = str(edge_row.ts)
                u, v = int(edge_row.u), int(edge_row.i)
                
                if timestamp not in ground_truth:
                    ground_truth[timestamp] = []
                ground_truth[timestamp].append([u, v])
        
        print("✅ Created fixed ground truth on the fly")
    
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

def create_graph_sequence(g_df, node_to_idx, n_feat, timestamps):
    """
    Create sequence of graph states for GraphRNN
    
    Args:
        g_df: Graph dataframe
        node_to_idx: Node mapping
        n_feat: Node features
        timestamps: List of timestamps
        
    Returns:
        graph_sequence: List of (node_features, adj_matrix) for each timestamp
    """
    num_nodes = len(node_to_idx)
    graph_sequence = []
    
    # Prepare node features
    if n_feat.shape[0] < num_nodes:
        node_features = np.zeros((num_nodes, n_feat.shape[1]), dtype=np.float32)
        node_features[:n_feat.shape[0]] = n_feat
    else:
        node_features = n_feat[:num_nodes]
    
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
    
    # Create cumulative graph states
    cumulative_edges = []
    
    for ts in timestamps:
        # Add edges from this timestamp
        edges_at_ts = g_df[g_df['ts'] == ts]
        for _, row in edges_at_ts.iterrows():
            u_idx = node_to_idx[int(row.u)]
            v_idx = node_to_idx[int(row.i)]
            cumulative_edges.append((u_idx, v_idx))
        
        # Create adjacency matrix for cumulative graph
        adj_matrix = create_adjacency_matrix(cumulative_edges, num_nodes)
        adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
        
        graph_sequence.append((node_features_tensor, adj_matrix_tensor))
    
    return graph_sequence

def evaluate_graphrnn_sequence(model, graph_sequence, ground_truth_by_ts, timestamps, 
                              device, logger, balance_ratio=1.0, eval_timestamps=None):
    """
    Evaluate GraphRNN on sequence prediction
    
    USER'S INSIGHT: Predict next graph state at each timestamp
    
    Args:
        eval_timestamps: If provided, only evaluate transitions TO these timestamps
                        This allows proper validation/test evaluation without data leakage
    """
    model.eval()
    
    with torch.no_grad():
        # Get temporal contexts and node embeddings for all timestamps
        temporal_contexts, all_node_embeddings = model.forward_sequence(
            [(nf.to(device), adj.to(device)) for nf, adj in graph_sequence]
        )
    
    all_predictions = []
    all_labels = []
    
    # Evaluate predictions for each timestamp (except the last one)
    for i in range(len(timestamps) - 1):
        current_ts = timestamps[i]
        next_ts = timestamps[i + 1]
        
        # USER'S INSIGHT: Only evaluate transitions TO validation/test timestamps
        if eval_timestamps is not None and next_ts not in eval_timestamps:
            continue
        
        if next_ts not in ground_truth_by_ts:
            continue
            
        # Get ground truth edges for next timestamp
        gt_edges = ground_truth_by_ts[next_ts]
        if len(gt_edges) == 0:
            continue
        
        # Current graph state
        node_embeddings = all_node_embeddings[i].to(device)
        temporal_context = temporal_contexts[i].to(device)
        num_nodes = node_embeddings.shape[0]
        
        # Create ground truth set
        gt_edge_set = set()
        for u, v in gt_edges:
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                gt_edge_set.add((u, v))
                gt_edge_set.add((v, u))
        
        # Generate edge pairs for evaluation
        pos_pairs = []
        neg_pairs = []
        
        # Sample pairs for efficiency
        max_pairs = min(1000, num_nodes * (num_nodes - 1) // 2)
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
            continue
        
        # Balanced sampling
        num_neg_samples = min(len(neg_pairs), int(len(pos_pairs) * balance_ratio))
        if num_neg_samples > 0:
            neg_indices = np.random.choice(len(neg_pairs), num_neg_samples, replace=False)
            sampled_neg_pairs = [neg_pairs[j] for j in neg_indices]
        else:
            sampled_neg_pairs = []
        
        # Combine pairs
        all_pairs = pos_pairs + sampled_neg_pairs
        all_labels_ts = [1] * len(pos_pairs) + [0] * len(sampled_neg_pairs)
        
        if len(all_pairs) == 0:
            continue
        
        # Get predictions
        edge_pairs_tensor = torch.tensor(all_pairs, dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = model.predict_next_edges(node_embeddings, temporal_context, edge_pairs_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        all_predictions.extend(probs)
        all_labels.extend(all_labels_ts)
        
        logger.info(f"Timestamp {current_ts}→{next_ts}: {len(pos_pairs)} pos, {len(sampled_neg_pairs)} neg")
    
    if len(all_predictions) == 0:
        return {'accuracy': 0.0, 'auc': 0.5, 'ap': 0.0}
    
    # Calculate overall metrics
    y_true = np.array(all_labels)
    y_scores = np.array(all_predictions)
    y_pred = (y_scores > 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    else:
        auc = 0.5
        ap = 0.0
    
    pos_ratio = np.sum(y_true) / len(y_true)
    logger.info(f"SEQUENCE Evaluation: {np.sum(y_true)} pos, {len(y_true) - np.sum(y_true)} neg ({pos_ratio:.1%} positive)")
    
    return {'accuracy': accuracy, 'auc': auc, 'ap': ap}

def train_sincos_graphrnn(data_name, epochs=200, lr=0.001, hidden_dim=128, pos_dim=64):
    """Train Sin/Cos Positional GraphRNN"""
    
    logger = setup_logging(data_name)
    logger.info(f"🚀 Training Sin/Cos Positional GraphRNN on {data_name}")
    logger.info(f"💡 USER'S INSIGHTS: Raw sin/cos embeddings + Dynamic graph modeling")
    logger.info(f"🎯 APPROACH: Predict next graph state at each timestamp")
    
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
    
    # Create graph sequences
    logger.info(f"🔄 Creating graph sequences for dynamic modeling...")
    train_sequence = create_graph_sequence(g_df, node_to_idx, n_feat, train_timestamps)
    val_sequence = create_graph_sequence(g_df, node_to_idx, n_feat, timestamps[:train_ts + val_ts])
    test_sequence = create_graph_sequence(g_df, node_to_idx, n_feat, timestamps)
    
    # Organize ground truth by timestamp (now using FIXED format)
    ground_truth_by_ts = {}
    for ts_str, edge_list in ground_truth.items():
        ts = int(ts_str)
        gt_edges = []
        
        # Ground truth now contains actual (u, v) pairs, not indices
        for edge in edge_list:
            u, v = edge[0], edge[1]
            if u in node_to_idx and v in node_to_idx:
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                gt_edges.append((u_idx, v_idx))
        
        ground_truth_by_ts[ts] = gt_edges
    
    print(f"✅ Loaded ground truth for {len(ground_truth_by_ts)} timestamps")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    logger.info(f"🧮 Initializing Sin/Cos Positional GraphRNN...")
    model = SinCosGraphRNN(
        node_feat_dim=n_feat.shape[1],
        num_nodes=num_nodes,
        pos_dim=pos_dim,           # USER'S SUGGESTION: 64 or 128
        hidden_dim=hidden_dim,
        rnn_layers=2,
        gnn_layers=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    
    best_val_ap = 0.0
    patience_counter = 0
    max_patience = 40
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Train on each transition (predict next timestamp)
        for i in range(len(train_timestamps) - 1):
            current_ts = train_timestamps[i]
            next_ts = train_timestamps[i + 1]
            
            if next_ts not in ground_truth_by_ts:
                continue
                
            gt_edges = ground_truth_by_ts[next_ts]
            if len(gt_edges) == 0:
                continue
            
            # Recompute forward pass for this specific transition
            current_sequence = train_sequence[:i+1]  # Sequence up to current timestamp
            temporal_contexts, all_node_embeddings = model.forward_sequence(
                [(nf.to(device), adj.to(device)) for nf, adj in current_sequence]
            )
            
            # Current state (last in sequence)
            node_embeddings = all_node_embeddings[-1]
            temporal_context = temporal_contexts[-1]
            
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
            logits = model.predict_next_edges(node_embeddings, temporal_context, edge_pairs_tensor)
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
            # USER'S INSIGHT: Provide full context but only evaluate on validation transitions
            val_metrics = evaluate_graphrnn_sequence(
                model, val_sequence, ground_truth_by_ts, 
                timestamps[:train_ts + val_ts], device, logger,
                eval_timestamps=set(val_timestamps)  # Only evaluate transitions TO validation timestamps
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
    # USER'S INSIGHT: Provide full context but only evaluate on test transitions
    test_metrics = evaluate_graphrnn_sequence(
        model, test_sequence, ground_truth_by_ts, timestamps, device, logger,
        eval_timestamps=set(test_timestamps)  # Only evaluate transitions TO test timestamps
    )
    
    logger.info("🎯 Final Test Results:")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  Test AP: {test_metrics['ap']:.4f}")
    
    print("=" * 70)
    print("FINAL RESULTS (Sin/Cos Positional GraphRNN)")
    print("=" * 70)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC:      {test_metrics['auc']:.4f}")
    print(f"Test AP:       {test_metrics['ap']:.4f}")
    print(f"Best Val AP:   {best_val_ap:.4f}")
    print("=" * 70)
    print("💡 USER'S INSIGHTS IMPLEMENTED:")
    print("   ✅ Raw sin/cos positional embeddings (no MLP)")
    print("   ✅ Dynamic graph modeling with GraphRNN")
    print("   ✅ Sequential prediction of next graph state")
    print("   ✅ Node identity + temporal dynamics")
    print("=" * 70)
    
    return test_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='triadic_perfect_long_dense')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--pos_dim', type=int, default=64)
    
    args = parser.parse_args()
    
    print("🚀 SIN/COS POSITIONAL GRAPHRNN")
    print("💡 User's insights: Raw positional embeddings + Dynamic graph modeling")
    
    train_sincos_graphrnn(
        data_name=args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        pos_dim=args.pos_dim
    )

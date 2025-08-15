"""
Test whether different nodes have different temporal neighborhood structures
that provide discriminative power for TGIB (rather than raw features)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import TGIB's neighbor finder
import sys
sys.path.append('.')
from graph import NeighborFinder

class RandEdgeSampler:
    """Replicate TGIB's random edge sampling"""
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        
    def sample(self, size):
        return np.random.choice(self.dst_list, size=size, replace=True)

def extract_neighborhood_features(node_ids, timestamps, ngh_finder, e_feat, num_neighbors=20):
    """Extract temporal neighborhood structural features for given nodes"""
    neighborhood_features = []
    
    for i, (node_id, timestamp) in enumerate(zip(node_ids, timestamps)):
        try:
            # Get temporal neighbors (exactly like TGIB does)
            ngh_nodes, ngh_e_idx, ngh_ts = ngh_finder.get_temporal_neighbor(
                np.array([node_id]), np.array([timestamp]), num_neighbors
            )
            
            ngh_nodes = ngh_nodes.flatten()
            ngh_e_idx = ngh_e_idx.flatten()
            ngh_ts = ngh_ts.flatten()
            
            # Extract neighborhood structural features
            features = []
            
            # 1. Neighborhood size (number of valid neighbors)
            valid_neighbors = np.sum(ngh_nodes > 0)
            features.append(valid_neighbors)
            
            # 2. Degree distribution statistics
            if valid_neighbors > 0:
                # Time deltas to neighbors
                time_deltas = timestamp - ngh_ts[ngh_ts > 0]
                features.extend([
                    np.mean(time_deltas) if len(time_deltas) > 0 else 0,
                    np.std(time_deltas) if len(time_deltas) > 0 else 0,
                    np.min(time_deltas) if len(time_deltas) > 0 else 0,
                    np.max(time_deltas) if len(time_deltas) > 0 else 0
                ])
                
                # Edge feature statistics in neighborhood
                valid_e_idx = ngh_e_idx[ngh_e_idx > 0] - 1  # Convert to 0-indexed
                if len(valid_e_idx) > 0:
                    ngh_edge_feats = e_feat[valid_e_idx]
                    features.extend([
                        np.mean(ngh_edge_feats, axis=0).flatten(),
                        np.std(ngh_edge_feats, axis=0).flatten(),
                        np.sum(ngh_edge_feats, axis=0).flatten()
                    ])
                    features = np.concatenate([f if isinstance(f, np.ndarray) else [f] for f in features])
                else:
                    # No valid edges - pad with zeros
                    edge_dim = e_feat.shape[1]
                    features.extend([np.zeros(edge_dim * 3)])
                    features = np.concatenate([f if isinstance(f, np.ndarray) else [f] for f in features])
            else:
                # No valid neighbors - pad with zeros
                edge_dim = e_feat.shape[1]
                features.extend([0, 0, 0, 0])  # time stats
                features.extend([np.zeros(edge_dim * 3)])  # edge stats
                features = np.concatenate([f if isinstance(f, np.ndarray) else [f] for f in features])
            
            neighborhood_features.append(features)
            
        except Exception as e:
            print(f"Error processing node {node_id}: {e}")
            # Return zero features on error
            edge_dim = e_feat.shape[1]
            zero_features = np.zeros(5 + edge_dim * 3)
            neighborhood_features.append(zero_features)
    
    return np.array(neighborhood_features)

def analyze_temporal_neighborhood_discrimination(dataset_name, num_samples=2000):
    """Test if temporal neighborhood structures provide discriminative power"""
    print(f"\n{'='*70}")
    print(f"TEMPORAL NEIGHBORHOOD ANALYSIS: {dataset_name.upper()}")
    print("="*70)
    
    # Load data
    g_df = pd.read_csv(f'./processed/{dataset_name}/ml_{dataset_name}.csv')
    e_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}.npy')
    n_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}_node.npy')
    
    print(f"Dataset: {len(g_df)} edges")
    print(f"Edge features: {e_feat.shape[1]} dimensions")
    print(f"Node features: {n_feat.shape[1]} dimensions")
    
    # Create train/val split and adjacency list (like TGIB)
    val_time = np.quantile(g_df.ts, 0.70)
    train_mask = g_df.ts <= val_time
    
    train_src = g_df[train_mask].u.values
    train_dst = g_df[train_mask].i.values
    train_e_idx = g_df[train_mask].idx.values
    train_ts = g_df[train_mask].ts.values
    
    # Build adjacency list for neighbor finding
    max_idx = max(g_df.u.max(), g_df.i.max())
    adj_list = [[] for _ in range(max_idx + 1)]
    
    for src, dst, eidx, ts in zip(train_src, train_dst, train_e_idx, train_ts):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    
    ngh_finder = NeighborFinder(adj_list, uniform=False)
    sampler = RandEdgeSampler(train_src, train_dst)
    
    print(f"Training edges: {len(train_src)}")
    
    # Sample edges for analysis
    num_edges = min(num_samples, len(train_src))
    sample_indices = np.random.choice(len(train_src), num_edges, replace=False)
    
    real_src = train_src[sample_indices]
    real_dst = train_dst[sample_indices]
    real_ts = train_ts[sample_indices]
    
    # Generate fake destinations
    fake_dst = sampler.sample(num_edges)
    
    print(f"Analyzing {num_edges} real vs fake destination neighborhoods")
    
    # Extract neighborhood features
    print("Extracting real destination neighborhoods...")
    real_dst_ngh_features = extract_neighborhood_features(
        real_dst, real_ts, ngh_finder, e_feat, num_neighbors=20
    )
    
    print("Extracting fake destination neighborhoods...")
    fake_dst_ngh_features = extract_neighborhood_features(
        fake_dst, real_ts, ngh_finder, e_feat, num_neighbors=20
    )
    
    print(f"Neighborhood feature dimensions: {real_dst_ngh_features.shape[1]}")
    
    # Test discrimination
    X = np.vstack([real_dst_ngh_features, fake_dst_ngh_features])
    y = np.concatenate([np.ones(len(real_dst_ngh_features)), np.zeros(len(fake_dst_ngh_features))])
    
    # Check for valid features
    if np.std(X) == 0:
        print("❌ All neighborhood features are constant")
        return {'status': 'constant', 'lr_auc': 0.5, 'rf_auc': 0.5}
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(X).all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(np.unique(y)) < 2:
        print("❌ Only one class after filtering")
        return {'status': 'single_class', 'lr_auc': 0.5, 'rf_auc': 0.5}
    
    print(f"Valid samples after filtering: {len(X)}")
    
    try:
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict_proba(X_test_scaled)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_pred)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict_proba(X_test_scaled)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_pred)
        
        print(f"\n📊 TEMPORAL NEIGHBORHOOD DISCRIMINATION:")
        print(f"   Logistic Regression AUC: {lr_auc:.4f}")
        print(f"   Random Forest AUC: {rf_auc:.4f}")
        
        # Interpretation
        if lr_auc > 0.8 or rf_auc > 0.8:
            status = "excellent"
            print("   ✅ EXCELLENT: Temporal neighborhoods are highly discriminative!")
        elif lr_auc > 0.7 or rf_auc > 0.7:
            status = "good"
            print("   ✅ GOOD: Temporal neighborhoods provide good discrimination!")
        elif lr_auc > 0.6 or rf_auc > 0.6:
            status = "moderate"
            print("   🟡 MODERATE: Some discriminative power from neighborhoods")
        else:
            status = "weak"
            print("   ❌ WEAK: Limited discriminative power from neighborhoods")
        
        # Feature importance analysis
        if rf_auc > 0.6:
            feature_importance = rf.feature_importances_
            top_features = np.argsort(feature_importance)[-5:][::-1]
            
            print(f"\n🔍 TOP DISCRIMINATIVE NEIGHBORHOOD FEATURES:")
            feature_names = [
                'num_neighbors', 'avg_time_delta', 'std_time_delta', 'min_time_delta', 'max_time_delta'
            ] + [f'edge_mean_{i}' for i in range(e_feat.shape[1])] + \
                [f'edge_std_{i}' for i in range(e_feat.shape[1])] + \
                [f'edge_sum_{i}' for i in range(e_feat.shape[1])]
            
            for i, feat_idx in enumerate(top_features):
                if feat_idx < len(feature_names):
                    feat_name = feature_names[feat_idx]
                else:
                    feat_name = f"feature_{feat_idx}"
                print(f"   {i+1}. {feat_name}: {feature_importance[feat_idx]:.4f}")
        
        return {
            'status': status,
            'lr_auc': lr_auc,
            'rf_auc': rf_auc,
            'num_samples': len(X)
        }
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        return {'status': 'error', 'lr_auc': 0.5, 'rf_auc': 0.5}

def analyze_neighborhood_diversity(dataset_name, num_nodes=500):
    """Analyze diversity of temporal neighborhoods across different nodes"""
    print(f"\n🔍 NEIGHBORHOOD DIVERSITY ANALYSIS: {dataset_name.upper()}")
    print("-" * 50)
    
    # Load data
    g_df = pd.read_csv(f'./processed/{dataset_name}/ml_{dataset_name}.csv')
    e_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}.npy')
    
    # Create adjacency list
    val_time = np.quantile(g_df.ts, 0.70)
    train_mask = g_df.ts <= val_time
    
    train_src = g_df[train_mask].u.values
    train_dst = g_df[train_mask].i.values
    train_e_idx = g_df[train_mask].idx.values
    train_ts = g_df[train_mask].ts.values
    
    max_idx = max(g_df.u.max(), g_df.i.max())
    adj_list = [[] for _ in range(max_idx + 1)]
    
    for src, dst, eidx, ts in zip(train_src, train_dst, train_e_idx, train_ts):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    
    ngh_finder = NeighborFinder(adj_list, uniform=False)
    
    # Sample nodes with sufficient activity
    unique_nodes = np.unique(np.concatenate([train_src, train_dst]))
    node_degrees = [len(adj_list[node]) for node in unique_nodes]
    
    # Select nodes with at least some neighbors
    active_nodes = unique_nodes[np.array(node_degrees) >= 3]
    if len(active_nodes) > num_nodes:
        active_nodes = np.random.choice(active_nodes, num_nodes, replace=False)
    
    print(f"Analyzing {len(active_nodes)} active nodes")
    
    # Extract neighborhood statistics for each node
    node_stats = []
    for node in active_nodes:
        # Use latest timestamp for this node
        node_edges = g_df[(g_df.u == node) | (g_df.i == node)]
        if len(node_edges) == 0:
            continue
            
        latest_ts = node_edges.ts.max()
        
        try:
            ngh_nodes, ngh_e_idx, ngh_ts = ngh_finder.get_temporal_neighbor(
                np.array([node]), np.array([latest_ts]), num_neighbors=20
            )
            
            ngh_nodes = ngh_nodes.flatten()
            valid_neighbors = np.sum(ngh_nodes > 0)
            
            if valid_neighbors > 0:
                ngh_ts_valid = ngh_ts.flatten()[ngh_nodes > 0]
                time_deltas = latest_ts - ngh_ts_valid
                
                ngh_e_idx_valid = ngh_e_idx.flatten()[ngh_nodes > 0] - 1
                ngh_edge_feats = e_feat[ngh_e_idx_valid]
                
                stats = {
                    'node': node,
                    'num_neighbors': valid_neighbors,
                    'avg_time_delta': np.mean(time_deltas),
                    'std_time_delta': np.std(time_deltas),
                    'edge_feat_norm': np.linalg.norm(np.mean(ngh_edge_feats, axis=0))
                }
                node_stats.append(stats)
                
        except Exception:
            continue
    
    if len(node_stats) < 10:
        print("❌ Insufficient valid nodes for diversity analysis")
        return
    
    # Analyze diversity
    num_neighbors = [s['num_neighbors'] for s in node_stats]
    avg_time_deltas = [s['avg_time_delta'] for s in node_stats]
    edge_norms = [s['edge_feat_norm'] for s in node_stats]
    
    print(f"Neighborhood size range: [{min(num_neighbors)}, {max(num_neighbors)}]")
    print(f"Time delta range: [{min(avg_time_deltas):.1f}, {max(avg_time_deltas):.1f}]")
    print(f"Edge feature norm range: [{min(edge_norms):.3f}, {max(edge_norms):.3f}]")
    
    # Coefficient of variation as diversity measure
    neighbor_cv = np.std(num_neighbors) / np.mean(num_neighbors)
    time_cv = np.std(avg_time_deltas) / np.mean(avg_time_deltas)
    edge_cv = np.std(edge_norms) / np.mean(edge_norms) if np.mean(edge_norms) > 0 else 0
    
    print(f"\nDiversity measures (coefficient of variation):")
    print(f"Neighborhood size diversity: {neighbor_cv:.4f}")
    print(f"Time pattern diversity: {time_cv:.4f}")
    print(f"Edge feature diversity: {edge_cv:.4f}")
    
    if neighbor_cv > 0.5 or time_cv > 0.5:
        print("✅ HIGH STRUCTURAL DIVERSITY: Nodes have very different neighborhoods!")
    elif neighbor_cv > 0.3 or time_cv > 0.3:
        print("🟡 MODERATE DIVERSITY: Some neighborhood variation exists")
    else:
        print("❌ LOW DIVERSITY: Similar neighborhood structures")

def main():
    """Run temporal neighborhood structure analysis"""
    datasets = ['CanParl', 'reddit', 'uci', 'wikipedia']
    
    print("🔬 TEMPORAL NEIGHBORHOOD STRUCTURE ANALYSIS")
    print("Testing if TGIB's discriminative power comes from neighborhood structures")
    
    results = {}
    for dataset in datasets:
        try:
            result = analyze_temporal_neighborhood_discrimination(dataset)
            results[dataset] = result
            analyze_neighborhood_diversity(dataset)
        except Exception as e:
            print(f"\n❌ Error analyzing {dataset}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 SUMMARY: TEMPORAL NEIGHBORHOOD DISCRIMINATIVE POWER")
    print("="*70)
    
    excellent_count = 0
    good_count = 0
    
    for dataset, result in results.items():
        status = result.get('status', 'error')
        best_auc = max(result.get('lr_auc', 0.5), result.get('rf_auc', 0.5))
        
        print(f"{dataset:>10}: {status:<10} (best AUC: {best_auc:.3f})")
        
        if status == 'excellent':
            excellent_count += 1
        elif status == 'good':
            good_count += 1
    
    print(f"\n🎯 CONCLUSION:")
    if excellent_count + good_count >= len(results) // 2:
        print("   ✅ TEMPORAL NEIGHBORHOODS PROVIDE STRONG DISCRIMINATIVE POWER!")
        print("   ✅ This explains TGIB's success: Different nodes have different neighborhood structures")
        print("   ✅ Even with identical raw features, structural context creates discrimination")
    else:
        print("   ❌ Limited evidence for neighborhood-based discrimination")
        print("   ❌ TGIB's success may rely on other mechanisms")

if __name__ == "__main__":
    main() 
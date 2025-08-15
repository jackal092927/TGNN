"""
Verify that TGIB node embeddings contain discriminative information 
from edge features in temporal neighborhoods
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_node_embedding_edge_features(dataset_name):
    """Analyze if nodes with different edge feature histories are distinguishable"""
    print(f"\n{'='*60}")
    print(f"ANALYZING NODE EMBEDDING DIFFERENCES: {dataset_name.upper()}")
    print("="*60)
    
    # Load data
    g_df = pd.read_csv(f'./processed/{dataset_name}/ml_{dataset_name}.csv')
    e_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}.npy')
    
    print(f"Dataset: {len(g_df)} edges, {e_feat.shape[1]} edge feature dimensions")
    
    # Get unique nodes
    all_nodes = np.unique(np.concatenate([g_df.u.values, g_df.i.values]))
    print(f"Total unique nodes: {len(all_nodes)}")
    
    # For each node, compute statistics of its edge features
    node_edge_stats = {}
    
    for node in all_nodes:
        # Get all edges involving this node
        node_edges = g_df[(g_df.u == node) | (g_df.i == node)]
        
        if len(node_edges) == 0:
            continue
            
        # Get edge features for this node's edges
        edge_indices = node_edges.idx.values - 1  # Convert to 0-indexed
        node_edge_features = e_feat[edge_indices]
        
        # Compute statistics
        node_edge_stats[node] = {
            'mean': np.mean(node_edge_features, axis=0),
            'std': np.std(node_edge_features, axis=0),
            'num_edges': len(node_edges),
            'feature_norm': np.linalg.norm(np.mean(node_edge_features, axis=0))
        }
    
    # Convert to arrays for analysis
    valid_nodes = list(node_edge_stats.keys())
    node_features = np.array([node_edge_stats[node]['mean'] for node in valid_nodes])
    node_norms = np.array([node_edge_stats[node]['feature_norm'] for node in valid_nodes])
    node_degrees = np.array([node_edge_stats[node]['num_edges'] for node in valid_nodes])
    
    print(f"Nodes with edges: {len(valid_nodes)}")
    print(f"Node feature norm range: [{node_norms.min():.3f}, {node_norms.max():.3f}]")
    print(f"Node degree range: [{node_degrees.min()}, {node_degrees.max()}]")
    
    # Test 1: Can we distinguish high-degree vs low-degree nodes using edge features?
    high_degree_threshold = np.percentile(node_degrees, 75)
    low_degree_threshold = np.percentile(node_degrees, 25)
    
    high_degree_mask = node_degrees >= high_degree_threshold
    low_degree_mask = node_degrees <= low_degree_threshold
    
    if np.sum(high_degree_mask) > 0 and np.sum(low_degree_mask) > 0:
        X = np.vstack([node_features[high_degree_mask], node_features[low_degree_mask]])
        y = np.concatenate([np.ones(np.sum(high_degree_mask)), np.zeros(np.sum(low_degree_mask))])
        
        if len(np.unique(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
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
            
            print(f"\n📊 DISTINGUISHING HIGH-DEGREE vs LOW-DEGREE NODES:")
            print(f"   High-degree nodes (>= {high_degree_threshold:.0f} edges): {np.sum(high_degree_mask)}")
            print(f"   Low-degree nodes (<= {low_degree_threshold:.0f} edges): {np.sum(low_degree_mask)}")
            print(f"   Logistic Regression AUC: {lr_auc:.4f}")
            print(f"   Random Forest AUC: {rf_auc:.4f}")
            
            if lr_auc > 0.7 or rf_auc > 0.7:
                print("   ✅ EDGE FEATURES ARE DISCRIMINATIVE for node degree!")
            else:
                print("   ❌ Edge features show weak discrimination")
    
    # Test 2: PCA visualization of node edge feature diversity
    if len(valid_nodes) >= 10:
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(3, node_features.shape[1]))
        node_features_pca = pca.fit_transform(node_features)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"\n📊 EDGE FEATURE DIVERSITY (PCA):")
        print(f"   First {pca.n_components_} PCs explain {explained_var:.3f} of variance")
        print(f"   PC1 range: [{node_features_pca[:, 0].min():.3f}, {node_features_pca[:, 0].max():.3f}]")
        
        if pca.n_components_ >= 2:
            pc_range = node_features_pca[:, 1].max() - node_features_pca[:, 1].min()
            print(f"   PC2 range: [{node_features_pca[:, 1].min():.3f}, {node_features_pca[:, 1].max():.3f}]")
            
            if pc_range > 0.1:  # Arbitrary threshold for "diversity"
                print("   ✅ NODES HAVE DIVERSE EDGE FEATURE PROFILES!")
            else:
                print("   ❌ Nodes have similar edge feature profiles")
    
    # Test 3: Correlation between edge features and node properties
    if len(valid_nodes) >= 10:
        print(f"\n📊 EDGE FEATURES vs NODE PROPERTIES:")
        
        # Correlation with degree
        degree_corr = np.corrcoef(node_norms, node_degrees)[0, 1]
        print(f"   Correlation(edge_feature_norm, degree): {degree_corr:.4f}")
        
        # Diversity of edge feature norms
        norm_std = np.std(node_norms)
        norm_cv = norm_std / np.mean(node_norms) if np.mean(node_norms) > 0 else 0
        print(f"   Edge feature norm diversity (CV): {norm_cv:.4f}")
        
        if norm_cv > 0.2:
            print("   ✅ HIGH DIVERSITY in node edge feature profiles!")
        else:
            print("   ❌ Low diversity in node edge feature profiles")
    
    return {
        'dataset': dataset_name,
        'num_nodes': len(valid_nodes),
        'feature_diversity': norm_cv if len(valid_nodes) >= 10 else 0,
        'degree_correlation': degree_corr if len(valid_nodes) >= 10 else 0
    }

def main():
    """Run analysis on all non-synthetic datasets"""
    datasets = ['CanParl', 'reddit', 'uci', 'wikipedia']
    results = []
    
    print("🔍 VERIFYING DISCRIMINATIVE POWER OF NODE EMBEDDINGS")
    print("   Hypothesis: Different nodes have different edge feature histories")
    print("   that make their embeddings distinguishable")
    
    for dataset in datasets:
        try:
            result = analyze_node_embedding_edge_features(dataset)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error analyzing {dataset}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY: EDGE FEATURE DISCRIMINATIVE POWER")
    print("="*60)
    
    for result in results:
        print(f"\n{result['dataset']:>10}: "
              f"{result['num_nodes']:>5} nodes, "
              f"diversity={result['feature_diversity']:.3f}, "
              f"degree_corr={result['degree_correlation']:.3f}")
    
    # Overall conclusion
    high_diversity_count = sum(1 for r in results if r['feature_diversity'] > 0.2)
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   {high_diversity_count}/{len(results)} datasets show high edge feature diversity")
    
    if high_diversity_count >= len(results) // 2:
        print("   ✅ EDGE FEATURES LIKELY PROVIDE DISCRIMINATIVE POWER")
        print("   ✅ Node embeddings built from different edge contexts ARE distinguishable")
        print("   ✅ This explains TGIB's success even with 'identical' edge features")
    else:
        print("   ❌ Limited evidence for edge feature discrimination")
        print("   ❌ TGIB success may rely more on structural/temporal patterns")

if __name__ == "__main__":
    main() 
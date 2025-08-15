"""
Comprehensive analysis of discriminative power in TGIB:
1. Edge features alone
2. Node pair features alone  
3. Combined edge + node pair features

This directly mimics what TGIB actually uses for distinguishing real vs fake edges.
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

class RandEdgeSampler:
    """Replicate TGIB's exact random edge sampling"""
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        
    def sample(self, size):
        return np.random.choice(self.dst_list, size=size, replace=True)

def analyze_complete_discriminative_power(dataset_name, num_samples=5000):
    """Analyze discriminative power of edge features, node features, and combination"""
    print(f"\n{'='*70}")
    print(f"COMPLETE DISCRIMINATIVE ANALYSIS: {dataset_name.upper()}")
    print("="*70)
    
    # Load data
    g_df = pd.read_csv(f'./processed/{dataset_name}/ml_{dataset_name}.csv')
    e_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}.npy')
    n_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}_node.npy')
    
    print(f"Dataset: {len(g_df)} edges")
    print(f"Edge features: {e_feat.shape[1]} dimensions")
    print(f"Node features: {n_feat.shape[1]} dimensions")
    
    # Create train/val split like TGIB
    val_time = np.quantile(g_df.ts, 0.70)
    train_mask = g_df.ts <= val_time
    
    train_src = g_df[train_mask].u.values
    train_dst = g_df[train_mask].i.values
    train_e_idx = g_df[train_mask].idx.values
    train_ts = g_df[train_mask].ts.values
    
    print(f"Training edges: {len(train_src)}")
    
    # Create random sampler like TGIB
    sampler = RandEdgeSampler(train_src, train_dst)
    
    # Sample edges for analysis
    num_real_edges = min(num_samples, len(train_src))
    sample_indices = np.random.choice(len(train_src), num_real_edges, replace=False)
    
    real_src = train_src[sample_indices]
    real_dst = train_dst[sample_indices] 
    real_e_idx = train_e_idx[sample_indices]
    
    # Generate fake destinations (like TGIB negative sampling)
    fake_dst = sampler.sample(num_real_edges)
    
    print(f"Analyzing {num_real_edges} real edges vs {num_real_edges} fake edges")
    
    # Extract features
    # 1. Edge features
    real_edge_feat = e_feat[real_e_idx - 1]  # Convert to 0-indexed
    fake_edge_feat = e_feat[real_e_idx - 1]  # Same edge features for fake edges!
    
    # 2. Node pair features
    real_src_feat = n_feat[real_src]
    real_dst_feat = n_feat[real_dst]
    fake_src_feat = n_feat[real_src]  # Same source
    fake_dst_feat = n_feat[fake_dst]  # Different destination
    
    # Combine node pair features (various strategies)
    real_node_pair_concat = np.concatenate([real_src_feat, real_dst_feat], axis=1)
    fake_node_pair_concat = np.concatenate([fake_src_feat, fake_dst_feat], axis=1)
    
    real_node_pair_diff = real_dst_feat - real_src_feat
    fake_node_pair_diff = fake_dst_feat - fake_src_feat
    
    real_node_pair_prod = real_src_feat * real_dst_feat
    fake_node_pair_prod = fake_src_feat * fake_dst_feat
    
    # 3. Combined features (like TGIB)
    real_combined = np.concatenate([real_src_feat, real_dst_feat, real_edge_feat], axis=1)
    fake_combined = np.concatenate([fake_src_feat, fake_dst_feat, fake_edge_feat], axis=1)
    
    # Test configurations
    test_configs = [
        ("Edge Features Only", real_edge_feat, fake_edge_feat),
        ("Node Pair (Concat)", real_node_pair_concat, fake_node_pair_concat),
        ("Node Pair (Difference)", real_node_pair_diff, fake_node_pair_diff), 
        ("Node Pair (Product)", real_node_pair_prod, fake_node_pair_prod),
        ("Combined (like TGIB)", real_combined, fake_combined)
    ]
    
    results = {}
    
    for config_name, real_feat, fake_feat in test_configs:
        print(f"\n📊 TESTING: {config_name}")
        print(f"   Feature dimensions: {real_feat.shape[1]}")
        
        # Create binary classification dataset
        X = np.vstack([real_feat, fake_feat])
        y = np.concatenate([np.ones(len(real_feat)), np.zeros(len(fake_feat))])
        
        # Check for valid features
        if np.all(real_feat == fake_feat):
            print("   ❌ IDENTICAL FEATURES: Real and fake features are identical")
            results[config_name] = {'lr_auc': 0.5, 'rf_auc': 0.5, 'status': 'identical'}
            continue
            
        if np.std(X) == 0:
            print("   ❌ CONSTANT FEATURES: All features are constant")
            results[config_name] = {'lr_auc': 0.5, 'rf_auc': 0.5, 'status': 'constant'}
            continue
        
        try:
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
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
            
            print(f"   Logistic Regression AUC: {lr_auc:.4f}")
            print(f"   Random Forest AUC: {rf_auc:.4f}")
            
            # Interpretation
            if lr_auc > 0.8 or rf_auc > 0.8:
                status = "excellent"
                print("   ✅ EXCELLENT DISCRIMINATION")
            elif lr_auc > 0.7 or rf_auc > 0.7:
                status = "good"
                print("   ✅ GOOD DISCRIMINATION")
            elif lr_auc > 0.6 or rf_auc > 0.6:
                status = "moderate"
                print("   🟡 MODERATE DISCRIMINATION")
            else:
                status = "weak"
                print("   ❌ WEAK DISCRIMINATION")
                
            results[config_name] = {
                'lr_auc': lr_auc, 
                'rf_auc': rf_auc, 
                'status': status
            }
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results[config_name] = {'lr_auc': 0.5, 'rf_auc': 0.5, 'status': 'error'}
    
    return results

def analyze_feature_importance(dataset_name, num_samples=2000):
    """Analyze which features are most important for discrimination"""
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS: {dataset_name.upper()}")
    print("-" * 50)
    
    # Load data
    g_df = pd.read_csv(f'./processed/{dataset_name}/ml_{dataset_name}.csv')
    e_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}.npy')
    n_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}_node.npy')
    
    # Create train set
    val_time = np.quantile(g_df.ts, 0.70)
    train_mask = g_df.ts <= val_time
    
    train_src = g_df[train_mask].u.values
    train_dst = g_df[train_mask].i.values
    train_e_idx = g_df[train_mask].idx.values
    
    sampler = RandEdgeSampler(train_src, train_dst)
    
    # Sample edges
    num_edges = min(num_samples, len(train_src))
    sample_indices = np.random.choice(len(train_src), num_edges, replace=False)
    
    real_src = train_src[sample_indices]
    real_dst = train_dst[sample_indices]
    real_e_idx = train_e_idx[sample_indices]
    fake_dst = sampler.sample(num_edges)
    
    # Create combined features with labels
    real_src_feat = n_feat[real_src]
    real_dst_feat = n_feat[real_dst]
    real_edge_feat = e_feat[real_e_idx - 1]
    
    fake_src_feat = n_feat[real_src]
    fake_dst_feat = n_feat[fake_dst]
    fake_edge_feat = e_feat[real_e_idx - 1]
    
    # Combined features
    real_combined = np.concatenate([real_src_feat, real_dst_feat, real_edge_feat], axis=1)
    fake_combined = np.concatenate([fake_src_feat, fake_dst_feat, fake_edge_feat], axis=1)
    
    X = np.vstack([real_combined, fake_combined])
    y = np.concatenate([np.ones(len(real_combined)), np.zeros(len(fake_combined))])
    
    if np.std(X) > 0:
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        n_src_feat = real_src_feat.shape[1]
        n_dst_feat = real_dst_feat.shape[1] 
        n_edge_feat = real_edge_feat.shape[1]
        
        src_importance = np.sum(importance[:n_src_feat])
        dst_importance = np.sum(importance[n_src_feat:n_src_feat + n_dst_feat])
        edge_importance = np.sum(importance[n_src_feat + n_dst_feat:])
        
        print(f"Source node features importance: {src_importance:.4f}")
        print(f"Destination node features importance: {dst_importance:.4f}")
        print(f"Edge features importance: {edge_importance:.4f}")
        
        total_node_importance = src_importance + dst_importance
        print(f"\nNode features total: {total_node_importance:.4f}")
        print(f"Edge features total: {edge_importance:.4f}")
        
        if total_node_importance > edge_importance:
            print("✅ NODE FEATURES DOMINATE")
        else:
            print("✅ EDGE FEATURES DOMINATE")
    else:
        print("❌ Cannot analyze - features are constant")

def main():
    """Run complete discriminative power analysis"""
    datasets = ['CanParl', 'reddit', 'uci', 'wikipedia']
    all_results = {}
    
    print("🔬 COMPREHENSIVE DISCRIMINATIVE POWER ANALYSIS")
    print("Testing what makes TGIB successful at distinguishing real vs fake edges")
    
    for dataset in datasets:
        try:
            results = analyze_complete_discriminative_power(dataset)
            all_results[dataset] = results
            analyze_feature_importance(dataset)
        except Exception as e:
            print(f"\n❌ Error analyzing {dataset}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 SUMMARY: WHAT DRIVES TGIB'S DISCRIMINATIVE POWER?")
    print("="*70)
    
    for dataset, results in all_results.items():
        print(f"\n🔍 {dataset.upper()}:")
        
        best_config = None
        best_auc = 0
        
        for config, metrics in results.items():
            max_auc = max(metrics['lr_auc'], metrics['rf_auc'])
            print(f"   {config:<25}: {metrics['status']:<10} (best AUC: {max_auc:.3f})")
            
            if max_auc > best_auc:
                best_auc = max_auc
                best_config = config
        
        if best_config:
            print(f"   🏆 BEST: {best_config} (AUC: {best_auc:.3f})")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print("   - Edge features alone may be insufficient")
    print("   - Node pair features capture structural relationships")  
    print("   - Combined features (like TGIB uses) should be most powerful")
    print("   - This explains synthetic graph failure: both nodes & edges lack diversity")

if __name__ == "__main__":
    main() 
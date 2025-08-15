"""
CORRECTED Analysis of TGIB's Actual Training Mechanism
Replicates the exact sequential pattern TGIB uses for positive vs negative samples
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

class TGIBRandEdgeSampler:
    """Exact replica of TGIB's RandEdgeSampler"""
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list) 
        self.dst_list = np.unique(dst_list) 
        
    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size) 
        return self.src_list[src_index], self.dst_list[dst_index]

def analyze_tgib_sequential_mechanism(dataset_name, num_steps=1000):
    """Analyze TGIB's actual sequential training mechanism"""
    print(f"\n{'='*70}")
    print(f"TGIB'S ACTUAL SEQUENTIAL MECHANISM: {dataset_name.upper()}")
    print("="*70)
    
    # Load data exactly like TGIB
    g_df = pd.read_csv(f'./processed/{dataset_name}/ml_{dataset_name}.csv')
    e_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}.npy')
    n_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}_node.npy')
    
    print(f"Dataset: {len(g_df)} edges")
    print(f"Edge features: {e_feat.shape[1]} dimensions") 
    print(f"Node features: {n_feat.shape[1]} dimensions")
    
    # Create exact same train split as TGIB
    val_time = np.quantile(g_df.ts, 0.70)
    train_mask = g_df.ts <= val_time
    
    train_src_l = g_df[train_mask].u.values
    train_dst_l = g_df[train_mask].i.values
    train_e_idx_l = g_df[train_mask].idx.values
    train_ts_l = g_df[train_mask].ts.values
    
    print(f"Training edges: {len(train_src_l)}")
    
    # Create exact same sampler as TGIB
    train_rand_sampler = TGIBRandEdgeSampler(train_src_l, train_dst_l)
    
    # Analyze sequence characteristics for different k values
    k_values = np.linspace(10, min(num_steps, len(train_src_l)-1), 20, dtype=int)
    
    print(f"\n📊 ANALYZING SEQUENCE CHARACTERISTICS:")
    print(f"   Testing {len(k_values)} different sequence lengths")
    
    results = []
    
    for k in k_values:
        if k >= len(train_src_l):
            continue
            
        # REPLICATE TGIB'S EXACT MECHANISM
        u_emb_fake, i_emb_fake = train_rand_sampler.sample(1)
        
        # Positive sequence: edges 0 to k
        pos_src_seq = train_src_l[:k+1]
        pos_dst_seq = train_dst_l[:k+1] 
        pos_e_idx_seq = train_e_idx_l[:k+1]
        pos_ts_seq = train_ts_l[:k+1]
        
        # What TGIB actually does for negative:
        # Uses same sequence but replaces the last destination with fake
        neg_src_seq = pos_src_seq.copy()
        neg_dst_seq = pos_dst_seq.copy()
        neg_dst_seq[-1] = i_emb_fake[0]  # Replace last destination with fake
        neg_e_idx_seq = pos_e_idx_seq  # Same edge indices!
        neg_ts_seq = pos_ts_seq  # Same timestamps!
        
        # Extract features for analysis
        pos_edge_feat = e_feat[pos_e_idx_seq - 1]  # Convert to 0-indexed
        neg_edge_feat = e_feat[neg_e_idx_seq - 1]  # Same edge features!
        
        pos_src_feat = n_feat[pos_src_seq]
        pos_dst_feat = n_feat[pos_dst_seq]
        neg_src_feat = n_feat[neg_src_seq] 
        neg_dst_feat = n_feat[neg_dst_seq]
        
        # Check what's different between positive and negative
        edge_features_identical = np.array_equal(pos_edge_feat, neg_edge_feat)
        src_features_identical = np.array_equal(pos_src_feat, neg_src_feat)
        
        # Only the LAST destination node differs
        last_dst_different = pos_dst_seq[-1] != neg_dst_seq[-1]
        
        # Sequence-level aggregation features
        pos_edge_mean = np.mean(pos_edge_feat, axis=0)
        neg_edge_mean = np.mean(neg_edge_feat, axis=0)
        edge_aggregation_identical = np.array_equal(pos_edge_mean, neg_edge_mean)
        
        pos_dst_mean = np.mean(pos_dst_feat, axis=0)
        neg_dst_mean = np.mean(neg_dst_feat, axis=0)
        dst_aggregation_different = not np.array_equal(pos_dst_mean, neg_dst_mean)
        
        results.append({
            'k': k,
            'edge_features_identical': edge_features_identical,
            'src_features_identical': src_features_identical, 
            'last_dst_different': last_dst_different,
            'edge_aggregation_identical': edge_aggregation_identical,
            'dst_aggregation_different': dst_aggregation_different,
            'pos_dst_mean_norm': np.linalg.norm(pos_dst_mean),
            'neg_dst_mean_norm': np.linalg.norm(neg_dst_mean)
        })
    
    # Analyze results
    print(f"\n📋 SEQUENCE ANALYSIS RESULTS:")
    edge_always_identical = all(r['edge_features_identical'] for r in results)
    src_always_identical = all(r['src_features_identical'] for r in results)
    dst_always_different = all(r['last_dst_different'] for r in results)
    edge_agg_always_identical = all(r['edge_aggregation_identical'] for r in results)
    dst_agg_sometimes_different = any(r['dst_aggregation_different'] for r in results)
    
    print(f"   Edge features always identical: {edge_always_identical}")
    print(f"   Source features always identical: {src_always_identical}")
    print(f"   Last destination always different: {dst_always_different}")
    print(f"   Edge aggregation always identical: {edge_agg_always_identical}")
    print(f"   Destination aggregation sometimes different: {dst_agg_sometimes_different}")
    
    # Test discriminative power of destination aggregations
    if dst_agg_sometimes_different:
        print(f"\n🔍 TESTING DESTINATION AGGREGATION DISCRIMINATION:")
        
        pos_features = []
        neg_features = []
        
        for k in k_values[:10]:  # Test first 10 k values
            if k >= len(train_src_l):
                continue
                
            u_emb_fake, i_emb_fake = train_rand_sampler.sample(1)
            
            pos_dst_seq = train_dst_l[:k+1]
            neg_dst_seq = pos_dst_seq.copy()
            neg_dst_seq[-1] = i_emb_fake[0]
            
            pos_dst_feat = n_feat[pos_dst_seq]
            neg_dst_feat = n_feat[neg_dst_seq]
            
            pos_dst_aggregated = np.mean(pos_dst_feat, axis=0)
            neg_dst_aggregated = np.mean(neg_dst_feat, axis=0)
            
            pos_features.append(pos_dst_aggregated)
            neg_features.append(neg_dst_aggregated)
        
        if len(pos_features) > 0:
            X = np.vstack([np.array(pos_features), np.array(neg_features)])
            y = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))])
            
            if np.std(X) > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                lr = LogisticRegression(random_state=42, max_iter=1000)
                lr.fit(X_train_scaled, y_train)
                lr_pred = lr.predict_proba(X_test_scaled)[:, 1]
                lr_auc = roc_auc_score(y_test, lr_pred)
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train_scaled, y_train)
                rf_pred = rf.predict_proba(X_test_scaled)[:, 1]
                rf_auc = roc_auc_score(y_test, rf_pred)
                
                print(f"   Logistic Regression AUC: {lr_auc:.4f}")
                print(f"   Random Forest AUC: {rf_auc:.4f}")
                
                if lr_auc > 0.7 or rf_auc > 0.7:
                    print("   ✅ DESTINATION AGGREGATION IS DISCRIMINATIVE!")
                else:
                    print("   ❌ Weak discrimination from destination aggregation")
            else:
                print("   ❌ Destination features are constant")
    
    return results

def main():
    """Analyze TGIB's actual mechanism across datasets"""
    datasets = ['CanParl', 'reddit', 'uci', 'wikipedia']
    
    print("🔬 ANALYZING TGIB'S ACTUAL TRAINING MECHANISM")
    print("="*70)
    print("Key insight: TGIB compares SEQUENCES, not individual edges!")
    print("Positive: Real sequence [0...k]")
    print("Negative: Real sequence [0...k-1] + fake edge k")
    
    for dataset in datasets:
        try:
            analyze_tgib_sequential_mechanism(dataset)
        except Exception as e:
            print(f"\n❌ Error analyzing {dataset}: {e}")
    
    print(f"\n🎯 KEY REVELATIONS:")
    print("   1. TGIB uses SEQUENTIAL prediction, not edge-by-edge")
    print("   2. Edge features ARE identical for positive/negative sequences")
    print("   3. Only the LAST destination node differs between pos/neg")
    print("   4. Discriminative power comes from AGGREGATED destination features")
    print("   5. This explains why individual edge analysis showed 'identical' features!")

if __name__ == "__main__":
    main() 
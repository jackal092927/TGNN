"""
CORRECTED Analysis of edge feature discriminative power for link prediction
Examines whether edge features can distinguish real edges from randomly sampled edges
(which is the actual task TGIB performs, not predicting the CSV label column)
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
    """Replicate the exact same random sampling used by TGIB"""
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list) 
        self.dst_list = np.unique(dst_list) 
    
    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size) 
        return self.src_list[src_index], self.dst_list[dst_index]

def analyze_dataset(dataset_name, num_negative_samples=10000):
    """Analyze edge feature discriminative power for real vs fake edges"""
    print(f"\n{'='*60}")
    print(f"ANALYZING DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load data
        df = pd.read_csv(f'./processed/{dataset_name}/ml_{dataset_name}.csv')
        edge_features = np.load(f'./processed/{dataset_name}/ml_{dataset_name}.npy')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Edge features shape: {edge_features.shape}")
        
        # Fix dimension mismatch - align the sizes
        min_len = min(len(df), len(edge_features))
        df = df.iloc[:min_len]
        edge_features = edge_features[:min_len]
        
        print(f"After alignment - Dataset shape: {df.shape}, Edge features shape: {edge_features.shape}")
        print(f"Number of edge features: {edge_features.shape[1] if len(edge_features.shape) > 1 else 1}")
        
        # Create positive examples: all real edges (ignore CSV label column)
        real_edges = set(zip(df['u'].values, df['i'].values))
        print(f"Number of real edges: {len(real_edges)}")
        
        # Create negative examples using the same random sampling as TGIB
        sampler = RandEdgeSampler(df['u'].values, df['i'].values)
        fake_edges = []
        fake_edge_indices = []
        
        # Sample negative edges that don't exist in the real edge set
        attempts = 0
        max_attempts = num_negative_samples * 10  # Prevent infinite loop
        
        while len(fake_edges) < num_negative_samples and attempts < max_attempts:
            src_samples, dst_samples = sampler.sample(num_negative_samples * 2)
            
            for src, dst in zip(src_samples, dst_samples):
                if (src, dst) not in real_edges and len(fake_edges) < num_negative_samples:
                    fake_edges.append((src, dst))
                    # Find a random edge index for the fake edge (since we need edge features)
                    fake_edge_indices.append(np.random.randint(0, len(edge_features)))
            
            attempts += num_negative_samples * 2
        
        if len(fake_edges) == 0:
            print("ERROR: Could not generate any negative samples!")
            return None
            
        print(f"Generated {len(fake_edges)} negative samples after {attempts} attempts")
        
        # Prepare features and labels
        if len(edge_features.shape) == 1:
            edge_features = edge_features.reshape(-1, 1)
            
        # Positive samples: real edge features
        pos_features = edge_features[:len(real_edges)]
        pos_labels = np.ones(len(real_edges))
        
        # Negative samples: random edge features (simulating random sampling)
        neg_features = edge_features[fake_edge_indices[:len(fake_edges)]]
        neg_labels = np.zeros(len(fake_edges))
        
        # Combine positive and negative
        all_features = np.vstack([pos_features, neg_features])
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        print(f"Final dataset: {len(all_features)} samples, {all_features.shape[1]} features")
        print(f"Positive ratio: {np.mean(all_labels):.4f}")
        
        # Remove constant features
        feature_std = np.std(all_features, axis=0)
        non_constant_mask = feature_std > 1e-10
        meaningful_features = all_features[:, non_constant_mask]
        
        print(f"Features with variation: {meaningful_features.shape[1]} out of {all_features.shape[1]}")
        
        if meaningful_features.shape[1] == 0:
            print("WARNING: No meaningful edge features found (all constant)!")
            return {
                'dataset': dataset_name,
                'num_features': all_features.shape[1],
                'meaningful_features': 0,
                'lr_auc': 0.5,
                'lr_ap': np.mean(all_labels),
                'rf_auc': 0.5,
                'rf_ap': np.mean(all_labels),
                'max_correlation': 0,
                'mean_correlation': 0,
                'improvement_auc': 0,
                'improvement_ap': 0,
                'note': 'No meaningful features'
            }
        
        # Calculate correlation with labels (real vs fake edge distinction)
        correlations = []
        for i in range(meaningful_features.shape[1]):
            corr = np.corrcoef(meaningful_features[:, i], all_labels)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        max_correlation = max(correlations) if correlations else 0
        mean_correlation = np.mean(correlations) if correlations else 0
        
        print(f"Max absolute correlation with real/fake label: {max_correlation:.4f}")
        print(f"Mean absolute correlation with real/fake label: {mean_correlation:.4f}")
        
        # Feature statistics
        print(f"\nEdge feature statistics:")
        print(f"Mean feature values: {np.mean(meaningful_features, axis=0)[:5]}..." if meaningful_features.shape[1] > 5 else f"Mean feature values: {np.mean(meaningful_features, axis=0)}")
        print(f"Std feature values: {np.std(meaningful_features, axis=0)[:5]}..." if meaningful_features.shape[1] > 5 else f"Std feature values: {np.std(meaningful_features, axis=0)}")
        
        # Split data for classification
        X_train, X_test, y_train, y_test = train_test_split(
            meaningful_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_probs)
        lr_ap = average_precision_score(y_test, lr_probs)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_probs)
        rf_ap = average_precision_score(y_test, rf_probs)
        
        print(f"\nClassification performance for real vs fake edges:")
        print(f"Logistic Regression - AUC: {lr_auc:.4f}, AP: {lr_ap:.4f}")
        print(f"Random Forest - AUC: {rf_auc:.4f}, AP: {rf_ap:.4f}")
        
        # Feature importance from Random Forest
        if meaningful_features.shape[1] <= 20:
            importances = rf.feature_importances_
            print(f"\nTop 5 feature importances: {sorted(importances, reverse=True)[:5]}")
        
        # Baseline comparison
        baseline_auc = 0.5
        baseline_ap = np.mean(y_test)
        print(f"Baseline (random) - AUC: {baseline_auc:.4f}, AP: {baseline_ap:.4f}")
        
        # Check if features are significantly better than random
        improvement_auc = max(lr_auc, rf_auc) - baseline_auc
        improvement_ap = max(lr_ap, rf_ap) - baseline_ap
        print(f"Best improvement over baseline - AUC: +{improvement_auc:.4f}, AP: +{improvement_ap:.4f}")
        
        return {
            'dataset': dataset_name,
            'num_features': all_features.shape[1],
            'meaningful_features': meaningful_features.shape[1],
            'lr_auc': lr_auc,
            'lr_ap': lr_ap,
            'rf_auc': rf_auc,
            'rf_ap': rf_ap,
            'max_correlation': max_correlation,
            'mean_correlation': mean_correlation,
            'improvement_auc': improvement_auc,
            'improvement_ap': improvement_ap,
            'note': 'Success - Real vs Fake Edge Prediction'
        }
        
    except Exception as e:
        print(f"Error analyzing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'num_features': 0,
            'meaningful_features': 0,
            'lr_auc': 0.5,
            'lr_ap': 0,
            'rf_auc': 0.5,
            'rf_ap': 0,
            'max_correlation': 0,
            'mean_correlation': 0,
            'improvement_auc': 0,
            'improvement_ap': 0,
            'note': f'Error: {str(e)}'
        }

def main():
    """Main analysis function"""
    datasets = ['CanParl', 'reddit', 'uci', 'wikipedia']
    results = []
    
    print("CORRECTED Analysis: Edge feature discriminative power for REAL vs FAKE edge prediction")
    print("(This is the actual task TGIB performs - not predicting CSV label column)")
    print("=" * 90)
    
    for dataset in datasets:
        result = analyze_dataset(dataset, num_negative_samples=5000)
        if result:
            results.append(result)
    
    # Summary comparison
    print(f"\n{'='*90}")
    print("SUMMARY COMPARISON")
    print(f"{'='*90}")
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Analysis
    print(f"\n{'='*90}")
    print("ANALYSIS")
    print(f"{'='*90}")
    
    strong_discriminative = df_results[df_results['improvement_auc'] > 0.1]
    moderate_discriminative = df_results[(df_results['improvement_auc'] > 0.05) & (df_results['improvement_auc'] <= 0.1)]
    weak_discriminative = df_results[df_results['improvement_auc'] <= 0.05]
    
    print(f"Datasets with STRONG discriminative edge features (AUC improvement > 0.1):")
    if len(strong_discriminative) > 0:
        for _, row in strong_discriminative.iterrows():
            print(f"  - {row['dataset']}: AUC improvement = {row['improvement_auc']:.4f}")
    else:
        print("  None")
    
    print(f"\nDatasets with MODERATE discriminative edge features (0.05 < AUC improvement <= 0.1):")
    if len(moderate_discriminative) > 0:
        for _, row in moderate_discriminative.iterrows():
            print(f"  - {row['dataset']}: AUC improvement = {row['improvement_auc']:.4f}")
    else:
        print("  None")
    
    print(f"\nDatasets with WEAK discriminative edge features (AUC improvement <= 0.05):")
    if len(weak_discriminative) > 0:
        for _, row in weak_discriminative.iterrows():
            print(f"  - {row['dataset']}: AUC improvement = {row['improvement_auc']:.4f}")
    else:
        print("  None")
    
    # Correlation analysis
    high_corr = df_results[df_results['max_correlation'] > 0.3]
    print(f"\nDatasets with high feature correlation for real vs fake edge distinction (max correlation > 0.3):")
    if len(high_corr) > 0:
        for _, row in high_corr.iterrows():
            print(f"  - {row['dataset']}: max correlation = {row['max_correlation']:.4f}")
    else:
        print("  None")
    
    print(f"\nConclusion:")
    successful_results = df_results[df_results['note'].str.contains('Success', na=False)]
    if len(successful_results) > 0:
        avg_improvement = successful_results['improvement_auc'].mean()
        print(f"Average AUC improvement across successfully analyzed datasets: {avg_improvement:.4f}")
        
        if avg_improvement > 0.1:
            print("STRONG evidence that edge features help distinguish real from randomly sampled edges.")
            print("This suggests TGIB's success DOES depend significantly on discriminative edge features.")
        elif avg_improvement > 0.05:
            print("MODERATE evidence that edge features help distinguish real from randomly sampled edges.")
            print("This suggests TGIB's success moderately depends on discriminative edge features.")
        else:
            print("WEAK evidence that edge features help distinguish real from randomly sampled edges.")
            print("This supports your hypothesis: TGIB's success may NOT strongly depend on discriminative edge features.")
            print("The model might be learning temporal/structural patterns rather than relying on edge content.")
    else:
        print("No datasets could be successfully analyzed.")

if __name__ == "__main__":
    main() 
"""
Analyze temporal neighborhood diversity in real vs synthetic datasets
Hypothesis: Synthetic graphs lack the temporal neighborhood diversity that TGIB relies on
"""
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_temporal_neighborhood_diversity(dataset_name):
    """Analyze diversity of temporal neighborhood structures"""
    print(f"\n{'='*70}")
    print(f"TEMPORAL NEIGHBORHOOD DIVERSITY: {dataset_name.upper()}")
    print("="*70)
    
    try:
        # Load data
        g_df = pd.read_csv(f'./processed/{dataset_name}/ml_{dataset_name}.csv')
        e_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}.npy')
        n_feat = np.load(f'./processed/{dataset_name}/ml_{dataset_name}_node.npy')
        
        print(f"Dataset: {len(g_df)} edges, {len(np.unique(np.concatenate([g_df.u, g_df.i])))} nodes")
        
        # Create train split
        val_time = np.quantile(g_df.ts, 0.70)
        train_mask = g_df.ts <= val_time
        train_df = g_df[train_mask].copy()
        
        print(f"Training edges: {len(train_df)}")
        
        # Build temporal neighborhoods for each node
        all_nodes = np.unique(np.concatenate([train_df.u, train_df.i]))
        node_neighborhoods = {}
        
        print(f"\n📊 BUILDING TEMPORAL NEIGHBORHOODS:")
        for node in all_nodes:
            # Get all edges involving this node, sorted by time
            node_edges = train_df[(train_df.u == node) | (train_df.i == node)].sort_values('ts')
            
            if len(node_edges) == 0:
                continue
                
            # Get neighbors (nodes this node has connected to)
            neighbors = []
            edge_times = []
            edge_features = []
            
            for _, edge in node_edges.iterrows():
                if edge.u == node:
                    neighbor = edge.i
                else:
                    neighbor = edge.u
                
                neighbors.append(neighbor)
                edge_times.append(edge.ts)
                # Fix indexing issue
                edge_idx = int(edge.idx - 1)  # Convert to 0-indexed
                if 0 <= edge_idx < len(e_feat):
                    edge_features.append(e_feat[edge_idx])
                else:
                    edge_features.append(np.zeros(e_feat.shape[1] if len(e_feat.shape) > 1 else 1))
            
            if len(edge_features) > 0:
                edge_features = np.array(edge_features)
                if len(edge_features.shape) == 1:
                    edge_features = edge_features.reshape(-1, 1)
            else:
                edge_features = np.zeros((0, e_feat.shape[1] if len(e_feat.shape) > 1 else 1))
            
            node_neighborhoods[node] = {
                'neighbors': np.array(neighbors),
                'times': np.array(edge_times), 
                'edge_features': edge_features,
                'degree': len(neighbors),
                'time_span': edge_times[-1] - edge_times[0] if len(edge_times) > 1 else 0
            }
        
        print(f"   Nodes with neighborhoods: {len(node_neighborhoods)}")
        
        # Extract neighborhood characteristics
        neighborhood_features = []
        node_ids = []
        
        for node, nhood in node_neighborhoods.items():
            if nhood['degree'] == 0:
                continue
                
            # Temporal characteristics
            degree = nhood['degree']
            time_span = nhood['time_span']
            time_density = degree / (time_span + 1e-6)  # edges per time unit
            
            # Neighbor diversity (how many unique neighbors)
            unique_neighbors = len(np.unique(nhood['neighbors']))
            neighbor_diversity = unique_neighbors / degree  # ratio of unique neighbors
            
            # Edge feature characteristics
            if len(nhood['edge_features']) > 0:
                edge_feat_mean = np.mean(nhood['edge_features'], axis=0)
                edge_feat_std = np.std(nhood['edge_features'], axis=0)
                edge_feat_norm = np.linalg.norm(edge_feat_mean)
                edge_feat_std_mean = np.mean(edge_feat_std)
            else:
                edge_feat_norm = 0
                edge_feat_std_mean = 0
            
            # Time pattern characteristics
            if len(nhood['times']) > 1:
                time_intervals = np.diff(nhood['times'])
                time_regularity = np.std(time_intervals) / (np.mean(time_intervals) + 1e-6)
            else:
                time_regularity = 0
            
            # Combine into feature vector
            features = [
                degree,
                time_span,
                time_density,
                neighbor_diversity,
                edge_feat_norm,
                edge_feat_std_mean,
                time_regularity,
                unique_neighbors
            ]
            
            neighborhood_features.append(features)
            node_ids.append(node)
        
        neighborhood_features = np.array(neighborhood_features)
        node_ids = np.array(node_ids)
        
        print(f"   Feature matrix shape: {neighborhood_features.shape}")
        
        if len(neighborhood_features) < 10:
            print("   ❌ Too few nodes for analysis")
            return None
        
        # Analyze diversity
        print(f"\n📈 NEIGHBORHOOD DIVERSITY ANALYSIS:")
        
        # 1. Overall diversity (coefficient of variation)
        feature_means = np.mean(neighborhood_features, axis=0)
        feature_stds = np.std(neighborhood_features, axis=0)
        feature_cvs = feature_stds / (feature_means + 1e-6)
        
        feature_names = [
            'degree', 'time_span', 'time_density', 'neighbor_diversity',
            'edge_feat_norm', 'edge_feat_std', 'time_regularity', 'unique_neighbors'
        ]
        
        overall_diversity = np.mean(feature_cvs)
        print(f"   Overall diversity (mean CV): {overall_diversity:.4f}")
        
        for name, cv in zip(feature_names, feature_cvs):
            print(f"   {name:>15}: CV = {cv:.4f}")
        
        # 2. Clustering analysis
        best_silhouette = 0
        if np.std(neighborhood_features) > 0:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(neighborhood_features)
            
            # Try different numbers of clusters
            best_silhouette = -1
            best_k = 2
            
            for k in range(2, min(10, len(neighborhood_features)//3)):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    cluster_labels = kmeans.fit_predict(scaled_features)
                    silhouette = silhouette_score(scaled_features, cluster_labels)
                    
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_k = k
                except:
                    continue
            
            print(f"\n🎯 CLUSTERING ANALYSIS:")
            print(f"   Best number of clusters: {best_k}")
            print(f"   Best silhouette score: {best_silhouette:.4f}")
            
            if best_silhouette > 0.3:
                print("   ✅ DISTINCT NEIGHBORHOOD CLUSTERS FOUND")
            elif best_silhouette > 0.1:
                print("   🟡 MODERATE NEIGHBORHOOD CLUSTERING")
            else:
                print("   ❌ POOR NEIGHBORHOOD CLUSTERING (low diversity)")
        else:
            print("   ❌ No variation in neighborhood features")
        
        # 3. PCA analysis
        pca_first_component = 1.0
        if np.std(neighborhood_features) > 0:
            pca = PCA()
            pca_features = pca.fit_transform(StandardScaler().fit_transform(neighborhood_features))
            explained_variance = pca.explained_variance_ratio_
            pca_first_component = explained_variance[0]
            
            print(f"\n📊 PCA ANALYSIS:")
            print(f"   First PC explains: {explained_variance[0]:.3f} of variance")
            print(f"   First 2 PCs explain: {np.sum(explained_variance[:2]):.3f} of variance")
            print(f"   First 3 PCs explain: {np.sum(explained_variance[:3]):.3f} of variance")
            
            if explained_variance[0] > 0.8:
                print("   ❌ VERY LOW DIVERSITY: One dimension dominates")
            elif explained_variance[0] > 0.6:
                print("   🟡 MODERATE DIVERSITY: Some dominant patterns")
            else:
                print("   ✅ HIGH DIVERSITY: Multiple important dimensions")
        
        # 4. Neighbor overlap analysis
        print(f"\n🔗 NEIGHBOR OVERLAP ANALYSIS:")
        neighbor_sets = {node: set(nhood['neighbors']) for node, nhood in node_neighborhoods.items() if nhood['degree'] > 0}
        
        mean_overlap = 0
        if len(neighbor_sets) > 1:
            overlaps = []
            node_pairs = list(neighbor_sets.keys())
            
            for i in range(min(100, len(node_pairs))):  # Sample to avoid too much computation
                for j in range(i+1, min(100, len(node_pairs))):
                    set1 = neighbor_sets[node_pairs[i]]
                    set2 = neighbor_sets[node_pairs[j]]
                    
                    if len(set1) > 0 and len(set2) > 0:
                        overlap = len(set1 & set2) / len(set1 | set2)  # Jaccard similarity
                        overlaps.append(overlap)
            
            if overlaps:
                mean_overlap = np.mean(overlaps)
                print(f"   Mean neighbor overlap (Jaccard): {mean_overlap:.4f}")
                
                if mean_overlap < 0.1:
                    print("   ✅ LOW OVERLAP: Nodes have distinct neighborhoods")
                elif mean_overlap < 0.3:
                    print("   🟡 MODERATE OVERLAP: Some shared neighborhoods")
                else:
                    print("   ❌ HIGH OVERLAP: Many nodes share similar neighborhoods")
            else:
                print("   ❌ Cannot compute neighbor overlap")
        else:
            print("   ❌ Too few nodes for overlap analysis")
        
        return {
            'dataset': dataset_name,
            'overall_diversity': overall_diversity,
            'silhouette_score': best_silhouette,
            'neighbor_overlap': mean_overlap,
            'num_nodes': len(node_neighborhoods),
            'pca_first_component': pca_first_component
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    """Compare temporal neighborhood diversity across real and synthetic datasets"""
    
    # Real datasets
    real_datasets = ['CanParl', 'reddit', 'uci', 'wikipedia']
    
    # Synthetic datasets
    synthetic_datasets = [
        'triadic_medium', 'triadic_sparse', 'triadic_dense', 'triadic_closure_demo',
        'synthetic_sd_ba', 'synthetic_sd_er', 'synthetic_cc_ws', 
        'synthetic_icm_ba', 'synthetic_ltm_ba', 'synthetic'
    ]
    
    print("🔬 TEMPORAL NEIGHBORHOOD DIVERSITY ANALYSIS")
    print("="*70)
    print("Hypothesis: Synthetic graphs lack neighborhood diversity that TGIB needs")
    print("Real datasets should show high diversity, synthetic should show low diversity")
    
    real_results = []
    synthetic_results = []
    
    print(f"\n{'🌍 ANALYZING REAL DATASETS':=^70}")
    for dataset in real_datasets:
        try:
            result = analyze_temporal_neighborhood_diversity(dataset)
            if result:
                real_results.append(result)
        except Exception as e:
            print(f"\n❌ Error analyzing {dataset}: {e}")
    
    print(f"\n{'🤖 ANALYZING SYNTHETIC DATASETS':=^70}")
    for dataset in synthetic_datasets:
        try:
            result = analyze_temporal_neighborhood_diversity(dataset)
            if result:
                synthetic_results.append(result)
        except Exception as e:
            print(f"\n❌ Error analyzing {dataset}: {e}")
    
    # Comparative Summary
    if real_results or synthetic_results:
        print(f"\n{'='*70}")
        print("📋 COMPARATIVE SUMMARY: REAL vs SYNTHETIC")
        print("="*70)
        
        print(f"\n🌍 REAL DATASETS:")
        for result in real_results:
            print(f"   {result['dataset']:>15}: "
                  f"diversity={result['overall_diversity']:.3f}, "
                  f"clustering={result['silhouette_score']:.3f}, "
                  f"overlap={result['neighbor_overlap']:.3f}")
        
        print(f"\n🤖 SYNTHETIC DATASETS:")
        for result in synthetic_results:
            print(f"   {result['dataset']:>15}: "
                  f"diversity={result['overall_diversity']:.3f}, "
                  f"clustering={result['silhouette_score']:.3f}, "
                  f"overlap={result['neighbor_overlap']:.3f}")
        
        # Statistical comparison
        if real_results and synthetic_results:
            real_diversity = [r['overall_diversity'] for r in real_results]
            synthetic_diversity = [r['overall_diversity'] for r in synthetic_results]
            
            real_clustering = [r['silhouette_score'] for r in real_results]
            synthetic_clustering = [r['silhouette_score'] for r in synthetic_results]
            
            real_overlap = [r['neighbor_overlap'] for r in real_results]
            synthetic_overlap = [r['neighbor_overlap'] for r in synthetic_results]
            
            print(f"\n📊 STATISTICAL COMPARISON:")
            print(f"   Diversity (CV):")
            print(f"      Real datasets:      {np.mean(real_diversity):.3f} ± {np.std(real_diversity):.3f}")
            print(f"      Synthetic datasets: {np.mean(synthetic_diversity):.3f} ± {np.std(synthetic_diversity):.3f}")
            
            print(f"   Clustering (Silhouette):")
            print(f"      Real datasets:      {np.mean(real_clustering):.3f} ± {np.std(real_clustering):.3f}")
            print(f"      Synthetic datasets: {np.mean(synthetic_clustering):.3f} ± {np.std(synthetic_clustering):.3f}")
            
            print(f"   Neighbor Overlap (Jaccard):")
            print(f"      Real datasets:      {np.mean(real_overlap):.3f} ± {np.std(real_overlap):.3f}")
            print(f"      Synthetic datasets: {np.mean(synthetic_overlap):.3f} ± {np.std(synthetic_overlap):.3f}")
            
            # Hypothesis testing
            diversity_diff = np.mean(real_diversity) - np.mean(synthetic_diversity)
            clustering_diff = np.mean(real_clustering) - np.mean(synthetic_clustering)
            overlap_diff = np.mean(synthetic_overlap) - np.mean(real_overlap)
            
            print(f"\n🎯 HYPOTHESIS TEST RESULTS:")
            if diversity_diff > 0.2:
                print(f"   ✅ DIVERSITY: Real > Synthetic by {diversity_diff:.3f}")
            else:
                print(f"   ❌ DIVERSITY: Difference too small ({diversity_diff:.3f})")
                
            if clustering_diff > 0.1:
                print(f"   ✅ CLUSTERING: Real > Synthetic by {clustering_diff:.3f}")
            else:
                print(f"   ❌ CLUSTERING: Difference too small ({clustering_diff:.3f})")
                
            if overlap_diff > 0.1:
                print(f"   ✅ OVERLAP: Synthetic > Real by {overlap_diff:.3f} (more homogeneous)")
            else:
                print(f"   ❌ OVERLAP: Difference too small ({overlap_diff:.3f})")
            
            # Final verdict
            evidence_count = sum([diversity_diff > 0.2, clustering_diff > 0.1, overlap_diff > 0.1])
            
            print(f"\n🏆 FINAL VERDICT:")
            if evidence_count >= 2:
                print(f"   ✅ HYPOTHESIS CONFIRMED ({evidence_count}/3 metrics support it)")
                print("   ✅ Synthetic datasets lack temporal neighborhood diversity")
                print("   ✅ This explains TGIB's poor performance on synthetic data")
            else:
                print(f"   ❌ HYPOTHESIS NOT CONFIRMED ({evidence_count}/3 metrics support it)")
                print("   ❓ Need to investigate other factors for TGIB's synthetic data failure")

if __name__ == "__main__":
    main() 
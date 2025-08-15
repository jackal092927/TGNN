"""
Analysis of GNN Issue for Triadic Closure Detection

USER'S CRITICAL INSIGHT:
Standard GNNs can't detect common neighbors because they lack node identity!

PROBLEM: When computing embeddings for nodes u and v to check if they should 
be connected, the GNN receives messages from neighbors but doesn't know if 
the same node w sent messages to both u and v.

SOLUTION: Use spectral embeddings as positional encodings to give each node
a unique, learnable identity that can be preserved during message passing.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import pandas as pd

def analyze_gnn_triadic_problem():
    """Analyze why standard GNNs fail at triadic closure detection"""
    
    print("🔍" + "="*70 + "🔍")
    print("ANALYZING GNN TRIADIC CLOSURE PROBLEM")
    print("🔍" + "="*70 + "🔍")
    
    print(f"\n❌ THE FUNDAMENTAL PROBLEM:")
    print(f"   Standard GNNs aggregate neighbor information but lose node identity!")
    
    print(f"\n📊 EXAMPLE SCENARIO:")
    print(f"   Graph: A-C, B-C (C is common neighbor of A and B)")
    print(f"   ")
    print(f"        A     B")
    print(f"         \\   /")
    print(f"          \\ /")
    print(f"           C")
    
    print(f"\n🤖 STANDARD GNN MESSAGE PASSING:")
    print(f"   Step 1: A receives message from C")
    print(f"           h_A^(l+1) = UPDATE(h_A^(l), AGGREGATE([h_C^(l)]))")
    print(f"   ")
    print(f"   Step 2: B receives message from C")  
    print(f"           h_B^(l+1) = UPDATE(h_B^(l), AGGREGATE([h_C^(l)]))")
    print(f"   ")
    print(f"   Step 3: Predict edge A-B")
    print(f"           score = MLP([h_A^(final), h_B^(final)])")
    
    print(f"\n❌ THE CRITICAL ISSUE:")
    print(f"   - A knows it has a neighbor with embedding h_C")
    print(f"   - B knows it has a neighbor with embedding h_C") 
    print(f"   - But A and B don't know they share the SAME node C!")
    print(f"   - The node identity is lost during aggregation")
    
    print(f"\n💡 USER'S SOLUTION INSIGHT:")
    print(f"   'Use spectral embedding as positional encoding and node ID'")
    print(f"   → Give each node a unique, learnable identity")
    print(f"   → Preserve node identity during message passing")
    print(f"   → Allow detection of common neighbors")

def explain_spectral_embedding_solution():
    """Explain how spectral embeddings solve the node identity problem"""
    
    print(f"\n🎯" + "="*70 + "🎯")
    print("SPECTRAL EMBEDDING SOLUTION")
    print("🎯" + "="*70 + "🎯")
    
    print(f"\n📐 SPECTRAL EMBEDDINGS:")
    print(f"   - Compute eigendecomposition of graph Laplacian: L = D - A")
    print(f"   - Use k smallest non-zero eigenvectors as node features")
    print(f"   - Each node gets a unique k-dimensional 'coordinate'")
    print(f"   - These coordinates encode graph structure and node position")
    
    print(f"\n✅ HOW THIS SOLVES THE PROBLEM:")
    print(f"   1. Each node has unique spectral coordinates: spec_u, spec_v, spec_w")
    print(f"   2. During message passing, node identity is preserved:")
    print(f"      h_A^(l+1) = UPDATE(h_A^(l), AGGREGATE([h_C^(l) || spec_C]))")
    print(f"      h_B^(l+1) = UPDATE(h_B^(l), AGGREGATE([h_C^(l) || spec_C]))")
    print(f"   3. When predicting A-B edge, both embeddings contain spec_C")
    print(f"   4. The model can detect: 'A and B both received messages from same spec_C'")
    
    print(f"\n🔧 IMPLEMENTATION STRATEGY:")
    strategies = [
        "1. 📊 Compute spectral embeddings from initial graph",
        "2. 🏗️  Initialize node features: [original_features || spectral_embedding]",
        "3. 🔄 Preserve spectral info during message passing",
        "4. 🎯 Use spectral-aware edge prediction"
    ]
    
    for strategy in strategies:
        print(f"   {strategy}")

def compute_spectral_embeddings(adjacency_matrix, k=8):
    """
    Compute spectral embeddings for nodes
    
    Args:
        adjacency_matrix: scipy sparse matrix or numpy array
        k: number of eigenvectors to use
        
    Returns:
        numpy array of shape (n_nodes, k) with spectral coordinates
    """
    print(f"\n🧮 COMPUTING SPECTRAL EMBEDDINGS:")
    
    # Convert to sparse matrix if needed
    if not hasattr(adjacency_matrix, 'toarray'):
        adjacency_matrix = csr_matrix(adjacency_matrix)
    
    n_nodes = adjacency_matrix.shape[0]
    print(f"   Graph size: {n_nodes} nodes")
    
    # Compute degree matrix
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    degree_matrix = csr_matrix((degrees, (range(n_nodes), range(n_nodes))), 
                              shape=(n_nodes, n_nodes))
    
    # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    # This is more stable than unnormalized Laplacian
    deg_sqrt_inv = np.power(degrees + 1e-12, -0.5)  # Add epsilon for numerical stability
    deg_sqrt_inv_matrix = csr_matrix((deg_sqrt_inv, (range(n_nodes), range(n_nodes))), 
                                    shape=(n_nodes, n_nodes))
    
    normalized_adj = deg_sqrt_inv_matrix @ adjacency_matrix @ deg_sqrt_inv_matrix
    laplacian = csr_matrix(np.eye(n_nodes)) - normalized_adj
    
    print(f"   Computing {k} smallest eigenvectors...")
    
    try:
        # Compute k smallest eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigsh(laplacian, k=k+1, which='SM')
        
        # Remove the first eigenvector (constant vector with eigenvalue 0)
        spectral_embeddings = eigenvectors[:, 1:k+1]
        
        print(f"   Eigenvalues: {eigenvalues[1:k+1]}")
        print(f"   Spectral embedding shape: {spectral_embeddings.shape}")
        
        return spectral_embeddings
        
    except Exception as e:
        print(f"   ⚠️  Eigendecomposition failed: {e}")
        print(f"   Using random embeddings as fallback")
        return np.random.randn(n_nodes, k) * 0.1

def demonstrate_spectral_solution():
    """Demonstrate spectral embeddings on a simple triadic example"""
    
    print(f"\n🧪 DEMONSTRATION ON SIMPLE EXAMPLE:")
    
    # Create simple triadic graph: A-C, B-C, D-C (C is hub)
    nodes = ['A', 'B', 'C', 'D']
    edges = [('A', 'C'), ('B', 'C'), ('D', 'C')]
    
    print(f"   Graph: {edges}")
    print(f"   Expected triadic closures: A-B, A-D, B-D (all through C)")
    
    # Build adjacency matrix
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    for u, v in edges:
        i, j = node_to_idx[u], node_to_idx[v]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    
    print(f"   Adjacency matrix:")
    print(f"   {adj_matrix}")
    
    # Compute spectral embeddings
    spectral_emb = compute_spectral_embeddings(adj_matrix, k=3)
    
    print(f"\n   Spectral embeddings:")
    for i, node in enumerate(nodes):
        print(f"   {node}: {spectral_emb[i]}")
    
    # Analyze spectral distances
    print(f"\n   Spectral distances (L2 norm):")
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i < j:
                dist = np.linalg.norm(spectral_emb[i] - spectral_emb[j])
                print(f"   {node_i}-{node_j}: {dist:.3f}")

def design_spectral_gnn_architecture():
    """Design GNN architecture with spectral positional encodings"""
    
    print(f"\n🏗️" + "="*70 + "🏗️")
    print("SPECTRAL GNN ARCHITECTURE DESIGN")
    print("🏗️" + "="*70 + "🏗️")
    
    print(f"\n📋 ARCHITECTURE COMPONENTS:")
    
    components = [
        {
            "name": "1. Spectral Positional Encoder",
            "purpose": "Compute and embed spectral coordinates",
            "implementation": "SpectralEncoder(adj_matrix, k=8) → spectral_emb"
        },
        {
            "name": "2. Node Feature Initialization", 
            "purpose": "Combine original features with spectral embeddings",
            "implementation": "node_features = [original_feat || spectral_emb || learnable_emb]"
        },
        {
            "name": "3. Spectral-Aware Message Passing",
            "purpose": "Preserve node identity during aggregation",
            "implementation": "message = MLP([h_neighbor || spectral_neighbor])"
        },
        {
            "name": "4. Triadic Edge Predictor",
            "purpose": "Detect common neighbors using spectral info",
            "implementation": "score = MLP([h_u || h_v || spectral_interaction(u,v)])"
        }
    ]
    
    for comp in components:
        print(f"\n   {comp['name']}:")
        print(f"   Purpose: {comp['purpose']}")
        print(f"   Implementation: {comp['implementation']}")
    
    print(f"\n💻 PYTORCH IMPLEMENTATION OUTLINE:")
    
    implementation = """
    class SpectralTriadicGNN(nn.Module):
        def __init__(self, node_feat_dim, spectral_dim=8, hidden_dim=64):
            super().__init__()
            
            # Spectral positional encoding
            self.spectral_dim = spectral_dim
            self.spectral_proj = nn.Linear(spectral_dim, hidden_dim)
            
            # Node feature encoder
            self.node_encoder = nn.Linear(node_feat_dim + hidden_dim, hidden_dim)
            
            # Message passing layers
            self.gnn_layers = nn.ModuleList([
                SpectralGNNLayer(hidden_dim, spectral_dim) 
                for _ in range(3)
            ])
            
            # Edge predictor
            self.edge_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2 + spectral_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, node_features, adj_matrix, spectral_emb):
            # Initialize with spectral positional encoding
            spectral_encoded = self.spectral_proj(spectral_emb)
            h = self.node_encoder(torch.cat([node_features, spectral_encoded], dim=1))
            
            # Message passing with spectral awareness
            for layer in self.gnn_layers:
                h = layer(h, adj_matrix, spectral_emb)
            
            return h
    """
    
    print(implementation)

def main():
    """Main analysis function"""
    
    analyze_gnn_triadic_problem()
    explain_spectral_embedding_solution()
    demonstrate_spectral_solution()
    design_spectral_gnn_architecture()
    
    print(f"\n" + "🎯" + "="*70 + "🎯")
    print("SUMMARY: USER'S INSIGHT IS BRILLIANT!")
    print("🎯" + "="*70 + "🎯")
    
    summary_points = [
        "🔍 Problem identified: Standard GNNs lose node identity during aggregation",
        "💡 Root cause: Can't detect common neighbors without knowing node IDs",
        "🎯 Solution: Spectral embeddings as positional encodings",
        "🏗️  Architecture: Spectral-aware GNN with identity preservation",
        "🚀 Expected result: GNN can now detect triadic closures effectively"
    ]
    
    for point in summary_points:
        print(f"   {point}")
    
    print(f"\n🔧 NEXT STEPS:")
    print(f"   1. Implement SpectralTriadicGNN architecture")
    print(f"   2. Test on triadic_perfect_long_dense dataset") 
    print(f"   3. Compare with rule-based baseline (100% accuracy)")
    print(f"   4. Analyze if spectral embeddings enable triadic detection")

if __name__ == "__main__":
    main()

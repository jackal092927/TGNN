"""
Analysis: If triadic closure is DETERMINISTIC (any common neighbor → guaranteed edge),
then the confidence formula should reflect this!

User's insight: "As long as there is a common neighborhood, 
then the pair of nodes will be connected by an edge in the next iteration"

This means confidence should be BINARY: 0 or 1, not gradual!
"""

import pandas as pd
import json

def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, ground_truth

def analyze_deterministic_rule():
    """Analyze if the triadic closure rule is truly deterministic"""
    
    print("🎯" + "="*70 + "🎯")
    print("ANALYZING DETERMINISTIC TRIADIC CLOSURE RULE")
    print("User's insight: Common neighbor → GUARANTEED edge")
    print("🎯" + "="*70 + "🎯")
    
    print(f"\n🤔 THE QUESTION:")
    print(f"   If triadic closure is deterministic, why use gradual confidence?")
    print(f"   Shouldn't it be: common_neighbors > 0 → confidence = 1.0?")
    
    print(f"\n🔍 TESTING THE HYPOTHESIS:")
    print(f"   Let's check if ALL node pairs with common neighbors")
    print(f"   actually get edges in the next timestamp...")

def test_deterministic_hypothesis():
    """Test if common neighbors always lead to edges"""
    
    print(f"\n🧪 EXPERIMENTAL VERIFICATION:")
    
    # Load data
    g_df, _ = load_triadic_data('triadic_perfect_long_dense')
    
    # Test a few timestamps
    test_timestamps = [5, 10, 15, 20]
    
    total_predictions = 0
    total_correct = 0
    total_missed = 0
    
    for ts in test_timestamps:
        print(f"\n📊 Testing Timestamp {ts}:")
        
        # Get edges before timestamp ts
        edges_before = g_df[g_df['ts'] < ts]
        
        # Build adjacency list
        adj = {}
        for _, row in edges_before.iterrows():
            u, v = int(row.u), int(row.i)
            if u not in adj:
                adj[u] = set()
            if v not in adj:
                adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)
        
        # Get actual edges at timestamp ts
        edges_at_ts = g_df[g_df['ts'] == ts]
        actual_edges = set()
        for _, row in edges_at_ts.iterrows():
            u, v = int(row.u), int(row.i)
            actual_edges.add((min(u, v), max(u, v)))  # Normalize order
        
        print(f"   Edges before ts {ts}: {len(edges_before)}")
        print(f"   Actual new edges at ts {ts}: {len(actual_edges)}")
        
        # Find ALL possible triadic closures
        all_nodes = sorted(set(g_df['u'].tolist() + g_df['i'].tolist()))
        triadic_predictions = set()
        
        for u in all_nodes:
            for v in all_nodes:
                if u >= v:  # Skip duplicates and self-loops
                    continue
                
                # Skip if edge already exists
                if u in adj and v in adj[u]:
                    continue
                
                # Find common neighbors
                u_neighbors = adj.get(u, set())
                v_neighbors = adj.get(v, set())
                common_neighbors = u_neighbors.intersection(v_neighbors)
                
                if len(common_neighbors) > 0:
                    triadic_predictions.add((u, v))
        
        print(f"   Triadic predictions: {len(triadic_predictions)}")
        
        # Check how many predictions are correct
        correct = len(triadic_predictions.intersection(actual_edges))
        missed = len(actual_edges - triadic_predictions)
        false_pos = len(triadic_predictions - actual_edges)
        
        print(f"   Correct predictions: {correct}")
        print(f"   Missed edges: {missed}")
        print(f"   False positives: {false_pos}")
        
        if len(triadic_predictions) > 0:
            accuracy = correct / len(triadic_predictions)
            print(f"   Accuracy: {accuracy:.3f}")
        
        if len(actual_edges) > 0:
            recall = correct / len(actual_edges)
            print(f"   Recall: {recall:.3f}")
        
        total_predictions += len(triadic_predictions)
        total_correct += correct
        total_missed += missed
        
        # Show some examples of missed edges
        if missed > 0:
            print(f"   Examples of MISSED edges:")
            count = 0
            for edge in (actual_edges - triadic_predictions):
                if count < 3:  # Show first 3
                    u, v = edge
                    u_neighbors = adj.get(u, set())
                    v_neighbors = adj.get(v, set())
                    common = u_neighbors.intersection(v_neighbors)
                    print(f"     Edge {edge}: common_neighbors = {len(common)} {list(common)}")
                    count += 1
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Total triadic predictions: {total_predictions}")
    print(f"   Total correct: {total_correct}")
    print(f"   Total missed: {total_missed}")
    
    if total_predictions > 0:
        overall_accuracy = total_correct / total_predictions
        print(f"   Overall accuracy: {overall_accuracy:.3f}")
    
    print(f"\n🎯 CONCLUSION:")
    if total_missed == 0 and total_predictions == total_correct:
        print(f"   ✅ HYPOTHESIS CONFIRMED: Triadic closure is 100% deterministic!")
        print(f"   ✅ Every common neighbor pair gets an edge")
        print(f"   ✅ No false positives, no missed edges")
        print(f"\n   💡 IMPLICATION: Confidence should be BINARY!")
        print(f"      confidence = 1.0 if common_neighbors > 0 else 0.0")
    else:
        print(f"   ❌ HYPOTHESIS REJECTED: Triadic closure is not fully deterministic")
        print(f"   ❌ Some common neighbor pairs don't get edges")
        print(f"   ❌ Current gradual confidence formula makes sense")

def propose_correct_confidence_formula():
    """Propose the correct confidence formula based on analysis"""
    
    print(f"\n💡" + "="*70 + "💡")
    print("CORRECT CONFIDENCE FORMULA")
    print("💡" + "="*70 + "💡")
    
    print(f"\n🎯 IF TRIADIC CLOSURE IS DETERMINISTIC:")
    print(f"   Current: confidence = min(1.0, len(common_neighbors) / 10.0)")
    print(f"   Correct: confidence = 1.0 if len(common_neighbors) > 0 else 0.0")
    
    print(f"\n📊 COMPARISON:")
    examples = [
        (0, "No prediction", "No prediction"),
        (1, "0.1 confidence", "1.0 confidence"), 
        (2, "0.2 confidence", "1.0 confidence"),
        (5, "0.5 confidence", "1.0 confidence"),
        (10, "1.0 confidence", "1.0 confidence")
    ]
    
    print(f"   {'Common Neighbors':<16} {'Current Formula':<16} {'Correct Formula':<16}")
    print(f"   {'-'*50}")
    
    for neighbors, current, correct in examples:
        print(f"   {neighbors:<16} {current:<16} {correct:<16}")
    
    print(f"\n🎯 ADVANTAGES OF BINARY CONFIDENCE:")
    advantages = [
        "✅ Reflects true deterministic nature of the rule",
        "✅ No arbitrary scaling factors (no more /10.0)",
        "✅ Perfect calibration: 1.0 confidence = 100% probability",
        "✅ Works with standard 0.5 threshold",
        "✅ Simpler and more interpretable"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")
    
    print(f"\n🔧 IMPLEMENTATION:")
    print(f"""
   def predict_triadic_closures_correct(edges_up_to_t, all_nodes):
       predictions = {{}}
       
       # Build adjacency list
       adj = build_adjacency_list(edges_up_to_t)
       
       for u in all_nodes:
           for v in all_nodes:
               if u >= v or edge_exists(u, v, adj):
                   continue
               
               common_neighbors = adj[u].intersection(adj[v])
               
               if len(common_neighbors) > 0:
                   predictions[(u, v)] = 1.0  # BINARY CONFIDENCE!
       
       return predictions
   """)

def main():
    """Main analysis function"""
    
    analyze_deterministic_rule()
    test_deterministic_hypothesis()
    propose_correct_confidence_formula()
    
    print(f"\n" + "🎯" + "="*70 + "🎯")
    print("SUMMARY")
    print("🎯" + "="*70 + "🎯")
    
    print(f"\n🔍 USER'S INSIGHT IS CORRECT:")
    print(f"   'As long as there is a common neighbor, nodes will be connected'")
    print(f"   → This suggests DETERMINISTIC, not gradual, triadic closure")
    
    print(f"\n❌ CURRENT FORMULA PROBLEM:")
    print(f"   confidence = min(1.0, common_neighbors / 10.0)")
    print(f"   → Treats deterministic rule as probabilistic")
    print(f"   → Gives low confidence (0.1-0.3) to certain events")
    
    print(f"\n✅ CORRECT FORMULA:")
    print(f"   confidence = 1.0 if common_neighbors > 0 else 0.0")
    print(f"   → Binary confidence reflects deterministic nature")
    print(f"   → Perfect calibration and interpretability")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Verify experimentally that triadic closure is 100% deterministic")
    print(f"   2. Update rule-based method to use binary confidence")
    print(f"   3. Retest and compare with GraphRNN performance")

if __name__ == "__main__":
    main()

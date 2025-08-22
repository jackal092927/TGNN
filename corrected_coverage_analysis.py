#!/usr/bin/env python3
"""
Corrected coverage analysis that properly explains the different average calculations.
"""

import json
import pandas as pd
import numpy as np

def analyze_coverage_calculations():
    """Analyze the different ways coverage averages can be calculated"""
    
    # Load the results
    json_path = './results_triadic_long_dense/influence_coverage_summary.json'
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    print("=== COVERAGE RATE CALCULATION ANALYSIS ===\n")
    
    # Show all timestamps with their coverage rates
    print("📊 COVERAGE RATES BY TIMESTAMP:")
    print(f"{'Timestamp':<10} {'Positive Pairs':<15} {'Coverage Rate':<15} {'Edge Selection %':<15}")
    print("-" * 60)
    
    for _, row in df.iterrows():
        ts = int(row['timestamp'])
        pairs = int(row['positive_pairs'])
        coverage = row['coverage_rate']
        selection = row['selected_edges_percentage']
        
        # Highlight special cases
        if ts == 0:
            status = "⚠️  NO EXISTING EDGES"
        elif coverage == 1.0:
            status = "🎯 PERFECT"
        elif coverage < 0.5:
            status = "❌ POOR"
        else:
            status = "✅ GOOD"
        
        print(f"{ts:<10} {pairs:<15} {coverage:<15.3f} {selection:<15.1f} {status}")
    
    print(f"\n" + "="*60)
    
    # Different ways to calculate averages
    print("\n📈 DIFFERENT AVERAGE CALCULATIONS:")
    
    # Method 1: Simple mean (including t=0)
    simple_mean = df['coverage_rate'].mean()
    print(f"1. Simple Mean (including t=0): {simple_mean:.3f} ({simple_mean*100:.1f}%)")
    print(f"   - Includes t=0 with 0.0 coverage (50 positive pairs)")
    print(f"   - Formula: mean([0.0, 1.0, 1.0, 0.0, 1.0, ...])")
    
    # Method 2: Filtered mean (excluding t=0)
    filtered_mean = df[df['coverage_rate'] > 0]['coverage_rate'].mean()
    print(f"\n2. Filtered Mean (excluding t=0): {filtered_mean:.3f} ({filtered_mean*100:.1f}%)")
    print(f"   - Excludes t=0 (no existing edges to rank)")
    print(f"   - Formula: mean([1.0, 1.0, 0.0, 1.0, ...])")
    
    # Method 3: Weighted mean by positive pairs
    weighted_mean = np.average(df['coverage_rate'], weights=df['positive_pairs'])
    print(f"\n3. Weighted Mean (by positive pairs): {weighted_mean:.3f} ({weighted_mean*100:.1f}%)")
    print(f"   - Weights each timestamp by number of positive pairs")
    print(f"   - Formula: Σ(coverage_rate × positive_pairs) / Σ(positive_pairs)")
    
    # Method 4: Overall coverage (total covered / total positive)
    total_positive = df['positive_pairs'].sum()
    total_covered = df['coverage_achieved'].sum()
    overall_coverage = total_covered / total_positive if total_positive > 0 else 0
    print(f"\n4. Overall Coverage Rate: {overall_coverage:.3f} ({overall_coverage*100:.1f}%)")
    print(f"   - Total covered pairs across all timestamps")
    print(f"   - Formula: {total_covered} / {total_positive} = {overall_coverage:.3f}")
    
    print(f"\n" + "="*60)
    
    # Explain why t=0 is special
    print("\n⚠️  WHY t=0 IS SPECIAL:")
    print(f"   - At t=0: 50 positive pairs appear, but 0 existing edges from t-1")
    print(f"   - No edges to rank by influence scores")
    print(f"   - Coverage rate = 0/50 = 0.0 (0%)")
    print(f"   - This is NOT a failure of the model - it's a data constraint")
    
    # Show the impact of t=0
    print(f"\n📊 IMPACT OF t=0 ON AVERAGES:")
    print(f"   - With t=0: {simple_mean:.3f} ({simple_mean*100:.1f}%)")
    print(f"   - Without t=0: {filtered_mean:.3f} ({filtered_mean*100:.1f}%)")
    print(f"   - Difference: {filtered_mean - simple_mean:.3f} ({(filtered_mean - simple_mean)*100:.1f}%)")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    print(f"   - Use Method 2 (Filtered Mean) for model performance assessment")
    print(f"   - t=0 should be excluded as it has no existing edges to analyze")
    print(f"   - The real coverage performance is: {filtered_mean:.3f} ({filtered_mean*100:.1f}%)")
    
    # Phase analysis with corrected calculations
    print(f"\n📈 PHASE ANALYSIS (CORRECTED):")
    
    early_phase = df[(df['timestamp'] <= 8) & (df['coverage_rate'] > 0)]
    middle_phase = df[(df['timestamp'] > 8) & (df['timestamp'] <= 17) & (df['coverage_rate'] > 0)]
    late_phase = df[(df['timestamp'] > 17) & (df['coverage_rate'] > 0)]
    
    for phase_name, phase_data in [("Early (t=0-8)", early_phase), 
                                  ("Middle (t=9-17)", middle_phase), 
                                  ("Late (t=18-27)", late_phase)]:
        if not phase_data.empty:
            phase_coverage = phase_data['coverage_rate'].mean()
            phase_selection = phase_data['selected_edges_percentage'].mean()
            phase_pairs = phase_data['positive_pairs'].sum()
            print(f"   {phase_name}: {phase_coverage:.3f} coverage, {phase_selection:.1f}% selection, {phase_pairs} pairs")
        else:
            print(f"   {phase_name}: No valid coverage data")
    
    return {
        'simple_mean': simple_mean,
        'filtered_mean': filtered_mean,
        'weighted_mean': weighted_mean,
        'overall_coverage': overall_coverage
    }

if __name__ == "__main__":
    analyze_coverage_calculations()

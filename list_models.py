#!/usr/bin/env python3
"""
Helper script to list available trained models and show evaluation commands.
"""

import os
import glob
import argparse


def parse_model_filename(filename):
    """Parse model filename to extract components."""
    basename = os.path.basename(filename).replace('.pth', '')
    parts = basename.split('-')
    
    if len(parts) >= 4:
        prefix = parts[0] if parts[0] else "(empty)"
        agg_method = parts[1]
        attn_mode = parts[2] 
        dataset = '-'.join(parts[3:-1])  # Handle dataset names with hyphens
        seed = parts[-1]
        
        return {
            'filename': filename,
            'prefix': prefix,
            'agg_method': agg_method,
            'attn_mode': attn_mode,
            'dataset': dataset,
            'seed': seed
        }
    else:
        return None


def list_models(filter_pattern=None):
    """List all available models, optionally filtered by pattern."""
    
    model_files = glob.glob('saved_models/*.pth')
    
    if filter_pattern:
        model_files = [f for f in model_files if filter_pattern in f]
    
    if not model_files:
        print("No model files found.")
        if filter_pattern:
            print(f"Try without filter or check pattern: {filter_pattern}")
        return
    
    print(f"Found {len(model_files)} model file(s):")
    print("=" * 80)
    
    for model_file in sorted(model_files):
        parsed = parse_model_filename(model_file)
        
        if parsed:
            print(f"File: {model_file}")
            print(f"  Prefix: {parsed['prefix']}")
            print(f"  Method: {parsed['agg_method']}-{parsed['attn_mode']}")
            print(f"  Dataset: {parsed['dataset']}")
            print(f"  Seed: {parsed['seed']}")
            print(f"  Evaluation command:")
            print(f"    python learn_edge.py -d {parsed['dataset']} --eval_only --load_model {model_file}")
            print()
        else:
            print(f"File: {model_file} (unknown format)")
            print()


def main():
    parser = argparse.ArgumentParser(description='List available trained models')
    parser.add_argument('--filter', type=str, help='Filter models by pattern (e.g., "synthetic", "ltm")')
    parser.add_argument('--synthetic', action='store_true', help='Show only synthetic dataset models')
    
    args = parser.parse_args()
    
    filter_pattern = args.filter
    if args.synthetic:
        filter_pattern = "synthetic"
    
    list_models(filter_pattern)


if __name__ == "__main__":
    main() 
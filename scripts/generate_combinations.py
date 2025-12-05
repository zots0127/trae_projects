#!/usr/bin/env python3
"""
Generate all combinations for L1=L2 with L3
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def generate_combinations(data_file, output_file):
    """
    Generate all combinations for L1=L2 with L3
    """
    print("=" * 60)
    print("Generate ligand combinations")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data file: {data_file}")
    
    # 1) Load data
    print("\n1) Load data...")
    df = pd.read_csv(data_file)
    print(f"   Original rows: {len(df)}")
    
    # 2) Extract and deduplicate ligands
    print("\n2) Extract and deduplicate ligands...")
    
    # Merge L1 and L2, then deduplicate
    l1_values = df['L1'].dropna().values
    l2_values = df['L2'].dropna().values
    
    # Use numpy.unique for speed
    l12_all = np.concatenate([l1_values, l2_values])
    l12_unique = np.unique(l12_all)
    
    # Deduplicate L3
    l3_unique = df['L3'].dropna().unique()
    
    print(f"   L1 unique: {len(df['L1'].dropna().unique())}")
    print(f"   L2 unique: {len(df['L2'].dropna().unique())}")
    print(f"   L1 U L2 unique ligands: {len(l12_unique)}")
    print(f"   L3 unique: {len(l3_unique)}")
    print(f"   Theoretical combinations: {len(l12_unique) * len(l3_unique):,}")
    
    # 3) Generate combinations (vectorized)
    print("\n3) Generate combinations...")
    
    # Use meshgrid to generate indices for all combinations
    l12_idx, l3_idx = np.meshgrid(range(len(l12_unique)), 
                                   range(len(l3_unique)), 
                                   indexing='ij')
    
    # Flatten and get corresponding SMILES
    l12_flat = l12_unique[l12_idx.flatten()]
    l3_flat = l3_unique[l3_idx.flatten()]
    
    # Create DataFrame (L1=L2)
    result_df = pd.DataFrame({
        'L1': l12_flat,
        'L2': l12_flat,
        'L3': l3_flat
    })
    
    print(f"   Generated combinations: {len(result_df):,}")
    
    # 4) Save results
    print("\n4) Save results...")
    result_df.to_csv(output_file, index=False)
    print(f"   INFO: Saved to: {output_file}")
    
    # 5) Summary
    print("\n" + "=" * 60)
    print("Completion stats")
    print("-" * 60)
    print(f"Total combinations: {len(result_df):,}")
    print(f"File size: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Generate ligand combinations')
    parser.add_argument('--data', '-d', 
                       default='data/Database_normalized.csv',
                       help='Input data file')
    parser.add_argument('--output', '-o', 
                       default='ir_assemble.csv',
                       help='Output file name')
    
    args = parser.parse_args()
    
    # Generate combinations
    generate_combinations(args.data, args.output)

if __name__ == "__main__":
    main()

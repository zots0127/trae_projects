#!/usr/bin/env python3
"""
生成L1=L2与L3的所有组合
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def generate_combinations(data_file, output_file):
    """
    生成L1=L2与L3的所有组合
    """
    print("=" * 60)
    print("生成配体组合")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据文件: {data_file}")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    df = pd.read_csv(data_file)
    print(f"   原始数据: {len(df)} 行")
    
    # 2. 提取并去重配体
    print("\n2. 提取并去重配体...")
    
    # 合并L1和L2，去重
    l1_values = df['L1'].dropna().values
    l2_values = df['L2'].dropna().values
    
    # 使用numpy的unique更快
    l12_all = np.concatenate([l1_values, l2_values])
    l12_unique = np.unique(l12_all)
    
    # L3去重
    l3_unique = df['L3'].dropna().unique()
    
    print(f"   L1配体数: {len(df['L1'].dropna().unique())}")
    print(f"   L2配体数: {len(df['L2'].dropna().unique())}")
    print(f"   L1∪L2去重后: {len(l12_unique)} 个配体")
    print(f"   L3配体数: {len(l3_unique)} 个")
    print(f"   理论组合数: {len(l12_unique) * len(l3_unique):,}")
    
    # 3. 生成组合（向量化操作）
    print("\n3. 生成组合...")
    
    # 使用meshgrid生成所有组合的索引
    l12_idx, l3_idx = np.meshgrid(range(len(l12_unique)), 
                                   range(len(l3_unique)), 
                                   indexing='ij')
    
    # 展平并获取对应的SMILES
    l12_flat = l12_unique[l12_idx.flatten()]
    l3_flat = l3_unique[l3_idx.flatten()]
    
    # 创建DataFrame（L1=L2）
    result_df = pd.DataFrame({
        'L1': l12_flat,
        'L2': l12_flat,  # L1和L2相同
        'L3': l3_flat
    })
    
    print(f"   生成组合数: {len(result_df):,}")
    
    # 4. 保存结果
    print("\n4. 保存结果...")
    result_df.to_csv(output_file, index=False)
    print(f"   ✅ 已保存到: {output_file}")
    
    # 5. 统计信息
    print("\n" + "=" * 60)
    print("完成统计")
    print("-" * 60)
    print(f"总组合数: {len(result_df):,}")
    print(f"文件大小: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='生成配体组合')
    parser.add_argument('--data', '-d', 
                       default='../data/Database_normalized.csv',
                       help='输入数据文件')
    parser.add_argument('--output', '-o', 
                       default='ir_assemble.csv',
                       help='输出文件名')
    
    args = parser.parse_args()
    
    # 生成组合
    generate_combinations(args.data, args.output)

if __name__ == "__main__":
    main()
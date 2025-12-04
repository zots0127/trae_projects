#!/usr/bin/env python3
"""
分析组合数量的详细计算过程
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_FILE = BASE_DIR / "data" / "Database_normalized.csv"
DEFAULT_VIRTUAL_FILE = BASE_DIR / "data" / "ir_assemble.csv"


def analyze_combinations(data_file=None, virtual_file=None):
    """详细分析27万组合的来源"""

    data_path = Path(data_file) if data_file else DEFAULT_DATA_FILE
    virtual_path = Path(virtual_file) if virtual_file else DEFAULT_VIRTUAL_FILE

    print("=" * 80)
    print("组合数量详细分析")
    print("=" * 80)
    
    # 1. 读取原始数据
    print(f"\n1. 读取原始数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   原始数据行数: {len(df):,}")
    
    # 2. 分析L1配体
    print("\n2. 分析L1配体:")
    l1_unique = df['L1'].dropna().unique()
    l1_count = len(l1_unique)
    print(f"   L1去重后数量: {l1_count:,}")
    print(f"   示例: {l1_unique[:3].tolist()}")
    
    # 3. 分析L2配体
    print("\n3. 分析L2配体:")
    l2_unique = df['L2'].dropna().unique()
    l2_count = len(l2_unique)
    print(f"   L2去重后数量: {l2_count:,}")
    print(f"   示例: {l2_unique[:3].tolist()}")
    
    # 4. 分析L3配体
    print("\n4. 分析L3配体:")
    l3_unique = df['L3'].dropna().unique()
    l3_count = len(l3_unique)
    print(f"   L3去重后数量: {l3_count:,}")
    print(f"   示例: {l3_unique[:3].tolist()}")
    
    # 5. L1和L2的并集（用于L1=L2的组合）
    print("\n5. L1和L2的并集分析:")
    l1_l2_union = np.unique(np.concatenate([l1_unique, l2_unique]))
    l1_l2_count = len(l1_l2_union)
    print(f"   L1∪L2去重后数量: {l1_l2_count:,}")
    print(f"   说明: 因为要求L1=L2（使用相同配体），所以从L1和L2的并集中选择")
    
    # 6. 计算总组合数
    print("\n6. 计算总组合数:")
    print(f"   组合方式: L1=L2 (相同配体) × L3 (所有可能)")
    print(f"   L1=L2可选配体数: {l1_l2_count:,}")
    print(f"   L3可选配体数: {l3_count:,}")
    print(f"   总组合数 = {l1_l2_count:,} × {l3_count:,} = {l1_l2_count * l3_count:,}")
    
    # 7. 验证实际生成的文件
    print("\n7. 验证实际生成的组合文件:")
    if virtual_path.exists():
        actual_df = pd.read_csv(virtual_path)
        actual_count = len(actual_df)
        print(f"   {virtual_path.name}实际行数: {actual_count:,}")
        
        # 验证L1=L2
        same_count = (actual_df['L1'] == actual_df['L2']).sum()
        print(f"   L1=L2的行数: {same_count:,}")
        print(f"   验证: {'✅ 所有行都满足L1=L2' if same_count == actual_count else '❌ 存在L1≠L2的行'}")
    
    # 8. 详细统计
    print("\n8. 配体分布统计:")
    print(f"   L1独有配体: {len(set(l1_unique) - set(l2_unique)):,}")
    print(f"   L2独有配体: {len(set(l2_unique) - set(l1_unique)):,}")
    print(f"   L1∩L2共同配体: {len(set(l1_unique) & set(l2_unique)):,}")
    
    print("\n" + "=" * 80)
    print("总结:")
    print(f"  • L1配体库: {l1_count:,} 个")
    print(f"  • L2配体库: {l2_count:,} 个")
    print(f"  • L3配体库: {l3_count:,} 个")
    print(f"  • L1∪L2合并库: {l1_l2_count:,} 个")
    print(f"  • 理论组合数: {l1_l2_count:,} × {l3_count:,} = {l1_l2_count * l3_count:,}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="分析组装组合数量")
    parser.add_argument("--data", default=None,
                        help="原始数据文件路径，默认使用仓库 data/Database_normalized.csv")
    parser.add_argument("--virtual", "--assemble", dest="virtual", default=None,
                        help="虚拟数据库文件路径，默认使用仓库 data/ir_assemble.csv")

    args = parser.parse_args()
    analyze_combinations(data_file=args.data, virtual_file=args.virtual)


if __name__ == "__main__":
    main()

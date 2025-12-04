#!/usr/bin/env python3
"""
使用训练好的模型对所有组合进行完整预测
显示详细进度，不限制输出数量
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
from datetime import datetime
import time
import sys
import json
import platform
import subprocess
import psutil
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.feature_extractor import FeatureExtractor

# 全局性能记录
performance_stats = {
    'start_time': None,
    'end_time': None,
    'steps': [],
    'hardware_info': {}
}

def get_hardware_info():
    """获取硬件信息"""
    info = {}
    
    # 基本系统信息
    info['操作系统'] = platform.system()
    info['系统版本'] = platform.version()
    info['机器架构'] = platform.machine()
    info['处理器'] = platform.processor()
    info['Python版本'] = platform.python_version()
    
    # CPU信息
    try:
        info['CPU物理核心数'] = psutil.cpu_count(logical=False)
        info['CPU逻辑核心数'] = psutil.cpu_count(logical=True)
        info['CPU使用率'] = f"{psutil.cpu_percent(interval=1)}%"
    except:
        pass
    
    # 内存信息
    try:
        mem = psutil.virtual_memory()
        info['总内存'] = f"{mem.total / (1024**3):.1f} GB"
        info['可用内存'] = f"{mem.available / (1024**3):.1f} GB"
        info['内存使用率'] = f"{mem.percent}%"
    except:
        pass
    
    # macOS特定信息
    if platform.system() == 'Darwin':
        try:
            # 获取Mac型号
            result = subprocess.run(['sysctl', '-n', 'hw.model'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                info['Mac型号'] = result.stdout.strip()
            
            # 获取芯片信息
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                info['CPU型号'] = result.stdout.strip()
        except:
            pass
    
    return info

def load_models(project_dir, model_name='xgboost', use_intersection=False):
    """加载训练好的模型
    
    Args:
        project_dir: 项目目录
        model_name: 模型名称
        use_intersection: 是否使用交集训练的模型
    """
    step_start = time.time()
    print("\n" + "="*80)
    print("步骤1: 加载模型")
    print("-"*80)
    
    models = {}
    
    # 尝试多种可能的模型路径
    possible_paths = [
        # AutoML 训练路径
        Path(project_dir) / 'all_models' / 'automl_train' / model_name / 'models',
        Path(project_dir) / '*' / 'automl_train' / model_name / 'models',
        # 标准路径
        Path(project_dir) / model_name / 'models',
        Path(project_dir) / 'models' / model_name,
    ]
    
    # 根据是否使用交集选择模型目录
    if use_intersection:
        # 交集训练的模型通常在 intersection 子目录
        possible_paths.extend([
            Path(project_dir) / model_name / 'intersection' / f'{model_name}_intersection' / 'models',
            Path(project_dir) / model_name / 'intersection' / 'models',
        ])
    
    # 查找存在的模型目录
    model_dir = None
    for path in possible_paths:
        if '*' in str(path):
            # 处理通配符路径
            matches = list(Path(project_dir).glob(str(path.relative_to(Path(project_dir)))))
            if matches:
                model_dir = matches[0]
                break
        elif path.exists():
            model_dir = path
            break
    
    if model_dir is None:
        print(f"❌ 模型目录不存在: {project_dir}/{model_name}/models")
        print(f"  尝试过的路径:")
        for path in possible_paths[:3]:  # 只显示前3个主要路径
            print(f"    • {path}")
        return models
    
    print(f"  📁 找到模型目录: {model_dir}")
    if 'automl_train' in str(model_dir):
        print(f"  📌 使用AutoML训练的模型")
    elif 'intersection' in str(model_dir):
        print(f"  📌 使用交集训练模型")
    else:
        print(f"  📌 使用标准训练模型")
    
    print(f"  📁 模型目录: {model_dir}")
    
    # 查找模型文件（只加载wavelength和PLQY）
    for model_file in model_dir.glob("*.joblib"):
        filename = model_file.stem
        if 'wavelength' in filename.lower():
            models['wavelength'] = joblib.load(model_file)
            print(f"  ✅ 波长模型: {model_file.name}")
        elif 'plqy' in filename.lower():
            models['PLQY'] = joblib.load(model_file)
            print(f"  ✅ PLQY模型: {model_file.name}")
        # 跳过tau模型
    
    print(f"\n成功加载 {len(models)} 个模型")
    
    step_time = time.time() - step_start
    performance_stats['steps'].append({
        'name': '模型加载',
        'time_seconds': step_time,
        'details': f'加载{len(models)}个模型'
    })
    return models

def extract_features_batch(df, feature_type='combined', batch_size=1000):
    """批量提取特征，显示详细进度"""
    step_start = time.time()
    print("\n" + "="*80)
    print("步骤2: 特征提取")
    print("-"*80)
    print(f"配置:")
    print(f"  • 特征类型: {feature_type}")
    print(f"  • 批处理大小: {batch_size:,}")
    print(f"  • 总样本数: {len(df):,}")
    print("\n开始提取...")
    
    extractor = FeatureExtractor(
        feature_type=feature_type,
        morgan_radius=2,
        morgan_bits=1024,
        use_cache=True
    )
    
    n_samples = len(df)
    features_list = []
    valid_indices = []
    failed_count = 0
    
    start_time = time.time()
    
    for i in range(0, n_samples, batch_size):
        batch_start_time = time.time()
        batch_end = min(i + batch_size, n_samples)
        batch_df = df.iloc[i:batch_end]
        
        batch_valid = 0
        for idx, row in batch_df.iterrows():
            try:
                # 提取组合特征
                smiles_list = [row['L1'], row['L2'], row['L3']]
                features = extractor.extract_combination(smiles_list)
                
                if features is not None:
                    features_list.append(features)
                    valid_indices.append(idx)
                    batch_valid += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                continue
        
        # 计算速度和剩余时间
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        processed = batch_end
        rate = processed / elapsed_time if elapsed_time > 0 else 0
        remaining = (n_samples - processed) / rate if rate > 0 else 0
        
        # 显示进度（每1000个样本或每10个批次显示一次）
        if i % (batch_size * 10) == 0 or batch_end == n_samples:
            print(f"\r  进度: {processed:,}/{n_samples:,} ({100*processed/n_samples:.1f}%) | "
                  f"成功: {len(valid_indices):,} | 失败: {failed_count:,} | "
                  f"速度: {rate:.0f} samples/s | "
                  f"剩余时间: {remaining/60:.1f} min", end='', flush=True)
    
    print()  # 换行
    
    total_time = time.time() - start_time
    if features_list:
        X = np.vstack(features_list)
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"\n✅ 特征提取完成:")
        print(f"  • 成功: {len(X):,} 个样本")
        print(f"  • 失败: {failed_count:,} 个样本")
        print(f"  • 成功率: {100*len(X)/n_samples:.1f}%")
        print(f"  • 总用时: {total_time/60:.1f} 分钟")
        print(f"  • 平均速度: {n_samples/total_time:.0f} samples/s")
        
        step_time = time.time() - step_start
        performance_stats['steps'].append({
            'name': '特征提取',
            'time_seconds': step_time,
            'samples_processed': n_samples,
            'samples_success': len(X),
            'samples_failed': failed_count,
            'speed_samples_per_sec': n_samples/total_time,
            'details': f'{len(X):,}/{n_samples:,}样本成功'
        })
        return X, df_valid
    else:
        print(f"\n❌ 特征提取失败，没有有效的特征")
        return None, None

def predict_batch(models, X, df_valid, batch_size=10000):
    """批量预测，显示详细进度"""
    step_start = time.time()
    print("\n" + "="*80)
    print("步骤3: 批量预测")
    print("-"*80)
    print(f"配置:")
    print(f"  • 样本数: {len(X):,}")
    print(f"  • 批大小: {batch_size:,}")
    print(f"  • 模型数: {len(models)}")
    
    predictions = {}
    target_times = {}
    
    # 预测每个目标
    for target_idx, (target, model) in enumerate(models.items(), 1):
        print(f"\n预测目标 {target_idx}/{len(models)}: {target}")
        
        n_samples = len(X)
        preds = []
        
        start_time = time.time()
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            batch_pred = model.predict(batch_X)
            preds.append(batch_pred)
            
            # 计算进度
            elapsed = time.time() - start_time
            rate = batch_end / elapsed if elapsed > 0 else 0
            remaining = (n_samples - batch_end) / rate if rate > 0 else 0
            
            # 显示进度
            print(f"\r  进度: {batch_end:,}/{n_samples:,} ({100*batch_end/n_samples:.1f}%) | "
                  f"速度: {rate:.0f} samples/s | "
                  f"剩余: {remaining:.0f}s", end='', flush=True)
        
        predictions[target] = np.concatenate(preds)
        
        # 记录时间（提高精度）
        target_time = time.time() - start_time
        target_times[target] = target_time
        
        # 统计信息
        print(f"\n  ✅ 完成: {target}")
        print(f"    • 最小值: {predictions[target].min():.6f}")
        print(f"    • 最大值: {predictions[target].max():.6f}")
        print(f"    • 平均值: {predictions[target].mean():.6f}")
        print(f"    • 标准差: {predictions[target].std():.6f}")
        print(f"    • 用时: {target_time:.3f}秒")
        print(f"    • 速度: {n_samples/target_time:.0f} samples/s")
    
    # 添加预测到DataFrame
    print("\n添加预测结果到数据框...")
    if 'wavelength' in predictions:
        df_valid['Predicted_wavelength'] = predictions['wavelength']
    if 'PLQY' in predictions:
        df_valid['Predicted_PLQY'] = predictions['PLQY']
    
    step_time = time.time() - step_start
    prediction_speed = len(X) / step_time if step_time > 0 else 0
    
    performance_stats['steps'].append({
        'name': '批量预测',
        'time_seconds': step_time,
        'samples': len(X),
        'models': len(models),
        'prediction_speed_samples_per_sec': prediction_speed,
        'target_times': target_times,
        'details': f'预测{len(models)}个目标，速度: {prediction_speed:.0f} samples/s'
    })
    
    return df_valid

def analyze_results(df):
    """分析预测结果"""
    print("\n" + "="*80)
    print("步骤4: 结果分析")
    print("-"*80)
    
    if 'Predicted_PLQY' in df.columns:
        print("\n📊 PLQY分布:")
        plqy = df['Predicted_PLQY']
        print(f"  • 最小值: {plqy.min():.4f}")
        print(f"  • 25分位: {plqy.quantile(0.25):.4f}")
        print(f"  • 中位数: {plqy.median():.4f}")
        print(f"  • 75分位: {plqy.quantile(0.75):.4f}")
        print(f"  • 最大值: {plqy.max():.4f}")
        print(f"  • 平均值: {plqy.mean():.4f}")
        print(f"  • 标准差: {plqy.std():.4f}")
        
        # PLQY范围分布
        print("\n  PLQY范围分布:")
        ranges = [
            (0.9, 1.0, "0.9-1.0"),
            (0.8, 0.9, "0.8-0.9"),
            (0.7, 0.8, "0.7-0.8"),
            (0.6, 0.7, "0.6-0.7"),
            (0.5, 0.6, "0.5-0.6"),
            (0.0, 0.5, "0.0-0.5")
        ]
        for min_val, max_val, label in ranges:
            count = ((plqy >= min_val) & (plqy < max_val)).sum()
            pct = 100 * count / len(plqy)
            print(f"    {label}: {count:,} ({pct:.1f}%)")
        
        # Top 10 PLQY
        print("\n🏆 Top 10 PLQY组合:")
        top10 = df.nlargest(10, 'Predicted_PLQY')
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            print(f"\n  #{i}:")
            print(f"    PLQY: {row['Predicted_PLQY']:.4f}")
            if 'Predicted_wavelength' in df.columns:
                print(f"    波长: {row['Predicted_wavelength']:.1f} nm")
            if i <= 3:  # 显示前3个的SMILES
                print(f"    L1/L2: {row['L1'][:60]}...")
                print(f"    L3: {row['L3'][:60]}...")
    
    if 'Predicted_wavelength' in df.columns:
        print("\n📊 波长分布:")
        wl = df['Predicted_wavelength']
        print(f"  • 最小值: {wl.min():.1f} nm")
        print(f"  • 25分位: {wl.quantile(0.25):.1f} nm")
        print(f"  • 中位数: {wl.median():.1f} nm")
        print(f"  • 75分位: {wl.quantile(0.75):.1f} nm")
        print(f"  • 最大值: {wl.max():.1f} nm")
        print(f"  • 平均值: {wl.mean():.1f} nm")

def save_performance_stats(output_dir):
    """保存性能统计表格"""
    print("\n保存性能统计...")
    
    # 创建性能统计表格
    perf_data = []
    for step in performance_stats['steps']:
        row = {
            '步骤': step['name'],
            '耗时(秒)': f"{step['time_seconds']:.3f}",
            '耗时(分钟)': f"{step['time_seconds']/60:.3f}",
            '详细信息': step['details']
        }
        
        # 添加额外信息 - 确保所有字段都有值
        if 'samples_processed' in step:
            row['处理样本数'] = f"{step['samples_processed']:,}"
            row['成功样本数'] = f"{step['samples_success']:,}"
            row['失败样本数'] = f"{step['samples_failed']:,}"
            row['速度(样本/秒)'] = f"{step['speed_samples_per_sec']:.0f}"
        elif 'samples' in step:  # 批量预测步骤
            row['处理样本数'] = f"{step['samples']:,}"
            row['成功样本数'] = f"{step['samples']:,}"
            row['失败样本数'] = '0'
            if 'prediction_speed_samples_per_sec' in step:
                row['速度(样本/秒)'] = f"{step['prediction_speed_samples_per_sec']:.0f}"
            else:
                row['速度(样本/秒)'] = '-'
        else:
            row['处理样本数'] = '-'
            row['成功样本数'] = '-'
            row['失败样本数'] = '-'
            row['速度(样本/秒)'] = '-'
        
        if 'target_times' in step:
            for target, t in step['target_times'].items():
                row[f'{target}预测时间(秒)'] = f"{t:.3f}"
        
        perf_data.append(row)
    
    # 添加总计行
    total_time = performance_stats['end_time'] - performance_stats['start_time']
    perf_data.append({
        '步骤': '总计',
        '耗时(秒)': f"{total_time:.3f}",
        '耗时(分钟)': f"{total_time/60:.3f}",
        '详细信息': f"完成{len(performance_stats['steps'])}个步骤",
        '处理样本数': '-',
        '成功样本数': '-',
        '失败样本数': '-',
        '速度(样本/秒)': '-'
    })
    
    # 保存为CSV
    perf_df = pd.DataFrame(perf_data)
    perf_file = Path(output_dir) / 'performance_statistics.csv'
    perf_df.to_csv(perf_file, index=False, encoding='utf-8-sig')
    print(f"  ✅ 性能统计: {perf_file}")
    
    # 保存硬件信息
    hardware_df = pd.DataFrame([performance_stats['hardware_info']])
    hardware_file = Path(output_dir) / 'hardware_info.csv'
    hardware_df.to_csv(hardware_file, index=False, encoding='utf-8-sig')
    print(f"  ✅ 硬件信息: {hardware_file}")
    
    # 保存为JSON
    json_file = Path(output_dir) / 'performance_statistics.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(performance_stats, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 详细统计: {json_file}")
    
    # 打印性能表格
    print("\n性能统计汇总:")
    print("-" * 80)
    print(perf_df.to_string(index=False))
    print("-" * 80)
    
    # 打印硬件信息
    print("\n硬件配置:")
    print("-" * 80)
    for key, value in performance_stats['hardware_info'].items():
        print(f"{key:20s}: {value}")
    print("-" * 80)
    
    # 打印性能摘要
    print("\n" + "="*80)
    print("🚀 性能指标摘要")
    print("="*80)
    
    # 提取关键性能数据
    feature_speed = 0
    prediction_speed = 0
    total_samples = 0
    
    for step in performance_stats['steps']:
        if step['name'] == '特征提取' and 'speed_samples_per_sec' in step:
            feature_speed = step['speed_samples_per_sec']
            total_samples = step.get('samples_processed', 0)
        elif step['name'] == '批量预测' and 'prediction_speed_samples_per_sec' in step:
            prediction_speed = step['prediction_speed_samples_per_sec']
            if total_samples == 0:
                total_samples = step.get('samples', 0)
    
    total_time = performance_stats['end_time'] - performance_stats['start_time']
    end_to_end_speed = total_samples / total_time if total_time > 0 else 0
    
    print(f"  📊 处理样本数: {total_samples:,}")
    print(f"  ⏱️  总耗时: {total_time:.1f}秒 ({total_time/60:.2f}分钟)")
    print(f"  🔬 特征提取速度: {feature_speed:,.0f} samples/s")
    print(f"  🎯 模型预测速度: {prediction_speed:,.0f} samples/s")
    print(f"  🏁 端到端速度: {end_to_end_speed:,.0f} samples/s")
    
    if prediction_speed > 100000:
        print(f"  ⚡ 超高速预测: {prediction_speed/1000:.0f}K samples/s!")
    
    print("="*80)
    
    # 保存详细性能报告
    performance_report = {
        'summary': {
            'total_samples': total_samples,
            'total_time_seconds': total_time,
            'feature_extraction_speed': feature_speed,
            'prediction_speed': prediction_speed,
            'end_to_end_speed': end_to_end_speed
        },
        'steps': performance_stats['steps'],
        'hardware': performance_stats['hardware_info']
    }
    
    report_file = Path(output_dir) / 'performance_detailed.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(performance_report, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ 详细性能报告: {report_file}")
    
    return perf_df

def save_results(df, output_file):
    """保存结果，包括排序版本"""
    step_start = time.time()
    print("\n" + "="*80)
    print("步骤5: 保存结果")
    print("-"*80)
    
    # 保存完整结果
    print(f"\n保存完整预测结果...")
    df.to_csv(output_file, index=False)
    print(f"  ✅ 文件: {output_file}")
    print(f"  ✅ 行数: {len(df):,}")
    
    # 按PLQY排序保存
    if 'Predicted_PLQY' in df.columns:
        sorted_file = output_file.replace('.csv', '_sorted_by_plqy.csv')
        df_sorted = df.sort_values('Predicted_PLQY', ascending=False)
        df_sorted.to_csv(sorted_file, index=False)
        print(f"  ✅ PLQY排序版: {sorted_file}")
        
        # 保存不同阈值的筛选结果
        thresholds = [0.9, 0.8, 0.7]
        for threshold in thresholds:
            filtered = df[df['Predicted_PLQY'] >= threshold]
            if len(filtered) > 0:
                threshold_file = output_file.replace('.csv', f'_plqy_{threshold:.1f}+.csv')
                filtered.to_csv(threshold_file, index=False)
                print(f"  ✅ PLQY≥{threshold}: {threshold_file} ({len(filtered):,} 个)")
    
    step_time = time.time() - step_start
    performance_stats['steps'].append({
        'name': '结果保存',
        'time_seconds': step_time,
        'details': f'保存{len(df):,}条预测结果'
    })

def main():
    parser = argparse.ArgumentParser(description='预测所有组合性质')
    parser.add_argument('--project', '-p',
                       help='模型项目目录 (如: paper_table_20250912_123547)')
    parser.add_argument('--input', '-i', 
                       help='组合文件 (默认: PROJECT/ir_assemble.csv)')
    parser.add_argument('--output', '-o',
                       help='输出文件 (默认: PROJECT/ir_assemble_predicted_all.csv)')
    parser.add_argument('--intersection', action='store_true',
                       help='使用交集训练的模型（只用三个目标都有值的数据训练）')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='特征提取批处理大小')
    parser.add_argument('--predict-batch', type=int, default=10000,
                       help='预测批处理大小')
    
    args = parser.parse_args()
    
    # 自动检测最新的项目目录
    if not args.project:
        # 查找最新的paper_table目录
        import glob
        project_dirs = sorted(glob.glob('paper_table_*'))
        if project_dirs:
            args.project = project_dirs[-1]  # 使用最新的
            print(f"自动选择最新项目目录: {args.project}")
        else:
            print("❌ 未找到项目目录，请使用 --project 指定")
            return
    
    # 设置默认输入输出路径
    if not args.input:
        args.input = f"{args.project}/ir_assemble.csv"
    if not args.output:
        if args.intersection:
            args.output = f"{args.project}/ir_assemble_predicted_intersection.csv"
        else:
            args.output = f"{args.project}/ir_assemble_predicted_all.csv"
    
    print("="*80)
    print("完整预测流程 - 272,104个组合")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n配置:")
    print(f"  • 项目目录: {args.project}")
    print(f"  • 模型类型: {'交集训练模型' if args.intersection else '完整数据训练模型'}")
    print(f"  • 输入文件: {args.input}")
    print(f"  • 输出文件: {args.output}")
    
    # 记录开始时间和硬件信息
    performance_stats['start_time'] = time.time()
    performance_stats['hardware_info'] = get_hardware_info()
    total_start = time.time()
    
    # 显示硬件信息
    print("\n硬件配置:")
    for key, value in performance_stats['hardware_info'].items():
        print(f"  • {key}: {value}")
    
    # 1. 加载组合
    print(f"\n加载组合文件...")
    df = pd.read_csv(args.input)
    print(f"  ✅ 加载 {len(df):,} 个组合")
    
    # 验证L1=L2
    same_count = (df['L1'] == df['L2']).sum()
    print(f"  ✅ L1=L2验证: {same_count:,}/{len(df):,}")
    
    # 2. 加载模型
    models = load_models(args.project, use_intersection=args.intersection)
    if not models:
        print("❌ 没有找到模型")
        return
    
    # 3. 提取特征
    X, df_valid = extract_features_batch(df, batch_size=args.batch_size)
    if X is None:
        print("❌ 特征提取失败")
        return
    
    # 4. 预测
    df_predicted = predict_batch(models, X, df_valid, batch_size=args.predict_batch)
    
    # 5. 分析结果
    analyze_results(df_predicted)
    
    # 6. 保存结果
    save_results(df_predicted, args.output)
    
    # 记录结束时间
    performance_stats['end_time'] = time.time()
    
    # 7. 保存性能统计
    output_dir = Path(args.output).parent
    save_performance_stats(output_dir)
    
    # 总结
    total_time = time.time() - total_start
    print("\n" + "="*80)
    print("✅ 预测完成!")
    print(f"  • 总用时: {total_time:.3f} 秒 ({total_time/60:.3f} 分钟)")
    print(f"  • 处理速度: {len(df)/total_time:.0f} samples/s")
    print(f"  • 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
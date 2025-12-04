#!/usr/bin/env python3
"""
项目级批量预测器
用于管理和执行整个项目的批量预测任务
"""

import json
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入必要的模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.feature_extractor import FeatureExtractor
from utils.batch_predictor_v2 import BatchPredictorV2
from utils.file_feature_cache import FileFeatureCache
from utils.timing import TimingTracker


class ProjectPredictor:
    """项目级批量预测器"""
    
    def __init__(self, project_dir: str, verbose: bool = True):
        """
        初始化项目预测器
        
        Args:
            project_dir: 项目目录路径
            verbose: 是否显示详细信息
        """
        self.project_dir = Path(project_dir)
        self.verbose = verbose
        
        if not self.project_dir.exists():
            raise ValueError(f"项目目录不存在: {project_dir}")
        
        # 加载项目信息
        self.models = {}
        self.configs = {}
        self.metadata = {}
        
        # 初始化时间追踪器
        self.timing = TimingTracker(f"project_predictor_{project_dir}")
        
        # 扫描并加载所有模型
        with self.timing.measure('load_models'):
            self._load_all_models()
        
        # 初始化批量预测器
        self.batch_predictor = BatchPredictorV2(
            batch_size=5000,
            show_progress=verbose
        )
        
        if self.verbose:
            print(f"✅ 加载项目: {self.project_dir}")
            print(f"   找到 {len(self.models)} 个模型")
    
    def _load_all_models(self):
        """加载项目中的所有模型"""
        # 查找所有模型文件
        model_files = list(self.project_dir.rglob("*.joblib"))
        
        for model_file in model_files:
            # 解析模型信息
            model_name = model_file.stem
            parts = model_name.split('_')
            
            if len(parts) >= 3:
                model_type = parts[0]
                # 查找 'final' 在部分中的位置
                if 'final' in parts:
                    final_idx = parts.index('final')
                    target = '_'.join(parts[1:final_idx])
                else:
                    # 假设最后一部分是时间戳
                    target = '_'.join(parts[1:-1])
                
                # 构建模型键
                key = f"{model_type}_{target}"
                
                # 加载模型
                try:
                    model = joblib.load(model_file)
                    
                    # 创建目标名称映射
                    target_mappings = {
                        'Max_wavelength_nm': 'Max_wavelength(nm)',
                        'tau_sx10-6': 'tau(s*10^-6)',
                        'PLQY': 'PLQY'
                    }
                    original_target = target_mappings.get(target, target)
                    
                    self.models[key] = {
                        'model': model,
                        'path': str(model_file),
                        'type': model_type,
                        'target': target,
                        'original_target': original_target,
                        'name': model_name
                    }
                    
                    # 尝试加载对应的配置
                    config_file = model_file.parent.parent / 'config.yaml'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            self.configs[key] = yaml.safe_load(f)
                    
                    # 尝试加载性能指标
                    # 创建可能的目标名称映射
                    target_mappings = {
                        'Max_wavelength_nm': 'Max_wavelength(nm)',
                        'tau_sx10-6': 'tau(s*10^-6)',
                        'PLQY': 'PLQY'
                    }
                    
                    # 获取原始目标名称
                    original_target = target_mappings.get(target, target)
                    
                    exports_dir = model_file.parent.parent / "exports"
                    summary_files = []
                    if exports_dir.exists():
                        # 使用原始目标名称查找
                        for f in exports_dir.glob(f"{model_type}_*_summary.json"):
                            if original_target in f.name:
                                summary_files.append(f)
                                break
                    if summary_files:
                        with open(summary_files[0], 'r') as f:
                            summary = json.load(f)
                            self.models[key]['performance'] = {
                                'r2': summary.get('mean_r2', 0),
                                'r2_std': summary.get('std_r2', 0),
                                'rmse': summary.get('mean_rmse', 0),
                                'rmse_std': summary.get('std_rmse', 0),
                                'mae': summary.get('mean_mae', 0),
                                'mae_std': summary.get('std_mae', 0)
                            }
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ 无法加载模型 {model_file}: {e}")
    
    def list_models(self) -> pd.DataFrame:
        """
        列出所有可用的模型
        
        Returns:
            包含模型信息的DataFrame
        """
        if not self.models:
            return pd.DataFrame()
        
        data = []
        for key, info in self.models.items():
            perf = info.get('performance', {})
            
            # 格式化带标准差的值
            def format_metric(mean_key, std_key, decimals=4):
                mean_val = perf.get(mean_key, 'N/A')
                std_val = perf.get(std_key, 0)
                if isinstance(mean_val, (int, float)):
                    if std_val > 0:
                        return f"{mean_val:.{decimals}f}±{std_val:.{decimals}f}"
                    else:
                        return f"{mean_val:.{decimals}f}"
                return 'N/A'
            
            row = {
                'Model': info['type'],
                'Target': info.get('original_target', info['target']),
                'R² (mean±std)': format_metric('r2', 'r2_std', 4),
                'RMSE (mean±std)': format_metric('rmse', 'rmse_std', 2),
                'MAE (mean±std)': format_metric('mae', 'mae_std', 2)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        if self.verbose:
            print("\n📊 项目模型列表:")
            print(df.to_string(index=False))
        
        return df
    
    def predict_all_models(self, data_path: str, output_dir: str = None,
                          smiles_columns: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        使用所有模型进行预测
        
        Args:
            data_path: 输入数据文件路径
            output_dir: 输出目录
            smiles_columns: SMILES列名
        
        Returns:
            包含所有预测结果的字典
        """
        # 读取数据
        with self.timing.measure('data_loading', {'file': data_path}):
            df = pd.read_csv(data_path)
        print(f"\n📁 加载数据: {data_path}")
        print(f"   样本数: {len(df)}")
        
        # 记录总体预测的吞吐量
        self.timing.calculate_throughput('data_loading', len(df))
        
        # 设置输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = self.project_dir / f"batch_predictions_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认SMILES列
        if smiles_columns is None:
            smiles_columns = ['L1', 'L2', 'L3']
        
        results = {}
        
        print(f"\n🚀 开始批量预测 ({len(self.models)} 个模型)...")
        
        for i, (key, model_info) in enumerate(self.models.items(), 1):
            print(f"\n[{i}/{len(self.models)}] 预测: {key}")
            
            try:
                with self.timing.measure(f'predict_{key}', {'model': key, 'samples': len(df)}):
                    # 获取配置
                    config = self.configs.get(key, {})
                    feature_config = config.get('feature', {})
                    
                    # 创建特征提取器
                    feature_extractor = FeatureExtractor(
                        feature_type=feature_config.get('feature_type', 'combined'),
                        morgan_bits=feature_config.get('morgan_bits', 1024),
                        morgan_radius=feature_config.get('morgan_radius', 2),
                        use_cache=True
                    )
                    
                    # 进行预测
                    pred_values, failed_indices = self.batch_predictor.predict_with_cache(
                        df=df,
                        model=model_info['model'],
                        feature_extractor=feature_extractor,
                        smiles_columns=smiles_columns,
                        feature_type=feature_config.get('feature_type', 'combined'),
                        combination_method=feature_config.get('combination_method', 'mean'),
                        input_file=str(data_path)
                    )
                    
                    # 创建预测结果DataFrame
                    predictions = df.copy()
                    pred_col = f"Predicted_{model_info.get('original_target', model_info['target'])}"
                    # 直接使用预测值（failed_indices已标记为NaN）
                    predictions[pred_col] = pred_values
                    
                    # 保存结果
                    output_file = output_dir / f"{key}_predictions.csv"
                    predictions.to_csv(output_file, index=False)
                    print(f"   ✅ 保存到: {output_file}")
                    
                    results[key] = predictions
                
                # 计算吞吐量
                self.timing.calculate_throughput(f'predict_{key}', len(df))
                
            except Exception as e:
                print(f"   ❌ 预测失败: {e}")
                continue
        
        # 生成汇总文件
        self._generate_summary(results, output_dir)
        
        print(f"\n✅ 批量预测完成!")
        print(f"   结果目录: {output_dir}")
        
        # 打印时间统计
        if self.verbose:
            print("\n" + "="*50)
            print("⏱️ 预测时间统计")
            print("="*50)
            self.timing.print_summary()
            
            # 保存时间报告
            try:
                timing_file = output_dir / "timing_report.json"
                self.timing.save_report(timing_file, format='json')
                print(f"\n💾 时间报告已保存到: {timing_file}")
            except Exception as e:
                print(f"⚠️ 保存时间报告失败: {e}")
        
        return results
    
    def predict_best_models(self, data_path: str, output_path: str = None,
                           smiles_columns: List[str] = None) -> pd.DataFrame:
        """
        只使用每个目标的最佳模型进行预测
        
        Args:
            data_path: 输入数据文件路径
            output_path: 输出文件路径
            smiles_columns: SMILES列名
        
        Returns:
            包含最佳模型预测结果的DataFrame
        """
        # 找出每个目标的最佳模型
        best_models = self._find_best_models()
        
        if not best_models:
            print("❌ 没有找到性能指标，无法选择最佳模型")
            return pd.DataFrame()
        
        # 读取数据
        df = pd.read_csv(data_path)
        result_df = df.copy()
        
        print(f"\n📁 加载数据: {data_path}")
        print(f"   样本数: {len(df)}")
        
        # 默认SMILES列
        if smiles_columns is None:
            smiles_columns = ['L1', 'L2', 'L3']
        
        print(f"\n🏆 使用最佳模型预测...")
        
        for target, model_key in best_models.items():
            model_info = self.models[model_key]
            print(f"\n目标: {target}")
            print(f"  最佳模型: {model_info['type']} (R²={model_info['performance']['r2']:.4f})")
            
            try:
                # 获取配置
                config = self.configs.get(model_key, {})
                feature_config = config.get('feature', {})
                
                # 创建特征提取器
                feature_extractor = FeatureExtractor(
                    feature_type=feature_config.get('feature_type', 'combined'),
                    morgan_bits=feature_config.get('morgan_bits', 1024),
                    morgan_radius=feature_config.get('morgan_radius', 2),
                    use_cache=True
                )
                
                # 进行预测
                pred_values, failed_indices = self.batch_predictor.predict_with_cache(
                    df=df,
                    model=model_info['model'],
                    feature_extractor=feature_extractor,
                    smiles_columns=smiles_columns,
                    feature_type=feature_config.get('feature_type', 'combined'),
                    combination_method=feature_config.get('combination_method', 'mean'),
                    input_file=str(data_path)
                )
                
                # 添加到结果（直接使用预测值，failed_indices已标记为NaN）
                result_df[f"Best_{target}"] = pred_values
                
            except Exception as e:
                print(f"  ❌ 预测失败: {e}")
                continue
        
        # 保存结果
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"best_predictions_{timestamp}.csv"
        
        result_df.to_csv(output_path, index=False)
        print(f"\n✅ 最佳模型预测完成!")
        print(f"   保存到: {output_path}")
        
        return result_df
    
    def predict_ensemble(self, data_path: str, output_path: str = None,
                        smiles_columns: List[str] = None,
                        method: str = 'mean') -> pd.DataFrame:
        """
        集成预测（多模型平均）
        
        Args:
            data_path: 输入数据文件路径
            output_path: 输出文件路径
            smiles_columns: SMILES列名
            method: 集成方法 ('mean', 'median', 'weighted')
        
        Returns:
            包含集成预测结果的DataFrame
        """
        # 先进行所有模型的预测
        all_predictions = self.predict_all_models(
            data_path=data_path,
            output_dir=None,
            smiles_columns=smiles_columns
        )
        
        if not all_predictions:
            return pd.DataFrame()
        
        # 读取原始数据
        df = pd.read_csv(data_path)
        result_df = df.copy()
        
        print(f"\n🔮 进行集成预测 (方法: {method})...")
        
        # 按目标分组
        targets = {}
        for key in all_predictions.keys():
            target = self.models[key]['target']
            if target not in targets:
                targets[target] = []
            targets[target].append(key)
        
        # 对每个目标进行集成
        for target, model_keys in targets.items():
            print(f"\n目标: {target}")
            print(f"  参与模型: {len(model_keys)}")
            
            # 收集所有预测
            predictions = []
            weights = []
            
            for key in model_keys:
                pred_col = f"Predicted_{target}"
                if pred_col in all_predictions[key].columns:
                    predictions.append(all_predictions[key][pred_col].values)
                    
                    # 如果使用加权平均，使用R²作为权重
                    if method == 'weighted' and 'performance' in self.models[key]:
                        r2 = self.models[key]['performance'].get('r2', 0)
                        weights.append(max(r2, 0))  # 确保权重非负
                    else:
                        weights.append(1.0)
            
            if predictions:
                predictions = np.array(predictions)
                
                if method == 'mean':
                    ensemble_pred = np.mean(predictions, axis=0)
                elif method == 'median':
                    ensemble_pred = np.median(predictions, axis=0)
                elif method == 'weighted':
                    weights = np.array(weights)
                    weights = weights / weights.sum()  # 归一化
                    ensemble_pred = np.average(predictions, axis=0, weights=weights)
                else:
                    raise ValueError(f"不支持的集成方法: {method}")
                
                result_df[f"Ensemble_{target}"] = ensemble_pred
                print(f"  ✅ 集成完成")
        
        # 保存结果
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"ensemble_predictions_{timestamp}.csv"
        
        result_df.to_csv(output_path, index=False)
        print(f"\n✅ 集成预测完成!")
        print(f"   保存到: {output_path}")
        
        return result_df
    
    def _find_best_models(self) -> Dict[str, str]:
        """
        找出每个目标的最佳模型
        
        Returns:
            目标到最佳模型键的映射
        """
        best_models = {}
        
        # 按目标分组 (使用原始目标名称)
        targets = {}
        for key, info in self.models.items():
            target = info.get('original_target', info['target'])
            if target not in targets:
                targets[target] = []
            if 'performance' in info:
                targets[target].append((key, info['performance'].get('r2', -1)))
        
        # 选择每个目标的最佳模型
        for target, models in targets.items():
            if models:
                # 按R²排序，选择最高的
                models.sort(key=lambda x: x[1], reverse=True)
                best_models[target] = models[0][0]
        
        return best_models
    
    def _generate_summary(self, results: Dict[str, pd.DataFrame], output_dir: Path):
        """生成预测结果汇总"""
        summary = {
            'project': str(self.project_dir),
            'timestamp': datetime.now().isoformat(),
            'models_used': len(results),
            'predictions': {}
        }
        
        for key, df in results.items():
            # 找出预测列
            pred_cols = [col for col in df.columns if col.startswith('Predicted_')]
            if pred_cols:
                pred_col = pred_cols[0]
                summary['predictions'][key] = {
                    'file': f"{key}_predictions.csv",
                    'samples': len(df),
                    'mean': float(df[pred_col].mean()),
                    'std': float(df[pred_col].std()),
                    'min': float(df[pred_col].min()),
                    'max': float(df[pred_col].max())
                }
        
        # 保存汇总
        summary_file = output_dir / 'prediction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 汇总文件: {summary_file}")
    
    def get_project_info(self) -> Dict:
        """
        获取项目信息
        
        Returns:
            项目信息字典
        """
        info = {
            'project_path': str(self.project_dir),
            'models_count': len(self.models),
            'models': {},
            'targets': set(),
            'best_models': {}
        }
        
        # 收集模型信息
        for key, model_info in self.models.items():
            model_type = model_info['type']
            target = model_info['target']
            
            info['targets'].add(target)
            
            if model_type not in info['models']:
                info['models'][model_type] = []
            
            info['models'][model_type].append({
                'target': target,
                'performance': model_info.get('performance', {})
            })
        
        # 找出最佳模型
        best = self._find_best_models()
        for target, model_key in best.items():
            model_info = self.models[model_key]
            info['best_models'][target] = {
                'model': model_info['type'],
                'r2': model_info.get('performance', {}).get('r2', 'N/A')
            }
        
        info['targets'] = list(info['targets'])
        
        return info


def main():
    """主函数，用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='项目级批量预测')
    parser.add_argument('project', help='项目目录')
    parser.add_argument('--data', required=True, help='预测数据文件')
    parser.add_argument('--mode', default='all', 
                       choices=['all', 'best', 'ensemble'],
                       help='预测模式')
    parser.add_argument('--output', help='输出路径')
    parser.add_argument('--smiles-columns', help='SMILES列名（逗号分隔）')
    parser.add_argument('--list-models', action='store_true',
                       help='列出所有模型')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = ProjectPredictor(args.project)
    
    if args.list_models:
        predictor.list_models()
        return
    
    # 解析SMILES列
    smiles_columns = None
    if args.smiles_columns:
        smiles_columns = args.smiles_columns.split(',')
    
    # 执行预测
    if args.mode == 'all':
        predictor.predict_all_models(
            data_path=args.data,
            output_dir=args.output,
            smiles_columns=smiles_columns
        )
    elif args.mode == 'best':
        predictor.predict_best_models(
            data_path=args.data,
            output_path=args.output,
            smiles_columns=smiles_columns
        )
    elif args.mode == 'ensemble':
        predictor.predict_ensemble(
            data_path=args.data,
            output_path=args.output,
            smiles_columns=smiles_columns
        )


if __name__ == '__main__':
    main()
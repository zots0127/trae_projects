#!/usr/bin/env python3
"""
训练日志记录模块
类似TensorBoard的全面训练过程记录器，保存所有训练数据用于论文绘图
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import hashlib
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict, field
import yaml


# ========================================
#           数据类定义
# ========================================

@dataclass
class FoldResult:
    """单个折的训练结果"""
    fold_id: int
    train_indices: List[int]
    val_indices: List[int]
    train_predictions: Optional[np.ndarray] = None
    val_predictions: np.ndarray = None
    train_true: Optional[np.ndarray] = None
    val_true: np.ndarray = None
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    training_history: Optional[List[Dict]] = None
    timing: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """转换为字典（处理numpy数组）"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


@dataclass
class ExperimentResult:
    """完整实验结果"""
    experiment_id: str
    timestamp: str
    model_type: str
    target: str
    feature_type: str
    n_samples: int
    n_features: int
    n_folds: int
    fold_results: List[FoldResult]
    final_model_metrics: Optional[Dict[str, float]] = None
    hyperparameters: Dict = field(default_factory=dict)
    data_info: Dict = field(default_factory=dict)
    system_info: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = asdict(self)
        result['fold_results'] = [fold.to_dict() for fold in self.fold_results]
        return result


# ========================================
#           训练记录器主类
# ========================================

class TrainingLogger:
    """训练过程记录器"""
    
    def __init__(self, 
                 project_name: str,
                 base_dir: str = "training_logs",
                 auto_save: bool = True,
                 save_plots: bool = True):
        """
        初始化训练记录器
        
        Args:
            project_name: 项目名称
            base_dir: 基础保存目录
            auto_save: 是否自动保存
            save_plots: 是否保存图表
        """
        self.project_name = project_name
        self.base_dir = Path(base_dir)
        self.auto_save = auto_save
        self.save_plots = save_plots
        
        # 创建项目目录结构
        self.project_dir = self.base_dir / project_name
        self.create_directory_structure()
        
        # 当前实验
        self.current_experiment = None
        self.experiment_history = []
        
        # 实时记录
        self.current_fold_data = {}
        self.global_metrics = {}
        
        print(f"📊 训练记录器初始化")
        print(f"   项目: {project_name}")
        print(f"   保存路径: {self.project_dir}")
    
    def create_directory_structure(self):
        """创建目录结构"""
        directories = [
            self.project_dir,
            self.project_dir / "experiments",
            self.project_dir / "models",
            self.project_dir / "predictions",
            self.project_dir / "plots",
            self.project_dir / "plots" / "fold_results",
            self.project_dir / "plots" / "comparison",
            self.project_dir / "plots" / "feature_importance",
            self.project_dir / "exports",
            self.project_dir / "exports" / "csv",
            self.project_dir / "exports" / "excel",
            self.project_dir / "exports" / "json",
            self.project_dir / "checkpoints"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    #           实验管理
    # ========================================
    
    def start_experiment(self, 
                        model_type: str,
                        target: str,
                        feature_type: str,
                        hyperparameters: Dict,
                        n_folds: int = 10,
                        **kwargs):
        """
        开始新实验
        
        Args:
            model_type: 模型类型
            target: 目标变量
            feature_type: 特征类型
            hyperparameters: 超参数
            n_folds: 交叉验证折数
            **kwargs: 其他信息
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{model_type}_{target}"
        
        self.current_experiment = {
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'model_type': model_type,
            'target': target,
            'feature_type': feature_type,
            'hyperparameters': hyperparameters,
            'n_folds': n_folds,
            'fold_results': [],
            'start_time': datetime.now(),
            'timing': {},
            **kwargs
        }
        
        # 保存实验配置
        config_path = self.project_dir / "experiments" / f"{experiment_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'experiment_id': experiment_id,
                'timestamp': timestamp,
                'model_type': model_type,
                'target': target,
                'feature_type': feature_type,
                'hyperparameters': hyperparameters,
                'n_folds': n_folds,
                **kwargs
            }, f, indent=2)
        
        print(f"\n🚀 实验开始: {experiment_id}")
        
        return experiment_id
    
    def end_experiment(self, final_metrics: Optional[Dict] = None):
        """
        结束当前实验
        
        Args:
            final_metrics: 最终评估指标
        """
        if self.current_experiment is None:
            return
        
        # 计算实验时长
        duration = (datetime.now() - self.current_experiment['start_time']).total_seconds()
        self.current_experiment['duration_seconds'] = duration
        self.current_experiment['final_metrics'] = final_metrics
        
        # 保存完整实验结果
        self.save_experiment_results()
        
        # 生成报告
        self.generate_experiment_report()
        
        # 添加到历史
        self.experiment_history.append(self.current_experiment)
        
        print(f"\n✅ 实验结束: {self.current_experiment['experiment_id']}")
        print(f"   耗时: {duration:.2f}秒")
        
        self.current_experiment = None

    def add_timing(self, key: str, seconds: float):
        """为当前实验添加时间记录"""
        try:
            if self.current_experiment is not None:
                timing = self.current_experiment.get('timing', {})
                timing[key] = float(seconds)
                self.current_experiment['timing'] = timing
        except Exception:
            pass
    
    # ========================================
    #           折记录
    # ========================================
    
    def log_fold_start(self, fold_id: int, train_indices: List[int], val_indices: List[int]):
        """
        记录折开始
        
        Args:
            fold_id: 折编号
            train_indices: 训练集索引
            val_indices: 验证集索引
        """
        self.current_fold_data = {
            'fold_id': fold_id,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'start_time': datetime.now()
        }
        
        print(f"\n  📁 折 {fold_id} 开始")
        print(f"     训练样本: {len(train_indices)}")
        print(f"     验证样本: {len(val_indices)}")
    
    def log_fold_end(self, 
                     y_train: np.ndarray,
                     y_train_pred: np.ndarray,
                     y_val: np.ndarray,
                     y_val_pred: np.ndarray,
                     metrics: Dict[str, float],
                     feature_importance: Optional[Dict] = None,
                     **kwargs):
        """
        记录折结束
        
        Args:
            y_train: 训练集真值
            y_train_pred: 训练集预测值
            y_val: 验证集真值
            y_val_pred: 验证集预测值
            metrics: 评估指标
            feature_importance: 特征重要性
            **kwargs: 其他信息
        """
        if not self.current_fold_data:
            return
        
        # 计算折时长
        duration = (datetime.now() - self.current_fold_data['start_time']).total_seconds()
        
        # 创建折结果
        fold_result = FoldResult(
            fold_id=self.current_fold_data['fold_id'],
            train_indices=self.current_fold_data['train_indices'],
            val_indices=self.current_fold_data['val_indices'],
            train_predictions=y_train_pred,
            val_predictions=y_val_pred,
            train_true=y_train,
            val_true=y_val,
            metrics=metrics,
            feature_importance=feature_importance,
            timing={'duration_seconds': duration}
        )
        
        # 添加到当前实验
        if self.current_experiment:
            self.current_experiment['fold_results'].append(fold_result)
        
        # 保存折数据
        if self.auto_save:
            self.save_fold_data(fold_result)
        
        # 生成折图表
        if self.save_plots:
            self.plot_fold_results(fold_result)
        
        print(f"     ✓ 折 {fold_result.fold_id} 完成 (耗时: {duration:.2f}s)")
        print(f"       验证RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"       验证R²: {metrics.get('r2', 0):.4f}")
    
    # ========================================
    #           数据保存
    # ========================================
    
    def save_fold_data(self, fold_result: FoldResult):
        """保存单折数据"""
        if not self.current_experiment:
            return
        
        exp_id = self.current_experiment['experiment_id']
        fold_id = fold_result.fold_id
        
        # 保存预测结果CSV
        pred_df = pd.DataFrame({
            'fold': fold_id,
            'split': ['train'] * len(fold_result.train_true) + ['val'] * len(fold_result.val_true),
            'true': np.concatenate([fold_result.train_true, fold_result.val_true]),
            'predicted': np.concatenate([fold_result.train_predictions, fold_result.val_predictions]),
            'error': np.concatenate([
                fold_result.train_true - fold_result.train_predictions,
                fold_result.val_true - fold_result.val_predictions
            ])
        })
        
        csv_path = self.project_dir / "predictions" / f"{exp_id}_fold{fold_id}.csv"
        pred_df.to_csv(csv_path, index=False)
        
        # 保存原始numpy数组（用于精确重现）
        np_path = self.project_dir / "predictions" / f"{exp_id}_fold{fold_id}.npz"
        np.savez(np_path,
                train_true=fold_result.train_true,
                train_pred=fold_result.train_predictions,
                val_true=fold_result.val_true,
                val_pred=fold_result.val_predictions,
                train_indices=fold_result.train_indices,
                val_indices=fold_result.val_indices)
    
    def save_experiment_results(self):
        """保存完整实验结果"""
        if not self.current_experiment:
            return
        
        exp_id = self.current_experiment['experiment_id']
        
        # 汇总所有折的结果
        all_val_true = []
        all_val_pred = []
        all_metrics = {key: [] for key in ['rmse', 'mae', 'r2', 'mape']}
        
        for fold_result in self.current_experiment['fold_results']:
            all_val_true.extend(fold_result.val_true)
            all_val_pred.extend(fold_result.val_predictions)
            for key in all_metrics:
                if key in fold_result.metrics:
                    all_metrics[key].append(fold_result.metrics[key])
        
        # 计算汇总指标
        summary_metrics = {
            f"{key}_mean": np.mean(values) if values else 0
            for key, values in all_metrics.items()
        }
        summary_metrics.update({
            f"{key}_std": np.std(values) if values else 0
            for key, values in all_metrics.items()
        })
        
        # 保存汇总CSV
        summary_df = pd.DataFrame({
            'true': all_val_true,
            'predicted': all_val_pred,
            'error': np.array(all_val_true) - np.array(all_val_pred),
            'absolute_error': np.abs(np.array(all_val_true) - np.array(all_val_pred)),
            'percentage_error': np.abs((np.array(all_val_true) - np.array(all_val_pred)) / 
                                     np.array(all_val_true)) * 100
        })
        
        csv_path = self.project_dir / "exports" / "csv" / f"{exp_id}_all_predictions.csv"
        summary_df.to_csv(csv_path, index=False)
        
        cfg = self.current_experiment.get('config', {}) if self.current_experiment else {}
        exp_formats = []
        if isinstance(cfg, dict):
            exp_conf_export = cfg.get('export')
            if isinstance(exp_conf_export, dict):
                exp_formats = exp_conf_export.get('formats', [])
        if 'excel' in exp_formats:
            excel_path = self.project_dir / "exports" / "excel" / f"{exp_id}_results.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='All_Predictions', index=False)
                fold_metrics_df = pd.DataFrame([
                    {
                        'fold': fold.fold_id,
                        **fold.metrics
                    }
                    for fold in self.current_experiment['fold_results']
                ])
                fold_metrics_df.to_excel(writer, sheet_name='Fold_Metrics', index=False)
                summary_metrics_df = pd.DataFrame([summary_metrics])
                summary_metrics_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 保存完整JSON
        json_path = self.project_dir / "exports" / "json" / f"{exp_id}_complete.json"
        with open(json_path, 'w') as f:
            json.dump({
                **{k: v for k, v in self.current_experiment.items() 
                   if k not in ['fold_results', 'start_time']},
                'fold_results': [fold.to_dict() for fold in self.current_experiment['fold_results']],
                'summary_metrics': summary_metrics
            }, f, indent=2)
        
        # 保存pickle（完整Python对象）
        pickle_path = self.project_dir / "experiments" / f"{exp_id}_complete.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.current_experiment, f)
        
        # 保存JSON汇总（用于automl analyze命令）
        json_summary_path = self.project_dir / "exports" / f"{exp_id}_summary.json"
        with open(json_summary_path, 'w') as f:
            json.dump({
                'experiment_id': exp_id,
                'model': self.current_experiment.get('model_type', 'unknown'),
                'target': self.current_experiment.get('target', 'unknown'),
                'feature_type': self.current_experiment.get('feature_type', 'unknown'),
                'n_folds': self.current_experiment.get('n_folds', 0),
                'timestamp': self.current_experiment.get('timestamp', ''),
                'mean_rmse': summary_metrics.get('rmse_mean', 0),
                'std_rmse': summary_metrics.get('rmse_std', 0),
                'mean_mae': summary_metrics.get('mae_mean', 0),
                'std_mae': summary_metrics.get('mae_std', 0),
                'mean_r2': summary_metrics.get('r2_mean', 0),
                'std_r2': summary_metrics.get('r2_std', 0),
                'total_duration': self.current_experiment.get('duration_seconds', 0),
                'hyperparameters': self.current_experiment.get('hyperparameters', {})
            }, f, indent=2)
    
    # ========================================
    #           可视化
    # ========================================
    
    def plot_fold_results(self, fold_result: FoldResult):
        """绘制单折结果图"""
        if not self.current_experiment:
            return
        
        exp_id = self.current_experiment['experiment_id']
        fold_id = fold_result.fold_id
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 散点图：真值 vs 预测值
        ax = axes[0, 0]
        ax.scatter(fold_result.val_true, fold_result.val_predictions, alpha=0.5)
        ax.plot([fold_result.val_true.min(), fold_result.val_true.max()],
                [fold_result.val_true.min(), fold_result.val_true.max()],
                'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'Fold {fold_id}: True vs Predicted')
        ax.text(0.05, 0.95, f"R² = {fold_result.metrics.get('r2', 0):.4f}",
                transform=ax.transAxes, va='top')
        
        # 2. 残差图
        ax = axes[0, 1]
        residuals = fold_result.val_true - fold_result.val_predictions
        ax.scatter(fold_result.val_predictions, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Fold {fold_id}: Residual Plot')
        
        # 3. 误差分布
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, edgecolor='black')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Fold {fold_id}: Error Distribution')
        ax.axvline(x=0, color='r', linestyle='--')
        
        # 4. Q-Q图
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Fold {fold_id}: Q-Q Plot')
        
        plt.suptitle(f'Experiment: {exp_id}', fontsize=14)
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.project_dir / "plots" / "fold_results" / f"{exp_id}_fold{fold_id}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_experiment_summary(self):
        """绘制实验汇总图表"""
        if not self.current_experiment or not self.current_experiment['fold_results']:
            return
        
        exp_id = self.current_experiment['experiment_id']
        
        # 收集所有数据
        all_val_true = []
        all_val_pred = []
        fold_metrics = []
        
        for fold_result in self.current_experiment['fold_results']:
            all_val_true.extend(fold_result.val_true)
            all_val_pred.extend(fold_result.val_predictions)
            fold_metrics.append({
                'fold': fold_result.fold_id,
                **fold_result.metrics
            })
        
        all_val_true = np.array(all_val_true)
        all_val_pred = np.array(all_val_pred)
        
        # 创建汇总图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 总体散点图
        ax = axes[0, 0]
        ax.scatter(all_val_true, all_val_pred, alpha=0.3)
        ax.plot([all_val_true.min(), all_val_true.max()],
                [all_val_true.min(), all_val_true.max()],
                'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title('All Folds: True vs Predicted')
        
        # 2. 各折RMSE
        ax = axes[0, 1]
        fold_df = pd.DataFrame(fold_metrics)
        ax.bar(fold_df['fold'], fold_df['rmse'])
        ax.set_xlabel('Fold')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE by Fold')
        ax.axhline(y=fold_df['rmse'].mean(), color='r', linestyle='--', 
                   label=f"Mean: {fold_df['rmse'].mean():.4f}")
        ax.legend()
        
        # 3. 各折R²
        ax = axes[0, 2]
        ax.bar(fold_df['fold'], fold_df['r2'])
        ax.set_xlabel('Fold')
        ax.set_ylabel('R²')
        ax.set_title('R² by Fold')
        ax.axhline(y=fold_df['r2'].mean(), color='r', linestyle='--',
                   label=f"Mean: {fold_df['r2'].mean():.4f}")
        ax.legend()
        
        # 4. 误差箱线图
        ax = axes[1, 0]
        errors_by_fold = []
        for fold_result in self.current_experiment['fold_results']:
            errors = fold_result.val_true - fold_result.val_predictions
            errors_by_fold.append(errors)
        ax.boxplot(errors_by_fold, labels=range(1, len(errors_by_fold)+1))
        ax.set_xlabel('Fold')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error Distribution by Fold')
        ax.axhline(y=0, color='r', linestyle='--')
        
        # 5. 学习曲线（如果有的话）
        ax = axes[1, 1]
        if fold_df.shape[0] > 1:
            ax.plot(fold_df['fold'], fold_df['rmse'], 'o-', label='Validation RMSE')
            ax.set_xlabel('Fold')
            ax.set_ylabel('RMSE')
            ax.set_title('Cross-Validation Performance')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Not enough folds for learning curve',
                   ha='center', va='center', transform=ax.transAxes)
        
        # 6. 指标汇总
        ax = axes[1, 2]
        ax.axis('off')
        metrics_text = f"""
        Summary Metrics:
        
        RMSE: {fold_df['rmse'].mean():.4f} ± {fold_df['rmse'].std():.4f}
        MAE:  {fold_df['mae'].mean():.4f} ± {fold_df['mae'].std():.4f}
        R²:   {fold_df['r2'].mean():.4f} ± {fold_df['r2'].std():.4f}
        
        Model: {self.current_experiment['model_type']}
        Target: {self.current_experiment['target']}
        Feature: {self.current_experiment['feature_type']}
        Folds: {self.current_experiment['n_folds']}
        """
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
                fontsize=10, va='top', family='monospace')
        
        plt.suptitle(f'Experiment Summary: {exp_id}', fontsize=14)
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.project_dir / "plots" / f"{exp_id}_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ========================================
    #           报告生成
    # ========================================
    
    def generate_experiment_report(self):
        """生成实验HTML报告"""
        if not self.current_experiment:
            return
        
        exp_id = self.current_experiment['experiment_id']
        
        # 生成汇总图表
        self.plot_experiment_summary()
        
        # 收集数据
        fold_metrics = pd.DataFrame([
            {
                'fold': fold.fold_id,
                **fold.metrics
            }
            for fold in self.current_experiment['fold_results']
        ])
        
        # 生成表格行
        table_rows = []
        for _, row in fold_metrics.iterrows():
            table_rows.append(f"""
                <tr>
                    <td>{row['fold']}</td>
                    <td>{row['rmse']:.4f}</td>
                    <td>{row['mae']:.4f}</td>
                    <td>{row['r2']:.4f}</td>
                    <td>{row.get('mape', 0):.2f}</td>
                </tr>
            """)
        table_rows_html = ''.join(table_rows)
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report: {exp_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-box {{ 
                    display: inline-block; 
                    padding: 10px; 
                    margin: 5px;
                    background: #f0f0f0; 
                    border-radius: 5px;
                }}
                .plot {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Training Report: {exp_id}</h1>
            
            <h2>Experiment Configuration</h2>
            <div class="metric-box">
                <strong>Model:</strong> {self.current_experiment['model_type']}<br>
                <strong>Target:</strong> {self.current_experiment['target']}<br>
                <strong>Feature Type:</strong> {self.current_experiment['feature_type']}<br>
                <strong>Cross-Validation:</strong> {self.current_experiment['n_folds']} folds<br>
                <strong>Duration:</strong> {self.current_experiment.get('duration_seconds', 0):.2f} seconds
            </div>
            
            <h2>Performance Summary</h2>
            <div class="metric-box">
                <strong>RMSE:</strong> {fold_metrics['rmse'].mean():.4f} ± {fold_metrics['rmse'].std():.4f}<br>
                <strong>MAE:</strong> {fold_metrics['mae'].mean():.4f} ± {fold_metrics['mae'].std():.4f}<br>
                <strong>R²:</strong> {fold_metrics['r2'].mean():.4f} ± {fold_metrics['r2'].std():.4f}
            </div>
            
            <h2>Fold-by-Fold Results</h2>
            <table>
                <tr>
                    <th>Fold</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R²</th>
                    <th>MAPE (%)</th>
                </tr>
                {table_rows_html}
            </table>
            
            <h2>Visualizations</h2>
            <div class="plot">
                <img src="../plots/{exp_id}_summary.png" alt="Summary Plot">
            </div>
            
            <h2>Hyperparameters</h2>
            <pre>{json.dumps(self.current_experiment.get('hyperparameters', {}), indent=2)}</pre>
            
            <hr>
            <p><small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </body>
        </html>
        """
        
        report_path = self.project_dir / "exports" / f"{exp_id}_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"   📄 报告已生成: {report_path}")
    
    # ========================================
    #           比较功能
    # ========================================
    
    def compare_experiments(self, experiment_ids: List[str] = None):
        """
        比较多个实验
        
        Args:
            experiment_ids: 要比较的实验ID列表，None则比较所有
        """
        # 加载所有实验
        experiments = []
        exp_dir = self.project_dir / "experiments"
        
        for pkl_file in exp_dir.glob("*_complete.pkl"):
            with open(pkl_file, 'rb') as f:
                exp = pickle.load(f)
                if experiment_ids is None or exp['experiment_id'] in experiment_ids:
                    experiments.append(exp)
        
        if not experiments:
            print("没有找到实验数据")
            return
        
        # 创建比较表
        comparison_data = []
        for exp in experiments:
            fold_results = exp['fold_results']
            metrics = pd.DataFrame([fold.metrics for fold in fold_results])
            
            comparison_data.append({
                'experiment_id': exp['experiment_id'],
                'model': exp['model_type'],
                'target': exp['target'],
                'feature': exp['feature_type'],
                'rmse_mean': metrics['rmse'].mean(),
                'rmse_std': metrics['rmse'].std(),
                'r2_mean': metrics['r2'].mean(),
                'r2_std': metrics['r2'].std(),
                'duration': exp.get('duration_seconds', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较结果
        comparison_path = self.project_dir / "exports" / "experiment_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # 生成比较图表
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE比较
        ax = axes[0]
        x = range(len(comparison_df))
        ax.bar(x, comparison_df['rmse_mean'], yerr=comparison_df['rmse_std'], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45)
        ax.set_ylabel('RMSE')
        ax.set_title('Model Comparison: RMSE')
        
        # R²比较
        ax = axes[1]
        ax.bar(x, comparison_df['r2_mean'], yerr=comparison_df['r2_std'], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45)
        ax.set_ylabel('R²')
        ax.set_title('Model Comparison: R²')
        
        plt.tight_layout()
        comparison_plot_path = self.project_dir / "plots" / "comparison" / "model_comparison.png"
        plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 比较结果已保存: {comparison_path}")
        
        return comparison_df
    
    # ========================================
    #           实用方法
    # ========================================
    
    def get_best_model(self, metric: str = 'rmse', ascending: bool = True):
        """
        获取最佳模型
        
        Args:
            metric: 评估指标
            ascending: 是否升序（True表示越小越好）
        
        Returns:
            最佳实验信息
        """
        experiments = []
        exp_dir = self.project_dir / "experiments"
        
        for pkl_file in exp_dir.glob("*_complete.pkl"):
            with open(pkl_file, 'rb') as f:
                exp = pickle.load(f)
                fold_results = exp['fold_results']
                metrics = pd.DataFrame([fold.metrics for fold in fold_results])
                exp['mean_' + metric] = metrics[metric].mean()
                experiments.append(exp)
        
        if not experiments:
            return None
        
        # 排序
        experiments.sort(key=lambda x: x['mean_' + metric], reverse=not ascending)
        best_exp = experiments[0]
        
        print(f"🏆 最佳模型 ({metric}):")
        print(f"   实验ID: {best_exp['experiment_id']}")
        print(f"   模型: {best_exp['model_type']}")
        print(f"   {metric}: {best_exp['mean_' + metric]:.4f}")
        
        return best_exp
    
    def export_for_paper(self, experiment_id: str, output_dir: str = None):
        """
        导出用于论文的数据
        
        Args:
            experiment_id: 实验ID
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = self.project_dir / "exports" / "paper_ready"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载实验
        pkl_path = self.project_dir / "experiments" / f"{experiment_id}_complete.pkl"
        with open(pkl_path, 'rb') as f:
            exp = pickle.load(f)
        
        # 导出预测数据
        all_true = []
        all_pred = []
        for fold in exp['fold_results']:
            all_true.extend(fold.val_true)
            all_pred.extend(fold.val_predictions)
        
        pred_df = pd.DataFrame({
            'true_value': all_true,
            'predicted_value': all_pred
        })
        pred_df.to_csv(output_dir / f"{experiment_id}_predictions.csv", index=False)
        
        # 导出指标
        metrics_df = pd.DataFrame([fold.metrics for fold in exp['fold_results']])
        metrics_df.to_csv(output_dir / f"{experiment_id}_fold_metrics.csv", index=False)
        
        # 复制图表
        plot_src = self.project_dir / "plots" / f"{experiment_id}_summary.png"
        if plot_src.exists():
            shutil.copy(plot_src, output_dir / f"{experiment_id}_summary.png")
        
        print(f"   📦 论文数据已导出: {output_dir}")
        
        return output_dir


# ========================================
#           便捷函数
# ========================================

def create_logger(project_name: str, **kwargs) -> TrainingLogger:
    """创建训练记录器的便捷函数"""
    return TrainingLogger(project_name, **kwargs)


def load_experiment(experiment_path: Union[str, Path]) -> Dict:
    """加载实验结果"""
    with open(experiment_path, 'rb') as f:
        return pickle.load(f)


def plot_paper_figure(true_values: np.ndarray, 
                      predicted_values: np.ndarray,
                      title: str = None,
                      save_path: str = None):
    """
    生成论文级别的图表
    
    Args:
        true_values: 真实值
        predicted_values: 预测值
        title: 图表标题
        save_path: 保存路径
    """
    # 设置论文风格
    plt.style.use('seaborn-v0_8-paper')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 散点图
    ax.scatter(true_values, predicted_values, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    
    # 对角线
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # 计算R²
    from sklearn.metrics import r2_score
    r2 = r2_score(true_values, predicted_values)
    
    # 设置标签
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    
    # 添加R²文本
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    # 测试代码
    logger = TrainingLogger("test_project")
    
    # 模拟实验
    logger.start_experiment(
        model_type="xgboost",
        target="wavelength",
        feature_type="morgan",
        hyperparameters={"n_estimators": 100, "max_depth": 6},
        n_folds=3
    )
    
    # 模拟折训练
    for fold in range(3):
        logger.log_fold_start(fold, list(range(80)), list(range(80, 100)))
        
        # 模拟数据
        y_train = np.random.randn(80)
        y_train_pred = y_train + np.random.randn(80) * 0.1
        y_val = np.random.randn(20)
        y_val_pred = y_val + np.random.randn(20) * 0.1
        
        metrics = {
            'rmse': np.sqrt(np.mean((y_val - y_val_pred)**2)),
            'mae': np.mean(np.abs(y_val - y_val_pred)),
            'r2': 1 - np.sum((y_val - y_val_pred)**2) / np.sum((y_val - y_val.mean())**2)
        }
        
        logger.log_fold_end(y_train, y_train_pred, y_val, y_val_pred, metrics)
    
    logger.end_experiment()
    print("\n✅ 测试完成")

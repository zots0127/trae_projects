#!/usr/bin/env python3
"""
训练曲线保存和可视化模块
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


class TrainingCurveRecorder:
    """训练曲线记录器"""
    
    def __init__(self, save_dir: Path, model_name: str, target: str):
        """
        初始化记录器
        
        Args:
            save_dir: 保存目录
            model_name: 模型名称
            target: 目标变量名
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.target = target
        
        # 存储训练数据
        self.fold_data = []
        self.current_fold = None
        
        # 创建子目录
        self.curves_dir = self.save_dir / "training_curves"
        self.curves_dir.mkdir(exist_ok=True)
        
    def start_fold(self, fold_idx: int):
        """开始新的折"""
        self.current_fold = {
            'fold': fold_idx,
            'train_scores': [],
            'val_scores': [],
            'iterations': [],
            'train_metrics': {},
            'val_metrics': {}
        }
    
    def add_iteration(self, iteration: int, train_score: float, val_score: float):
        """添加一次迭代的分数"""
        if self.current_fold is not None:
            self.current_fold['iterations'].append(iteration)
            self.current_fold['train_scores'].append(train_score)
            self.current_fold['val_scores'].append(val_score)
    
    def end_fold(self, train_metrics: Dict, val_metrics: Dict):
        """结束当前折"""
        if self.current_fold is not None:
            self.current_fold['train_metrics'] = train_metrics
            self.current_fold['val_metrics'] = val_metrics
            self.fold_data.append(self.current_fold)
            self.current_fold = None
    
    def save_curves(self):
        """保存训练曲线数据和图表"""
        if not self.fold_data:
            return
        
        # 保存原始数据为JSON
        data_file = self.curves_dir / f"{self.model_name}_{self.target}_curves.json"
        with open(data_file, 'w') as f:
            json.dump({
                'model': self.model_name,
                'target': self.target,
                'timestamp': datetime.now().isoformat(),
                'folds': self.fold_data
            }, f, indent=2)
        
        # 生成并保存可视化图表
        self._generate_plots()
        
        return data_file
    
    def _generate_plots(self):
        """生成训练曲线图表"""
        if not self.fold_data:
            return
        
        # 创建子图：每个折一个
        n_folds = len(self.fold_data)
        fig = make_subplots(
            rows=(n_folds + 1) // 2,
            cols=2,
            subplot_titles=[f"Fold {d['fold']}" for d in self.fold_data],
            vertical_spacing=0.1
        )
        
        # 为每个折添加曲线
        for idx, fold_data in enumerate(self.fold_data):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            if fold_data['iterations']:
                # 训练曲线
                fig.add_trace(
                    go.Scatter(
                        x=fold_data['iterations'],
                        y=fold_data['train_scores'],
                        mode='lines',
                        name=f'Train (Fold {fold_data["fold"]})',
                        line=dict(color='blue', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
                
                # 验证曲线
                fig.add_trace(
                    go.Scatter(
                        x=fold_data['iterations'],
                        y=fold_data['val_scores'],
                        mode='lines',
                        name=f'Val (Fold {fold_data["fold"]})',
                        line=dict(color='red', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
        
        # 更新布局
        fig.update_layout(
            title=f"{self.model_name} - {self.target} Training Curves",
            height=300 * ((n_folds + 1) // 2),
            showlegend=True,
            hovermode='x unified'
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text="Score")
        
        # 保存HTML
        html_file = self.curves_dir / f"{self.model_name}_{self.target}_curves.html"
        fig.write_html(str(html_file))
        
        # 同时生成汇总图（平均曲线）
        self._generate_average_plot()
    
    def _generate_average_plot(self):
        """生成平均训练曲线"""
        # 收集所有折的数据
        all_iterations = []
        all_train_scores = []
        all_val_scores = []
        
        for fold_data in self.fold_data:
            if fold_data['iterations']:
                all_iterations.append(fold_data['iterations'])
                all_train_scores.append(fold_data['train_scores'])
                all_val_scores.append(fold_data['val_scores'])
        
        if not all_iterations:
            return
        
        # 找到最小长度
        min_len = min(len(iters) for iters in all_iterations)
        
        # 截断到相同长度并计算平均
        train_mean = np.mean([scores[:min_len] for scores in all_train_scores], axis=0)
        train_std = np.std([scores[:min_len] for scores in all_train_scores], axis=0)
        val_mean = np.mean([scores[:min_len] for scores in all_val_scores], axis=0)
        val_std = np.std([scores[:min_len] for scores in all_val_scores], axis=0)
        iterations = all_iterations[0][:min_len]
        
        # 创建图表
        fig = go.Figure()
        
        # 训练曲线（带置信区间）
        fig.add_trace(go.Scatter(
            x=iterations,
            y=train_mean,
            mode='lines',
            name='Train (mean)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=iterations + iterations[::-1],
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Train std'
        ))
        
        # 验证曲线（带置信区间）
        fig.add_trace(go.Scatter(
            x=iterations,
            y=val_mean,
            mode='lines',
            name='Validation (mean)',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=iterations + iterations[::-1],
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(200,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Val std'
        ))
        
        # 更新布局
        fig.update_layout(
            title=f"{self.model_name} - {self.target} Average Training Curves",
            xaxis_title="Iteration",
            yaxis_title="Score",
            hovermode='x unified',
            showlegend=True
        )
        
        # 保存
        html_file = self.curves_dir / f"{self.model_name}_{self.target}_avg_curves.html"
        fig.write_html(str(html_file))


class TrainingCurveAggregator:
    """训练曲线聚合器 - 用于比较多个模型"""
    
    @staticmethod
    def aggregate_curves(curves_dir: Path, output_file: Optional[Path] = None):
        """
        聚合所有模型的训练曲线
        
        Args:
            curves_dir: 曲线数据目录
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = curves_dir / "all_models_comparison.html"
        
        # 读取所有JSON文件
        json_files = list(curves_dir.glob("*_curves.json"))
        if not json_files:
            return
        
        # 创建比较图
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["All Models Comparison"]
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        for idx, json_file in enumerate(json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            model_name = data['model']
            target = data['target']
            color = colors[idx % len(colors)]
            
            # 计算平均曲线
            all_val_scores = []
            for fold_data in data['folds']:
                if fold_data['val_scores']:
                    all_val_scores.append(fold_data['val_scores'])
            
            if all_val_scores:
                min_len = min(len(scores) for scores in all_val_scores)
                val_mean = np.mean([scores[:min_len] for scores in all_val_scores], axis=0)
                iterations = list(range(min_len))
                
                fig.add_trace(go.Scatter(
                    x=iterations,
                    y=val_mean,
                    mode='lines',
                    name=f"{model_name}_{target}",
                    line=dict(color=color, width=2)
                ))
        
        # 更新布局
        fig.update_layout(
            title="Model Training Curves Comparison",
            xaxis_title="Iteration",
            yaxis_title="Validation Score",
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        # 保存
        fig.write_html(str(output_file))
        
        return output_file
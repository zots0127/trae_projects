#!/usr/bin/env python3
"""
特征重要性保存和可视化模块
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import plotly.graph_objects as go
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy类型"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class FeatureImportanceRecorder:
    """特征重要性记录器"""
    
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
        
        # 创建子目录
        self.importance_dir = self.save_dir / "feature_importance"
        self.importance_dir.mkdir(exist_ok=True)
        
        # 存储数据
        self.importance_data = []
        
    def add_fold_importance(self, fold_idx: int, importance_dict: Dict[str, float], 
                           feature_names: Optional[List[str]] = None):
        """
        添加一个折的特征重要性
        
        Args:
            fold_idx: 折索引
            importance_dict: 特征重要性字典或数组
            feature_names: 特征名称列表
        """
        if isinstance(importance_dict, np.ndarray):
            # 如果是数组，转换为字典
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance_dict))]
            # 转换为Python原生类型
            importance_dict = {
                name: float(val) for name, val in zip(feature_names, importance_dict)
            }
        else:
            # 确保字典中的值都是Python原生类型
            importance_dict = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in importance_dict.items()
            }
        
        self.importance_data.append({
            'fold': int(fold_idx),
            'importance': importance_dict
        })
    
    def save_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        保存特征重要性数据和图表
        
        Args:
            feature_names: 特征名称列表（如果之前没有提供）
            
        Returns:
            保存的文件路径字典
        """
        if not self.importance_data:
            return {}
        
        saved_files = {}
        
        # 计算平均特征重要性
        avg_importance = self._calculate_average_importance()
        
        # 保存为JSON（使用自定义编码器处理NumPy类型）
        json_file = self.importance_dir / f"{self.model_name}_{self.target}_importance.json"
        with open(json_file, 'w') as f:
            json.dump({
                'model': self.model_name,
                'target': self.target,
                'timestamp': datetime.now().isoformat(),
                'average_importance': avg_importance,
                'fold_importance': self.importance_data
            }, f, indent=2, cls=NumpyEncoder)
        saved_files['json'] = json_file
        
        # 保存为CSV
        csv_file = self.importance_dir / f"{self.model_name}_{self.target}_importance.csv"
        df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in avg_importance.items()
        ])
        df = df.sort_values('importance', ascending=False)
        df.to_csv(csv_file, index=False)
        saved_files['csv'] = csv_file
        
        # 生成可视化
        html_file = self._generate_plot(avg_importance)
        if html_file:
            saved_files['html'] = html_file
        
        print(f"   💾 特征重要性已保存:")
        print(f"      - CSV: {csv_file.name}")
        print(f"      - JSON: {json_file.name}")
        if html_file:
            print(f"      - 图表: {html_file.name}")
        
        return saved_files
    
    def _calculate_average_importance(self) -> Dict[str, float]:
        """计算平均特征重要性"""
        if not self.importance_data:
            return {}
        
        # 收集所有特征
        all_features = set()
        for data in self.importance_data:
            all_features.update(data['importance'].keys())
        
        # 计算平均值
        avg_importance = {}
        for feature in all_features:
            values = []
            for data in self.importance_data:
                if feature in data['importance']:
                    values.append(data['importance'][feature])
            if values:
                avg_importance[feature] = np.mean(values)
        
        return avg_importance
    
    def _generate_plot(self, importance_dict: Dict[str, float], 
                      top_n: int = 20) -> Optional[Path]:
        """
        生成特征重要性图表
        
        Args:
            importance_dict: 特征重要性字典
            top_n: 显示前N个重要特征
            
        Returns:
            图表文件路径
        """
        if not importance_dict:
            return None
        
        # 排序并取前N个
        sorted_items = sorted(importance_dict.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:top_n]
        
        if not sorted_items:
            return None
        
        features = [item[0] for item in sorted_items]
        importances = [item[1] for item in sorted_items]
        
        # 创建条形图
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{v:.4f}' for v in importances],
            textposition='outside'
        ))
        
        # 更新布局
        fig.update_layout(
            title=f"{self.model_name} - {self.target} Feature Importance (Top {len(features)})",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(features) * 25),
            margin=dict(l=200),
            showlegend=False,
            yaxis=dict(autorange="reversed")  # 最重要的在顶部
        )
        
        # 保存HTML
        html_file = self.importance_dir / f"{self.model_name}_{self.target}_importance.html"
        fig.write_html(str(html_file))
        
        return html_file
    
    @staticmethod
    def extract_importance_from_model(model: Any) -> Optional[np.ndarray]:
        """
        从模型中提取特征重要性
        
        Args:
            model: 训练好的模型
            
        Returns:
            特征重要性数组，如果模型不支持则返回None
        """
        # 检查各种可能的属性名
        importance_attrs = [
            'feature_importances_',  # sklearn树模型
            'feature_importance',     # LightGBM
            'get_feature_importance', # CatBoost方法
            'feature_importances',    # 某些自定义模型
        ]
        
        for attr in importance_attrs:
            if hasattr(model, attr):
                importance = getattr(model, attr)
                if callable(importance):
                    # 如果是方法，调用它
                    try:
                        return importance()
                    except:
                        continue
                else:
                    # 如果是属性，直接返回
                    return importance
        
        # XGBoost特殊处理
        if hasattr(model, 'get_score'):
            try:
                scores = model.get_score(importance_type='gain')
                if scores:
                    # 转换为数组格式
                    max_idx = max(int(k[1:]) for k in scores.keys())
                    importance = np.zeros(max_idx + 1)
                    for k, v in scores.items():
                        idx = int(k[1:])  # 'f0' -> 0
                        importance[idx] = v
                    return importance
            except:
                pass
        
        return None


class FeatureImportanceAggregator:
    """特征重要性聚合器 - 用于比较多个模型"""
    
    @staticmethod
    def compare_models(importance_dir: Path, output_file: Optional[Path] = None):
        """
        比较多个模型的特征重要性
        
        Args:
            importance_dir: 特征重要性数据目录
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = importance_dir / "models_importance_comparison.html"
        
        # 读取所有CSV文件
        csv_files = list(importance_dir.glob("*_importance.csv"))
        if not csv_files:
            return None
        
        # 收集数据
        all_data = {}
        for csv_file in csv_files:
            # 从文件名提取模型和目标
            parts = csv_file.stem.replace('_importance', '').split('_')
            model_name = parts[0]
            target = '_'.join(parts[1:])
            
            df = pd.read_csv(csv_file)
            # 取前10个最重要的特征
            df_top = df.head(10)
            
            key = f"{model_name}_{target}"
            all_data[key] = df_top
        
        if not all_data:
            return None
        
        # 创建子图
        from plotly.subplots import make_subplots
        
        n_models = len(all_data)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=list(all_data.keys()),
            horizontal_spacing=0.1
        )
        
        # 为每个模型添加条形图
        for idx, (key, df) in enumerate(all_data.items(), 1):
            fig.add_trace(
                go.Bar(
                    x=df['importance'].values,
                    y=df['feature'].values,
                    orientation='h',
                    name=key,
                    showlegend=False,
                    marker=dict(color='lightblue')
                ),
                row=1, col=idx
            )
        
        # 更新布局
        fig.update_layout(
            title="Feature Importance Comparison Across Models",
            height=500,
            showlegend=False
        )
        
        # 更新x轴标签
        for i in range(1, n_models + 1):
            fig.update_xaxes(title_text="Importance", row=1, col=i)
            if i == 1:
                fig.update_yaxes(title_text="Feature", row=1, col=i)
        
        # 保存
        fig.write_html(str(output_file))
        
        return output_file
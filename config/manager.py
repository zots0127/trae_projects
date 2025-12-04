#!/usr/bin/env python3
"""
动态配置管理器 - 从YAML文件加载配置
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
import copy
from datetime import datetime
import glob

# 导入配置数据类
from config.system import (
    DataConfig, FeatureConfig, ModelConfig, 
    TrainingConfig, LoggingConfig,
    ExperimentConfig
)


class DynamicConfigManager:
    """动态配置管理器 - 从文件系统加载配置"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件根目录
        """
        self.config_dir = Path(config_dir)
        self.templates = {}
        self.config_cache = {}
        
        # 自动扫描并加载所有配置
        self.scan_and_load_configs()
    
    def scan_and_load_configs(self):
        """扫描配置目录并加载所有YAML配置"""
        if not self.config_dir.exists():
            print(f"⚠️ 配置目录不存在: {self.config_dir}")
            return
        
        # 查找所有YAML文件
        yaml_files = list(self.config_dir.glob("**/*.yaml")) + list(self.config_dir.glob("**/*.yml"))
        
        loaded_count = 0
        for yaml_file in yaml_files:
            try:
                # 生成配置键名（相对路径，去掉扩展名）
                relative_path = yaml_file.relative_to(self.config_dir)
                config_key = str(relative_path.with_suffix(''))
                
                # 也生成简短名称（仅文件名）
                short_key = yaml_file.stem
                
                # 加载配置
                config = self.load_config_file(yaml_file)
                
                # 存储配置（使用两个键名）
                self.templates[config_key] = config
                
                # 如果短名称不冲突，也使用短名称
                if short_key not in self.templates:
                    self.templates[short_key] = config
                
                loaded_count += 1
                
            except Exception as e:
                print(f"⚠️ 加载配置失败 {yaml_file}: {e}")
        
        print(f"✅ 成功加载 {loaded_count} 个配置模板")
    
    def load_config_file(self, file_path: Path) -> ExperimentConfig:
        """加载单个配置文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 转换为配置对象
        return self._dict_to_config(data, config_path=str(file_path))
    
    def _dict_to_config(self, data: Dict, config_path: Optional[str] = None) -> ExperimentConfig:
        """将字典转换为配置对象"""
        # 处理嵌套的配置对象
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        else:
            data['data'] = DataConfig()
        
        if 'feature' in data and isinstance(data['feature'], dict):
            data['feature'] = FeatureConfig(**data['feature'])
        else:
            data['feature'] = FeatureConfig()
        
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        else:
            data['model'] = ModelConfig()
        
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        else:
            data['training'] = TrainingConfig()
        
        # Optimization has been removed from the codebase
        if 'optimization' in data:
            # Skip optimization config as it's no longer supported
            data.pop('optimization', None)
        
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig(**data['logging'])
        else:
            data['logging'] = LoggingConfig()
        
        # 处理AutoML特殊配置
        if 'models' in data:
            # AutoML配置，包含多个模型
            data['automl_models'] = data.pop('models')
        
        if 'model_configs' in data:
            data['automl_model_configs'] = data.pop('model_configs')
        
        config = ExperimentConfig(**data)
        config.config_path = config_path
        return config
    
    def get_config(self, name: str) -> Optional[ExperimentConfig]:
        """
        获取配置
        
        Args:
            name: 配置名称或路径
        
        Returns:
            配置对象，如果不存在返回None
        """
        # 首先检查缓存的模板
        if name in self.templates:
            return self.templates[name].copy()
        
        # 尝试作为文件路径加载
        config_path = Path(name)
        if config_path.exists() and config_path.suffix in ['.yaml', '.yml']:
            try:
                return self.load_config_file(config_path)
            except Exception as e:
                print(f"⚠️ 加载配置文件失败 {config_path}: {e}")
                return None
        
        # 尝试在配置目录中查找
        possible_paths = [
            self.config_dir / f"{name}.yaml",
            self.config_dir / f"{name}.yml",
            self.config_dir / f"**/{name}.yaml",
            self.config_dir / f"**/{name}.yml"
        ]
        
        for pattern in possible_paths:
            if '*' in str(pattern):
                matches = list(self.config_dir.glob(str(pattern.relative_to(self.config_dir))))
                if matches:
                    try:
                        return self.load_config_file(matches[0])
                    except Exception as e:
                        print(f"⚠️ 加载配置文件失败 {matches[0]}: {e}")
            elif pattern.exists():
                try:
                    return self.load_config_file(pattern)
                except Exception as e:
                    print(f"⚠️ 加载配置文件失败 {pattern}: {e}")
        
        return None
    
    def list_configs(self) -> List[str]:
        """列出所有可用配置"""
        return sorted(list(self.templates.keys()))
    
    def get_config_info(self, name: str) -> Optional[Dict]:
        """获取配置信息"""
        config = self.get_config(name)
        if config:
            return {
                'name': config.name,
                'description': config.description,
                'model': config.model.model_type,
                'feature': config.feature.feature_type,
                'n_folds': config.training.n_folds,
                # 'optimization': removed from codebase
            }
        return None
    
    def save_config(self, config: ExperimentConfig, path: str):
        """保存配置到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            config.to_json(str(path))
        else:
            config.to_yaml(str(path))
    
    def create_config_from_params(self, **params) -> ExperimentConfig:
        """从参数创建配置"""
        # 如果指定了基础配置，先加载它
        base_config = None
        if 'config' in params:
            base_config = self.get_config(params.pop('config'))
        
        if base_config:
            config = base_config.copy()
        else:
            config = ExperimentConfig()
        
        # 应用参数更新
        config = self._apply_params_to_config(config, params)
        
        return config
    
    def _apply_params_to_config(self, config: ExperimentConfig, params: Dict) -> ExperimentConfig:
        """应用参数到配置"""
        for key, value in params.items():
            if '.' in key:
                # 处理嵌套参数，如 model.hyperparameters.n_estimators
                parts = key.split('.')
                obj = config
                
                # 导航到目标属性的父对象
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        # 如果属性不存在，创建一个字典
                        setattr(obj, part, {})
                        obj = getattr(obj, part)
                
                # 设置最终值
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
                elif isinstance(obj, dict):
                    obj[parts[-1]] = value
            else:
                # 处理顶层参数
                if hasattr(config, key):
                    setattr(config, key, value)
                # 特殊处理一些常用参数
                elif key == 'model' and hasattr(config.model, 'model_type'):
                    config.model.model_type = value
                elif key == 'n_folds' and hasattr(config.training, 'n_folds'):
                    config.training.n_folds = value
                elif key == 'feature' and hasattr(config.feature, 'feature_type'):
                    config.feature.feature_type = value
                elif key == 'multi_target' and hasattr(config.data, 'multi_target_strategy'):
                    config.data.multi_target_strategy = value
                elif key == 'nan_handling' and hasattr(config.data, 'nan_handling'):
                    config.data.nan_handling = value
                # Optimization parameters removed from codebase
                # elif key == 'optimization': removed
                # elif key == 'n_trials': removed
        
        return config
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("📚 可用配置模板")
        print("="*60)
        
        # 按模型类型分组
        configs_by_model = {}
        for name, config in self.templates.items():
            if '/' not in name:  # 只显示短名称
                model_type = config.model.model_type
                if model_type not in configs_by_model:
                    configs_by_model[model_type] = []
                configs_by_model[model_type].append((name, config))
        
        for model_type in sorted(configs_by_model.keys()):
            print(f"\n📦 {model_type.upper()}")
            print("-" * 40)
            
            for name, config in sorted(configs_by_model[model_type]):
                desc = config.description[:40] + "..." if len(config.description) > 40 else config.description
                print(f"  • {name:<20} {desc}")
        
        print("\n" + "="*60)
        print(f"总计: {len(self.templates)} 个配置模板")
        print("使用方法: python automl.py train config=<模板名>")
        print("="*60 + "\n")


# 创建全局配置管理器实例
config_manager = DynamicConfigManager()


def get_config(name_or_params: Union[str, Dict]) -> ExperimentConfig:
    """
    获取配置的便捷函数
    
    Args:
        name_or_params: 配置名称或参数字典
    
    Returns:
        配置对象
    """
    if isinstance(name_or_params, str):
        return config_manager.get_config(name_or_params)
    else:
        return config_manager.create_config_from_params(**name_or_params)


def list_configs() -> List[str]:
    """列出所有可用配置"""
    return config_manager.list_configs()


def save_config(config: ExperimentConfig, path: str):
    """保存配置"""
    config_manager.save_config(config, path)
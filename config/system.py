#!/usr/bin/env python3
"""
YOLO风格的配置系统
通过配置文件定义完整的训练流程
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
import copy
from datetime import datetime


# ========================================
#           配置数据类
# ========================================

@dataclass
class DataConfig:
    """数据配置"""
    data_path: str = "data/Database_normalized.csv"
    smiles_columns: List[str] = field(default_factory=lambda: ['L1', 'L2', 'L3'])
    target_columns: List[str] = field(default_factory=lambda: ['Max_wavelength(nm)', 'PLQY', 'tau(s*10^-6)'])
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    random_seed: int = 42
    # 可选：外部提供的测试集CSV路径。若提供，则在完整训练后进行一次测试评估
    test_data_path: Optional[str] = None
    
    # 缺失值处理策略
    # 可选: 'skip' (跳过含NaN的行), 'mean' (均值填充), 'median' (中位数填充), 
    #      'zero' (零值填充), 'forward' (前向填充), 'interpolate' (插值)
    nan_handling: str = "skip"
    
    # 缺失值处理的详细配置
    nan_threshold: float = 0.5  # 当某行缺失值比例超过此阈值时跳过
    feature_nan_strategy: str = "zero"  # 特征缺失值处理（当nan_handling不是skip时）
    target_nan_strategy: str = "skip"   # 目标值缺失处理
    
    # 多目标数据选择策略
    # 'intersection': 只使用所有目标都有值的数据（最严格，数据最少）
    # 'independent': 每个目标独立使用其有效数据（默认，数据利用率高）
    # 'union': 使用所有数据，缺失值填充（最宽松，需配合nan_handling）
    multi_target_strategy: str = "independent"
    
    # 数据采样（用于调试）
    sample_size: Optional[int] = None  # 如果设置，只使用前N个样本
    
    def validate(self):
        """验证配置"""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, "数据分割比例之和必须为1"
        assert len(self.smiles_columns) > 0, "至少需要一个SMILES列"
        assert len(self.target_columns) > 0, "至少需要一个目标列"
        assert self.nan_handling in ["skip", "mean", "median", "zero", "forward", "interpolate"], \
            f"不支持的缺失值处理方法: {self.nan_handling}"
        assert self.multi_target_strategy in ["intersection", "independent", "union"], \
            f"不支持的多目标策略: {self.multi_target_strategy}"


@dataclass
class FeatureConfig:
    """特征配置"""
    feature_type: str = "combined"  # morgan, descriptors, combined
    morgan_bits: int = 1024
    morgan_radius: int = 2
    combination_method: str = "mean"  # mean, sum, concat
    use_cache: bool = True
    cache_dir: str = "feature_cache"
    descriptor_count: int = 85
    
    def validate(self):
        """验证配置"""
        assert self.feature_type in ["morgan", "descriptors", "combined", "tabular", "auto"], \
            f"不支持的特征类型: {self.feature_type}"
        assert self.combination_method in ["mean", "sum", "concat"], \
            f"不支持的组合方法: {self.combination_method}"
        assert isinstance(self.descriptor_count, int) and self.descriptor_count > 0, \
            f"descriptor_count 必须是正整数: {self.descriptor_count}"


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str = "xgboost"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        # 设置默认超参数
        if not self.hyperparameters:
            self.hyperparameters = self.get_default_params()
    
    def get_default_params(self) -> Dict:
        """获取默认参数"""
        from models import MODEL_PARAMS
        return MODEL_PARAMS.get(self.model_type, {}).copy()
    
    def validate(self):
        """验证配置"""
        from models import ModelFactory
        assert self.model_type in ModelFactory.get_supported_models(), \
            f"不支持的模型类型: {self.model_type}"


@dataclass
class TrainingConfig:
    """训练配置"""
    n_folds: int = 10
    metrics: List[str] = field(default_factory=lambda: ["rmse", "mae", "r2", "mape"])
    early_stopping: bool = False
    early_stopping_rounds: int = 10
    verbose: int = 1
    save_fold_models: bool = True
    save_final_model: bool = True
    save_training_curves: bool = True  # 保存训练曲线（默认开启）
    save_feature_importance: bool = True  # 保存特征重要性（默认开启）
    model_selection: Optional[str] = None  # 模型选择策略（用于AutoML）: best_r2, best_rmse等
    
    def validate(self):
        """验证配置"""
        assert self.n_folds > 1, "交叉验证折数必须大于1"
        valid_metrics = ["rmse", "mae", "r2", "mape", "mse"]
        for metric in self.metrics:
            assert metric in valid_metrics, f"不支持的指标: {metric}"


@dataclass
class ComparisonConfig:
    """模型对比配置"""
    enable: bool = False  # 是否启用对比表生成
    formats: List[str] = field(default_factory=lambda: ["markdown", "html", "latex", "csv"])
    highlight_best: bool = True  # 高亮最佳模型
    include_std: bool = True  # 包含标准差
    save_to_file: bool = True  # 保存到文件
    output_dir: Optional[str] = None  # 输出目录（None表示使用训练目录）
    
    # 数值精度配置
    decimal_places: Dict[str, int] = field(default_factory=lambda: {
        'r2': 4,
        'rmse': 4,
        'mae': 4
    })
    
    def validate(self):
        """验证配置"""
        valid_formats = ["markdown", "html", "latex", "csv", "excel"]
        for fmt in self.formats:
            assert fmt in valid_formats, f"不支持的表格格式: {fmt}"


@dataclass
class ExportConfig:
    """导出配置"""
    enable: bool = True
    formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    include_predictions: bool = True
    include_feature_importance: bool = True
    include_cv_details: bool = True
    generate_plots: bool = True
    generate_report: bool = True
    stratified_analysis: bool = False  # 生成分段性能分析图（如PLQY范围混淆矩阵）
    
    def validate(self):
        """验证配置"""
        valid_formats = ["json", "csv", "excel", "pickle"]
        for fmt in self.formats:
            assert fmt in valid_formats, f"不支持的导出格式: {fmt}"


@dataclass
class LoggingConfig:
    """日志配置"""
    project_name: str = "ml_experiment"
    base_dir: str = "training_logs"
    auto_save: bool = True
    save_plots: bool = True
    generate_report: bool = True
    export_for_paper: bool = False
    log_level: str = "INFO"
    
    def validate(self):
        """验证配置"""
        assert self.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"], \
            f"不支持的日志级别: {self.log_level}"


@dataclass
class ExperimentConfig:
    """实验配置 - 主配置类"""
    name: str = "default_experiment"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    
    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # AutoML特殊配置（用于automl模板）
    automl_models: Optional[List[str]] = None  # AutoML要测试的模型列表（已废弃，使用models_to_train）
    automl_model_configs: Optional[Dict[str, Dict]] = None  # 每个模型的配置
    models_to_train: Optional[List[str]] = None  # 多模型训练时的模型列表
    
    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config_path: Optional[str] = None
    
    def validate(self):
        """验证所有配置"""
        self.data.validate()
        self.feature.validate()
        self.model.validate()
        self.training.validate()
        
        # 处理深拷贝后可能变成dict的情况
        if isinstance(self.comparison, dict):
            self.comparison = ComparisonConfig(**self.comparison)
        self.comparison.validate()
        
        if isinstance(self.export, dict):
            self.export = ExportConfig(**self.export)
        self.export.validate()
        
        self.logging.validate()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def to_yaml(self, path: Optional[str] = None) -> str:
        """转换为YAML格式"""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            with open(path, 'w') as f:
                f.write(yaml_str)
        return yaml_str
    
    def to_json(self, path: Optional[str] = None) -> str:
        """转换为JSON格式"""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """从YAML文件加载配置"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, config_path=path)
    
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """从JSON文件加载配置"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, config_path=path)
    
    @classmethod
    def from_dict(cls, data: Dict, config_path: Optional[str] = None) -> 'ExperimentConfig':
        """从字典创建配置"""
        # 创建子配置
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        if 'feature' in data and isinstance(data['feature'], dict):
            data['feature'] = FeatureConfig(**data['feature'])
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        if 'comparison' in data and isinstance(data['comparison'], dict):
            data['comparison'] = ComparisonConfig(**data['comparison'])
        if 'export' in data and isinstance(data['export'], dict):
            data['export'] = ExportConfig(**data['export'])
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig(**data['logging'])
        
        config = cls(**data)
        config.config_path = config_path
        return config
    
    def copy(self) -> 'ExperimentConfig':
        """深拷贝配置"""
        return copy.deepcopy(self)
    
    def update(self, updates: Dict) -> 'ExperimentConfig':
        """更新配置"""
        new_config = self.copy()
        
        for key, value in updates.items():
            if '.' in key:  # 支持嵌套更新，如 "model.hyperparameters.n_estimators"
                parts = key.split('.')
                obj = new_config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(new_config, key, value)
        
        return new_config


# ========================================
#           配置管理器
# ========================================

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 预定义配置模板
        self.templates = {}
        self.load_templates()
    
    def load_templates(self):
        """加载预定义模板"""
        # XGBoost快速模板
        self.templates['xgboost_quick'] = ExperimentConfig(
            name="xgboost_quick",
            description="XGBoost快速训练模板",
            model=ModelConfig(
                model_type="xgboost",
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            ),
            training=TrainingConfig(n_folds=5)
        )
        
        # XGBoost标准模板
        self.templates['xgboost_standard'] = ExperimentConfig(
            name="xgboost_standard",
            description="XGBoost标准训练模板",
            model=ModelConfig(
                model_type="xgboost",
                hyperparameters={
                    'n_estimators': 300,
                    'max_depth': 7,
                    'learning_rate': 0.07,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            ),
            training=TrainingConfig(n_folds=10)
        )
        
        # XGBoost完整模板
        self.templates['xgboost_full'] = ExperimentConfig(
            name="xgboost_full",
            description="XGBoost完整训练模板",
            model=ModelConfig(
                model_type="xgboost",
                hyperparameters={
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            ),
            training=TrainingConfig(
                n_folds=10,
                early_stopping=True,
                early_stopping_rounds=50
            ),
            logging=LoggingConfig(
                save_plots=True,
                generate_report=True,
                export_for_paper=True
            )
        )
        
        # LightGBM模板
        self.templates['lightgbm'] = ExperimentConfig(
            name="lightgbm",
            description="LightGBM训练模板",
            model=ModelConfig(
                model_type="lightgbm",
                hyperparameters={
                    'n_estimators': 200,
                    'num_leaves': 31,
                    'learning_rate': 0.1
                }
            )
        )
        
        # 集成学习模板
        self.templates['ensemble'] = ExperimentConfig(
            name="ensemble",
            description="集成学习模板（多模型）",
            model=ModelConfig(model_type="random_forest"),
            training=TrainingConfig(n_folds=10)
        )
        
        # 调试模板
        self.templates['debug'] = ExperimentConfig(
            name="debug",
            description="调试模板（小数据集，快速训练）",
            model=ModelConfig(
                model_type="xgboost",
                hyperparameters={'n_estimators': 10, 'max_depth': 3}
            ),
            training=TrainingConfig(n_folds=2),
            logging=LoggingConfig(save_plots=False, generate_report=False)
        )
        

        # LightGBM - 快速与完整模板
        self.templates['lightgbm_quick'] = ExperimentConfig(
            name="lightgbm_quick",
            description="LightGBM快速训练模板",
            model=ModelConfig(
                model_type="lightgbm",
                hyperparameters={
                    'n_estimators': 100,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5
                }
            ),
            training=TrainingConfig(n_folds=5)
        )

        self.templates['lightgbm_full'] = ExperimentConfig(
            name="lightgbm_full",
            description="LightGBM完整训练模板",
            model=ModelConfig(
                model_type="lightgbm",
                hyperparameters={
                    'n_estimators': 300,
                    'num_leaves': 63,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # LightGBM - 标准模板
        self.templates['lightgbm_standard'] = ExperimentConfig(
            name="lightgbm_standard",
            description="LightGBM标准训练模板",
            model=ModelConfig(
                model_type="lightgbm",
                hyperparameters={
                    'n_estimators': 200,
                    'num_leaves': 47,
                    'learning_rate': 0.07,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # LightGBM - 大型模板
        self.templates['lightgbm_large'] = ExperimentConfig(
            name="lightgbm_large",
            description="LightGBM大型训练模板",
            model=ModelConfig(
                model_type="lightgbm",
                hyperparameters={
                    'n_estimators': 500,
                    'num_leaves': 95,
                    'learning_rate': 0.03,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # CatBoost - 快速模板
        self.templates['catboost_quick'] = ExperimentConfig(
            name="catboost_quick",
            description="CatBoost快速训练模板",
            model=ModelConfig(
                model_type="catboost",
                hyperparameters={
                    'iterations': 200,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'verbose': False
                }
            ),
            training=TrainingConfig(n_folds=5)
        )

        # CatBoost - 标准模板
        self.templates['catboost_standard'] = ExperimentConfig(
            name="catboost_standard",
            description="CatBoost标准训练模板",
            model=ModelConfig(
                model_type="catboost",
                hyperparameters={
                    'iterations': 500,
                    'depth': 8,
                    'learning_rate': 0.07,
                    'verbose': False
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # CatBoost - 大型模板
        self.templates['catboost_large'] = ExperimentConfig(
            name="catboost_large",
            description="CatBoost大型训练模板",
            model=ModelConfig(
                model_type="catboost",
                hyperparameters={
                    'iterations': 1000,
                    'depth': 10,
                    'learning_rate': 0.03,
                    'verbose': False
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 随机森林 - 快速模板
        self.templates['random_forest_fast'] = ExperimentConfig(
            name="random_forest_fast",
            description="随机森林快速训练模板",
            model=ModelConfig(
                model_type="random_forest",
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 随机森林 - 标准模板
        self.templates['random_forest_standard'] = ExperimentConfig(
            name="random_forest_standard",
            description="随机森林标准训练模板",
            model=ModelConfig(
                model_type="random_forest",
                hyperparameters={
                    'n_estimators': 300,
                    'max_depth': 20,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 随机森林 - 大型模板
        self.templates['random_forest_large'] = ExperimentConfig(
            name="random_forest_large",
            description="随机森林大型训练模板",
            model=ModelConfig(
                model_type="random_forest",
                hyperparameters={
                    'n_estimators': 500,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性
        self.templates['random_forest'] = self.templates['random_forest_standard']

        # 梯度提升树 - 快速模板
        self.templates['gradient_boosting_fast'] = ExperimentConfig(
            name="gradient_boosting_fast",
            description="Gradient Boosting快速训练模板",
            model=ModelConfig(
                model_type="gradient_boosting",
                hyperparameters={
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 梯度提升树 - 标准模板
        self.templates['gradient_boosting_standard'] = ExperimentConfig(
            name="gradient_boosting_standard",
            description="Gradient Boosting标准训练模板",
            model=ModelConfig(
                model_type="gradient_boosting",
                hyperparameters={
                    'n_estimators': 300,
                    'learning_rate': 0.07,
                    'max_depth': 5
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 梯度提升树 - 大型模板
        self.templates['gradient_boosting_large'] = ExperimentConfig(
            name="gradient_boosting_large",
            description="Gradient Boosting大型训练模板",
            model=ModelConfig(
                model_type="gradient_boosting",
                hyperparameters={
                    'n_estimators': 500,
                    'learning_rate': 0.03,
                    'max_depth': 7
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性
        self.templates['gradient_boosting'] = self.templates['gradient_boosting_standard']

        # AdaBoost - 快速模板
        self.templates['adaboost_fast'] = ExperimentConfig(
            name="adaboost_fast",
            description="AdaBoost快速训练模板",
            model=ModelConfig(
                model_type="adaboost",
                hyperparameters={
                    'n_estimators': 50,
                    'learning_rate': 1.0
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # AdaBoost - 标准模板
        self.templates['adaboost_standard'] = ExperimentConfig(
            name="adaboost_standard",
            description="AdaBoost标准训练模板",
            model=ModelConfig(
                model_type="adaboost",
                hyperparameters={
                    'n_estimators': 200,
                    'learning_rate': 0.5
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # AdaBoost - 大型模板
        self.templates['adaboost_large'] = ExperimentConfig(
            name="adaboost_large",
            description="AdaBoost大型训练模板",
            model=ModelConfig(
                model_type="adaboost",
                hyperparameters={
                    'n_estimators': 500,
                    'learning_rate': 0.1
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性
        self.templates['adaboost'] = self.templates['adaboost_standard']

        # Extra Trees - 快速模板
        self.templates['extra_trees_fast'] = ExperimentConfig(
            name="extra_trees_fast",
            description="Extra Trees快速训练模板",
            model=ModelConfig(
                model_type="extra_trees",
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # Extra Trees - 标准模板
        self.templates['extra_trees_standard'] = ExperimentConfig(
            name="extra_trees_standard",
            description="Extra Trees标准训练模板",
            model=ModelConfig(
                model_type="extra_trees",
                hyperparameters={
                    'n_estimators': 300,
                    'max_depth': 20,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # Extra Trees - 大型模板
        self.templates['extra_trees_large'] = ExperimentConfig(
            name="extra_trees_large",
            description="Extra Trees大型训练模板",
            model=ModelConfig(
                model_type="extra_trees",
                hyperparameters={
                    'n_estimators': 500,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性
        self.templates['extra_trees'] = self.templates['extra_trees_standard']

        # SVR - 快速模板
        self.templates['svr_fast'] = ExperimentConfig(
            name="svr_fast",
            description="SVR快速训练模板",
            model=ModelConfig(
                model_type="svr",
                hyperparameters={
                    'kernel': 'rbf',
                    'C': 1.0,
                    'epsilon': 0.1
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # SVR - 标准模板
        self.templates['svr_standard'] = ExperimentConfig(
            name="svr_standard",
            description="SVR标准训练模板",
            model=ModelConfig(
                model_type="svr",
                hyperparameters={
                    'kernel': 'rbf',
                    'C': 10.0,
                    'epsilon': 0.1
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # SVR - 大型模板
        self.templates['svr_large'] = ExperimentConfig(
            name="svr_large",
            description="SVR大型训练模板",
            model=ModelConfig(
                model_type="svr",
                hyperparameters={
                    'kernel': 'rbf',
                    'C': 100.0,
                    'epsilon': 0.01
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性 (重命名为svr)
        self.templates['svr'] = self.templates['svr_standard']
        self.templates['svr_rbf'] = self.templates['svr_standard']

        # KNN - 快速模板
        self.templates['knn_fast'] = ExperimentConfig(
            name="knn_fast",
            description="KNN快速训练模板",
            model=ModelConfig(
                model_type="knn",
                hyperparameters={
                    'n_neighbors': 5,
                    'weights': 'uniform'
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # KNN - 标准模板
        self.templates['knn_standard'] = ExperimentConfig(
            name="knn_standard",
            description="KNN标准训练模板",
            model=ModelConfig(
                model_type="knn",
                hyperparameters={
                    'n_neighbors': 7,
                    'weights': 'distance'
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # KNN - 大型模板
        self.templates['knn_large'] = ExperimentConfig(
            name="knn_large",
            description="KNN大型训练模板",
            model=ModelConfig(
                model_type="knn",
                hyperparameters={
                    'n_neighbors': 10,
                    'weights': 'distance'
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性
        self.templates['knn'] = self.templates['knn_standard']

        # 决策树 - 快速模板
        self.templates['decision_tree_fast'] = ExperimentConfig(
            name="decision_tree_fast",
            description="决策树快速训练模板",
            model=ModelConfig(
                model_type="decision_tree",
                hyperparameters={
                    'max_depth': 5,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 决策树 - 标准模板
        self.templates['decision_tree_standard'] = ExperimentConfig(
            name="decision_tree_standard",
            description="决策树标准训练模板",
            model=ModelConfig(
                model_type="decision_tree",
                hyperparameters={
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 决策树 - 大型模板
        self.templates['decision_tree_large'] = ExperimentConfig(
            name="decision_tree_large",
            description="决策树大型训练模板",
            model=ModelConfig(
                model_type="decision_tree",
                hyperparameters={
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性
        self.templates['decision_tree'] = self.templates['decision_tree_standard']

        # Ridge - 快速模板
        self.templates['ridge_fast'] = ExperimentConfig(
            name="ridge_fast",
            description="Ridge快速训练模板",
            model=ModelConfig(
                model_type="ridge",
                hyperparameters={'alpha': 1.0}
            ),
            training=TrainingConfig(n_folds=10)
        )

        # Ridge - 标准模板
        self.templates['ridge_standard'] = ExperimentConfig(
            name="ridge_standard",
            description="Ridge标准训练模板",
            model=ModelConfig(
                model_type="ridge",
                hyperparameters={'alpha': 0.5}
            ),
            training=TrainingConfig(n_folds=10)
        )

        # Ridge - 大型模板
        self.templates['ridge_large'] = ExperimentConfig(
            name="ridge_large",
            description="Ridge大型训练模板",
            model=ModelConfig(
                model_type="ridge",
                hyperparameters={'alpha': 0.1}
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性
        self.templates['ridge'] = self.templates['ridge_standard']

        # Lasso - 快速模板
        self.templates['lasso_fast'] = ExperimentConfig(
            name="lasso_fast",
            description="Lasso快速训练模板",
            model=ModelConfig(
                model_type="lasso",
                hyperparameters={'alpha': 0.1}
            ),
            training=TrainingConfig(n_folds=10)
        )

        # Lasso - 标准模板
        self.templates['lasso_standard'] = ExperimentConfig(
            name="lasso_standard",
            description="Lasso标准训练模板",
            model=ModelConfig(
                model_type="lasso",
                hyperparameters={'alpha': 0.05}
            ),
            training=TrainingConfig(n_folds=10)
        )

        # Lasso - 大型模板
        self.templates['lasso_large'] = ExperimentConfig(
            name="lasso_large",
            description="Lasso大型训练模板",
            model=ModelConfig(
                model_type="lasso",
                hyperparameters={'alpha': 0.01}
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性
        self.templates['lasso'] = self.templates['lasso_standard']

        # ElasticNet - 快速模板
        self.templates['elasticnet_fast'] = ExperimentConfig(
            name="elasticnet_fast",
            description="ElasticNet快速训练模板",
            model=ModelConfig(
                model_type="elastic_net",
                hyperparameters={
                    'alpha': 0.5,
                    'l1_ratio': 0.5
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # ElasticNet - 标准模板
        self.templates['elasticnet_standard'] = ExperimentConfig(
            name="elasticnet_standard",
            description="ElasticNet标准训练模板",
            model=ModelConfig(
                model_type="elastic_net",
                hyperparameters={
                    'alpha': 0.1,
                    'l1_ratio': 0.7
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # ElasticNet - 大型模板
        self.templates['elasticnet_large'] = ExperimentConfig(
            name="elasticnet_large",
            description="ElasticNet大型训练模板",
            model=ModelConfig(
                model_type="elastic_net",
                hyperparameters={
                    'alpha': 0.01,
                    'l1_ratio': 0.5
                }
            ),
            training=TrainingConfig(n_folds=10)
        )

        # 保留原始模板以保持兼容性
        self.templates['elastic_net'] = self.templates['elasticnet_standard']
    
    def get_template(self, template_name: str) -> ExperimentConfig:
        """
        获取模板配置
        
        Args:
            template_name: 模板名称
        
        Returns:
            配置对象
        """
        if template_name not in self.templates:
            raise ValueError(f"模板不存在: {template_name}. 可用模板: {list(self.templates.keys())}")
        return self.templates[template_name].copy()
    
    def list_templates(self) -> List[str]:
        """列出所有可用模板"""
        return list(self.templates.keys())
    
    def save_config(self, config: ExperimentConfig, filename: str, format: str = "yaml"):
        """
        保存配置文件
        
        Args:
            config: 配置对象
            filename: 文件名（不含扩展名）
            format: 格式 (yaml/json)
        """
        if format == "yaml":
            path = self.config_dir / f"{filename}.yaml"
            config.to_yaml(str(path))
        elif format == "json":
            path = self.config_dir / f"{filename}.json"
            config.to_json(str(path))
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"配置已保存: {path}")
        return path
    
    def load_config(self, filename: str) -> ExperimentConfig:
        """
        加载配置文件
        
        Args:
            filename: 文件名或路径
        
        Returns:
            配置对象
        """
        # 尝试不同的路径和格式
        paths_to_try = [
            Path(filename),
            self.config_dir / filename,
            self.config_dir / f"{filename}.yaml",
            self.config_dir / f"{filename}.json"
        ]
        
        for path in paths_to_try:
            if path.exists():
                if path.suffix == '.yaml' or path.suffix == '.yml':
                    return ExperimentConfig.from_yaml(str(path))
                elif path.suffix == '.json':
                    return ExperimentConfig.from_json(str(path))
        
        raise FileNotFoundError(f"配置文件不存在: {filename}")
    
    def create_from_wizard(self) -> ExperimentConfig:
        """通过向导创建配置"""
        print("\n🔧 配置向导")
        print("=" * 50)
        
        # 选择模板
        print("\n可用模板:")
        for i, template in enumerate(self.templates.keys(), 1):
            desc = self.templates[template].description
            print(f"  {i}. {template}: {desc}")
        
        choice = input("\n选择模板 (输入编号或名称，直接回车使用默认): ").strip()
        
        if choice.isdigit():
            template_name = list(self.templates.keys())[int(choice) - 1]
        elif choice in self.templates:
            template_name = choice
        else:
            template_name = 'xgboost_quick'
        
        config = self.get_template(template_name)
        
        # 自定义配置
        name = input(f"实验名称 [{config.name}]: ").strip() or config.name
        config.name = name
        
        description = input(f"实验描述 [{config.description}]: ").strip() or config.description
        config.description = description
        
        # 模型参数
        n_folds = input(f"交叉验证折数 [{config.training.n_folds}]: ").strip()
        if n_folds.isdigit():
            config.training.n_folds = int(n_folds)
        
        # 特征类型
        feature_type = input(f"特征类型 (morgan/descriptors/combined) [{config.feature.feature_type}]: ").strip()
        if feature_type in ["morgan", "descriptors", "combined"]:
            config.feature.feature_type = feature_type
        
        print("\n✅ 配置创建完成!")
        return config


# ========================================
#           批量实验配置
# ========================================

@dataclass
class BatchExperimentConfig:
    """批量实验配置"""
    base_config: ExperimentConfig
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_experiment(self, name: str, updates: Dict):
        """
        添加实验
        
        Args:
            name: 实验名称
            updates: 配置更新
        """
        self.experiments.append({
            'name': name,
            'updates': updates
        })
    
    def generate_configs(self) -> List[ExperimentConfig]:
        """生成所有实验配置"""
        configs = []
        for exp in self.experiments:
            config = self.base_config.copy()
            config.name = exp['name']
            config = config.update(exp['updates'])
            configs.append(config)
        return configs
    
    @classmethod
    def create_grid_search(cls, 
                          base_config: ExperimentConfig,
                          param_grid: Dict[str, List]) -> 'BatchExperimentConfig':
        """
        创建网格搜索配置
        
        Args:
            base_config: 基础配置
            param_grid: 参数网格
        
        Returns:
            批量实验配置
        """
        batch = cls(base_config=base_config)
        
        # 生成所有参数组合
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        for i, combination in enumerate(itertools.product(*values)):
            updates = dict(zip(keys, combination))
            name = f"{base_config.name}_grid_{i+1}"
            batch.add_experiment(name, updates)
        
        return batch


# ========================================
#           配置验证器
# ========================================

class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_file_exists(config: ExperimentConfig) -> bool:
        """验证数据文件是否存在"""
        data_path = Path(config.data.data_path)
        if not data_path.exists():
            print(f"⚠️ 数据文件不存在: {data_path}")
            return False
        return True
    
    @staticmethod
    def validate_dependencies(config: ExperimentConfig) -> bool:
        """验证依赖是否安装"""
        base_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
        model_packages = {
            'xgboost': ['xgboost'],
            'lightgbm': ['lightgbm'],
            'catboost': ['catboost'],
            'random_forest': []
        }
        feature_requires_rdkit = getattr(config.feature, 'feature_type', None) in ['morgan', 'descriptors', 'combined']
        packages_to_check = list(base_packages)
        packages_to_check.extend(model_packages.get(config.model.model_type, []))
        if feature_requires_rdkit:
            packages_to_check.append('rdkit')
        missing = []
        for package in packages_to_check:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        if missing:
            print(f"⚠️ 缺少依赖包: {missing}")
            return False
        return True
    
    @staticmethod
    def validate_all(config: ExperimentConfig) -> bool:
        """执行所有验证"""
        try:
            # 配置内部验证
            config.validate()
            
            # 文件验证
            if not ConfigValidator.validate_file_exists(config):
                return False
            
            # 依赖验证
            if not ConfigValidator.validate_dependencies(config):
                return False
            
            print("✅ 配置验证通过")
            return True
            
        except Exception as e:
            print(f"❌ 配置验证失败: {e}")
            return False


# ========================================
#           便捷函数
# ========================================

def create_default_config(model_type: str = "xgboost") -> ExperimentConfig:
    """创建默认配置"""
    return ExperimentConfig(
        name=f"{model_type}_experiment",
        model=ModelConfig(model_type=model_type)
    )


def load_config(path: str) -> ExperimentConfig:
    """加载配置文件"""
    if path.endswith('.yaml') or path.endswith('.yml'):
        return ExperimentConfig.from_yaml(path)
    elif path.endswith('.json'):
        return ExperimentConfig.from_json(path)
    else:
        raise ValueError(f"不支持的配置文件格式: {path}")


def save_config(config: ExperimentConfig, path: str):
    """保存配置文件"""
    if path.endswith('.yaml') or path.endswith('.yml'):
        config.to_yaml(path)
    elif path.endswith('.json'):
        config.to_json(path)
    else:
        raise ValueError(f"不支持的配置文件格式: {path}")


if __name__ == "__main__":
    # 测试代码
    print("配置系统测试")
    print("=" * 50)
    
    # 创建配置管理器
    manager = ConfigManager()
    
    # 列出模板
    print("\n可用模板:")
    for template in manager.list_templates():
        print(f"  - {template}")
    
    # 获取模板
    config = manager.get_template('xgboost_full')
    
    # 保存配置
    manager.save_config(config, "test_config", "yaml")
    
    # 加载配置
    loaded_config = manager.load_config("test_config")
    
    # 验证配置
    ConfigValidator.validate_all(loaded_config)
    
    print("\n✅ 配置系统测试完成")

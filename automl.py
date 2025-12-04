#!/usr/bin/env python3
"""
AutoML - 自动化机器学习命令行接口
类似YOLO的简洁命令行工具

使用方式:
    automl train model=xgboost data=mydata.csv config=config.yaml
    automl predict model=saved_model.joblib data=test.csv
    automl validate config=config.yaml
    automl export model=xgboost target=wavelength format=onnx
"""

import sys
import os
from pathlib import Path
import argparse
import json
import yaml
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.system import ExperimentConfig, ConfigValidator
from config.manager import DynamicConfigManager, get_config, list_configs, save_config
from training.pipeline import TrainingPipeline
from models.base import load_model
from utils.run_manager import RunManager
from utils.analysis import ResultsAnalyzer
import joblib


# ========================================
#           命令解析器
# ========================================

class MLArgumentParser:
    """ML命令行参数解析器"""
    
    @staticmethod
    def parse_args_string(args_string: str) -> Dict[str, Any]:
        """
        解析 key=value 格式的参数字符串
        
        Args:
            args_string: 参数字符串，如 "model=xgboost data=file.csv"
        
        Returns:
            参数字典
        """
        params = {}
        
        # 分割参数
        parts = args_string.split()
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                
                # 移除外层引号（如果存在）
                if (value.startswith("'") and value.endswith("'")) or \
                   (value.startswith('"') and value.endswith('"')):
                    value = value[1:-1]
                
                # 尝试解析值的类型
                # 特殊参数：name和project应该始终是字符串
                if key in ['name', 'project']:
                    # 保持为字符串，不进行类型转换
                    pass
                # 布尔值
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                # 数字
                elif value.replace('.', '').replace('-', '').isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                # 列表
                elif value.startswith('[') and value.endswith(']'):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        # 尝试修复单引号的JSON
                        try:
                            fixed_value = value.replace("'", '"')
                            value = json.loads(fixed_value)
                        except:
                            # 如果还是失败，尝试作为逗号分隔的列表
                            inner = value[1:-1].strip()
                            if inner:
                                value = [v.strip().strip("'\"") for v in inner.split(',')]
                            else:
                                value = []
                # 字典
                elif value.startswith('{') and value.endswith('}'):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        # 尝试修复单引号的JSON
                        try:
                            fixed_value = value.replace("'", '"')
                            value = json.loads(fixed_value)
                        except:
                            pass  # 保持原值
                # 特殊处理models参数：支持逗号分隔格式
                elif key == 'models' and ',' in value:
                    value = [m.strip() for m in value.split(',')]
                
                params[key] = value
        
        return params
    
    @staticmethod
    def _parse_bool(value) -> bool:
        """解析布尔值"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['true', 'yes', '1', 'on']
        return bool(value)
    
    @staticmethod
    def merge_params_to_config(config: ExperimentConfig, params: Dict[str, Any]) -> ExperimentConfig:
        """
        将参数合并到配置中
        
        Args:
            config: 基础配置
            params: 要合并的参数
        
        Returns:
            更新后的配置
        """
        for key, value in params.items():
            # 处理嵌套键
            if '.' in key:
                parts = key.split('.')
                obj = config
                
                # 导航到嵌套对象
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        print(f"⚠️ 未知配置项: {key}")
                        continue
                
                # 设置值
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
            else:
                # 特殊处理一些常用参数
                if key == 'model':
                    config.model.model_type = value
                elif key == 'data':
                    config.data.data_path = value
                elif key == 'feature':
                    config.feature.feature_type = value
                elif key == 'folds':
                    config.training.n_folds = int(value) if isinstance(value, str) else value
                elif key == 'project':
                    config.logging.project_name = value
                elif key == 'target':
                    if isinstance(value, str):
                        config.data.target_columns = [value]
                    else:
                        config.data.target_columns = value
                elif key == 'save_curves':
                    # 处理保存训练曲线参数
                    config.training.save_training_curves = MLArgumentParser._parse_bool(value)
                elif key == 'save_importance':
                    # 处理保存特征重要性参数
                    config.training.save_feature_importance = MLArgumentParser._parse_bool(value)
                elif key in ['test_data', 'test_data_path']:
                    # 训练完成后对外部测试集进行评估
                    config.data.test_data_path = value
                    print(f"   ✅ 设置测试数据集: {value}")
                elif key in ['nan_handling', 'nan', 'missing']:
                    # 缺失值处理策略
                    config.data.nan_handling = value
                    print(f"   ✅ 设置缺失值处理: {value}")
                elif key in ['multi_target', 'multi_target_strategy', 'target_strategy']:
                    # 多目标数据选择策略
                    config.data.multi_target_strategy = value
                    print(f"   ✅ 设置多目标策略: {value}")
                elif hasattr(config, key):
                    setattr(config, key, value)
        
        return config


# ========================================
#           训练命令
# ========================================

def train_command(args: List[str]):
    """训练命令"""
    print("\n" + "="*60)
    print("AutoML Training System")
    print("="*60)
    from time import perf_counter as _pc
    _t0 = _pc()
    
    # 解析参数（带类型推断）
    parser = MLArgumentParser()
    params = parser.parse_args_string(' '.join(args))
    config_path = params.pop('config', None)
    name = params.get('name')
    project = params.get('project')
    # 检测全模型开关
    all_flag = any(flag in args for flag in ['-all', '--all'])
    
    # 解析NUMA和并行参数
    numa_enabled = parser._parse_bool(params.get('numa', False))
    cores_per_task = int(params.get('cores', 4)) if 'cores' in params else None
    parallel_tasks = int(params.get('parallel', 1)) if 'parallel' in params else 1
    bind_cpu = parser._parse_bool(params.get('bind_cpu', False))
    
    # 加载或创建配置
    _t_conf_start = _pc()
    manager = DynamicConfigManager()
    
    if config_path:
        # 尝试获取配置（支持模板名称或文件路径）
        config = manager.get_config(config_path)
        if config:
            print(f"✅ 使用配置: {config_path}")
        else:
            print(f"❌ 配置文件或模板不存在: {config_path}")
            return 1
    else:
        # 使用默认配置
        config = manager.get_config('xgboost_quick')
        if not config:
            # 如果没有找到配置文件，使用内置默认配置
            config = ExperimentConfig()
            print("✅ 使用默认配置")
        else:
            print("✅ 使用默认配置: xgboost_quick")
    
    # 合并命令行参数
    config = parser.merge_params_to_config(config, params)
    _t_conf_end = _pc(); conf_secs = _t_conf_end - _t_conf_start
    
    # 检查是否需要训练多个模型
    models_to_train = []
    if all_flag:
        # 使用 --all 标志训练所有模型
        from models import ModelFactory
        models_to_train = ModelFactory.get_supported_models()
        print("✅ 启用全模型训练模式")
        print(f"   将训练 {len(models_to_train)} 个模型: {models_to_train}")
    elif 'models' in params and params['models']:
        # 从命令行参数获取模型列表
        if isinstance(params['models'], list):
            models_to_train = params['models']
        elif isinstance(params['models'], str):
            # 支持逗号分隔的模型列表
            models_to_train = [m.strip() for m in params['models'].split(',')]
        print("✅ 多模型训练模式")
        print(f"   将训练 {len(models_to_train)} 个模型: {models_to_train}")
    
    # 保存模型列表到配置（用于后续训练）
    if models_to_train:
        config.models_to_train = models_to_train
    
    # 创建运行目录（类似YOLO）
    _t_run_dir_start = _pc()
    # 如果指定了project，使用project作为基础目录，否则使用默认的runs
    if project:
        run_manager = RunManager(base_dir=project, task="train")
        run_dir = run_manager.get_next_run_dir(name=name, project=None)  # project已经作为base_dir了
        # 对于指定project的情况，保持完整的目录结构
        config.logging.base_dir = str(run_dir.parent)
        config.logging.project_name = run_dir.name
    else:
        run_manager = RunManager(task="train")
        run_dir = run_manager.get_next_run_dir(name=name, project=None)
        config.logging.base_dir = str(run_dir.parent)
        config.logging.project_name = run_dir.name
    
    # 显示配置信息（YOLO风格的详细配置）
    print(f"\n" + "="*60)
    print("📋 配置信息 (Configuration)")
    print("="*60)
    
    # 数据配置
    print("\n🗂️  数据配置 (Data):")
    print(f"   训练数据: {config.data.data_path}")
    data_path = Path(config.data.data_path)
    if data_path.exists():
        print(f"   ✅ 训练数据存在 ({data_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"   ❌ 训练数据不存在!")
    
    # 测试数据配置
    if hasattr(config.data, 'test_data_path') and config.data.test_data_path:
        print(f"   测试数据: {config.data.test_data_path}")
        test_path = Path(config.data.test_data_path)
        if test_path.exists():
            print(f"   ✅ 测试数据存在 ({test_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"   ⚠️ 测试数据路径无效: {test_path}")
            # 尝试其他可能的路径
            alt_paths = [
                Path(test_path.name),
                Path("../data") / test_path.name,
                Path("data") / test_path.name
            ]
            for alt in alt_paths:
                if alt.exists():
                    print(f"   💡 找到文件在: {alt}")
                    config.data.test_data_path = str(alt)
                    break
    else:
        print("   测试数据: 未指定")
    
    print(f"   目标列: {config.data.target_columns}")
    print(f"   多目标策略: {config.data.multi_target_strategy}")
    if config.data.multi_target_strategy == "intersection":
        print(f"     → 使用所有目标都有值的数据（最严格）")
    elif config.data.multi_target_strategy == "independent":
        print(f"     → 每个目标独立使用有效数据（默认）")
    elif config.data.multi_target_strategy == "union":
        print(f"     → 使用所有数据，缺失值填充")
    print(f"   缺失值处理: {config.data.nan_handling}")
    if config.data.nan_handling != "skip":
        print(f"     - 特征NaN策略: {config.data.feature_nan_strategy}")
        print(f"     - 目标NaN策略: {config.data.target_nan_strategy}")
    
    # 模型配置
    print("\n🤖 模型配置 (Model):")
    print(f"   模型类型: {config.model.model_type}")
    print(f"   交叉验证: {config.training.n_folds}折")
    if config.model.hyperparameters:
        print("   超参数:")
        for key, value in config.model.hyperparameters.items():
            print(f"     - {key}: {value}")
    
    # 特征配置
    print("\n🔧 特征配置 (Features):")
    print(f"   特征类型: {config.feature.feature_type}")
    if hasattr(config.feature, 'morgan_bits'):
        print(f"   Morgan指纹位数: {config.feature.morgan_bits}")
    if hasattr(config.feature, 'morgan_radius'):
        print(f"   Morgan指纹半径: {config.feature.morgan_radius}")
    print(f"   缓存: {'启用' if config.feature.use_cache else '禁用'}")
    
    # 输出配置
    print("\n📁 输出配置 (Output):")
    print(f"   项目目录: {run_dir}")
    print(f"   模型保存: {run_dir}/models/")
    print(f"   结果导出: {run_dir}/exports/")
    print(f"   特征重要性: {run_dir}/feature_importance/")
    
    print("\n" + "="*60)
    if hasattr(config, 'models_to_train') and config.models_to_train:
        print(f"   多模型训练: 已启用")
        print(f"   训练模型: {len(config.models_to_train)} 个")
        print(f"   模型列表: {', '.join(config.models_to_train[:5])}{'...' if len(config.models_to_train) > 5 else ''}")
    if numa_enabled:
        print(f"   NUMA优化: 已启用")
        print(f"   并行任务数: {parallel_tasks}")
        if cores_per_task:
            print(f"   每任务核心数: {cores_per_task}")
    print(f"   运行目录: {run_dir}")
    
    # 验证配置
    _t_validate_start = _pc()
    if not ConfigValidator.validate_all(config):
        return 1
    _t_validate_end = _pc(); validate_secs = _t_validate_end - _t_validate_start
    
    # 执行训练
    _t_train_start = _pc()
    try:
        # 检查是否需要训练多个模型
        if hasattr(config, 'models_to_train') and config.models_to_train:
            # 多模型训练模式
            if parallel_tasks > 1:
                print(f"\n🚀 启动并行训练: {parallel_tasks} 个并发任务")
                results = parallel_train_models(
                    config, run_dir, 
                    numa_enabled, cores_per_task, parallel_tasks, bind_cpu
                )
            else:
                print(f"\n🚀 串行训练 {len(config.models_to_train)} 个模型...")
                all_results = []
                
                for i, model_type in enumerate(config.models_to_train, 1):
                    print(f"\n[{i}/{len(config.models_to_train)}] 训练模型: {model_type}")
                    print("-" * 40)
                    
                    # 创建模型专用配置（深拷贝）
                    import copy
                    model_config = copy.deepcopy(config)
                    model_config.model.model_type = model_type
                    
                    # 重要：重置超参数为模型特定的默认值，避免使用其他模型的参数
                    from models.base import MODEL_PARAMS
                    if model_type in MODEL_PARAMS:
                        model_config.model.hyperparameters = MODEL_PARAMS[model_type].copy()
                    else:
                        model_config.model.hyperparameters = {}
                    
                    # 修复深拷贝后的配置对象
                    from config.system import ComparisonConfig, ExportConfig
                    if isinstance(model_config.comparison, dict):
                        model_config.comparison = ComparisonConfig(**model_config.comparison)
                    if isinstance(model_config.export, dict):
                        model_config.export = ExportConfig(**model_config.export)
                    
                    # 创建统一的AutoML目录结构
                    automl_dir = run_dir / "automl_train"
                    automl_dir.mkdir(parents=True, exist_ok=True)
                    model_run_dir = automl_dir / model_type
                    model_run_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 更新日志配置，使用父目录
                    model_config.logging.base_dir = str(run_dir.parent)
                    model_config.logging.project_name = f"{run_dir.name}/automl_train/{model_type}"
                    
                    try:
                        # 设置CPU亲和性（如果启用）
                        if numa_enabled and cores_per_task:
                            setup_cpu_affinity(0, cores_per_task, bind_cpu)
                            if 'n_jobs' in model_config.model.hyperparameters:
                                model_config.model.hyperparameters['n_jobs'] = cores_per_task
                        
                        # 训练模型
                        pipeline = TrainingPipeline(model_config)
                        results = pipeline.run()
                        all_results.append({'model': model_type, 'success': True, 'results': results})
                        print(f"✅ {model_type} 训练完成")
                        
                    except Exception as e:
                        print(f"❌ {model_type} 训练失败: {e}")
                        all_results.append({'model': model_type, 'success': False, 'error': str(e)})
                
                # 汇总结果
                results = all_results
        
        else:
            # 单模型训练
            if numa_enabled and cores_per_task:
                # 设置CPU亲和性
                setup_cpu_affinity(0, cores_per_task, bind_cpu)
                # 更新模型的n_jobs参数
                if 'n_jobs' in config.model.hyperparameters:
                    config.model.hyperparameters['n_jobs'] = cores_per_task
            
            pipeline = TrainingPipeline(config)
            results = pipeline.run()
        
        _t_train_end = _pc(); train_secs = _t_train_end - _t_train_start
        total_secs = _pc() - _t0
        print("\n" + "="*60)
        print("✨ 训练完成!")
        print("="*60)
        
        # 如果启用了对比表格生成
        _t_table_start = _pc()
        if (hasattr(config, 'comparison') and hasattr(config.comparison, 'enable') and config.comparison.enable and
            hasattr(config, 'models_to_train') and config.models_to_train):
            print("\n📊 生成模型对比表格...")
            try:
                from utils.comparison_table import ComparisonTableGenerator
                
                # 创建对比表生成器
                generator = ComparisonTableGenerator(str(run_dir))
                
                # 收集所有结果
                df_comparison = generator.collect_all_results()
                
                if not df_comparison.empty:
                    # 导出所有格式
                    formats = config.comparison.formats if hasattr(config.comparison, 'formats') else ['markdown', 'csv']
                    output_files = generator.export_all_formats(
                        output_dir=str(run_dir),
                        formats=formats
                    )
                    
                    print("✅ 对比表格已生成:")
                    for fmt, path in output_files.items():
                        print(f"   - {fmt}: {Path(path).name}")
                else:
                    print("⚠️ 未找到足够的结果生成对比表")
                    
            except Exception as e:
                print(f"❌ 生成对比表失败: {e}")
        _t_table_end = _pc(); table_secs = _t_table_end - _t_table_start
        
        # 如果有测试集，显示测试结果汇总
        if hasattr(config.data, 'test_data_path') and config.data.test_data_path:
            print("\n📊 测试集评估汇总:")
            print("   测试文件: " + Path(config.data.test_data_path).name)
            print("   注: 详细测试结果见上方各目标的测试评估部分")
        
        # 保存运行信息
        run_manager.save_run_info(
            run_dir, 
            config.to_dict(),
            command=' '.join(['automl', 'train'] + args)
        )
        
        # 创建指向最新运行的链接
        RunManager.create_symlink(run_dir, "last")
        
        # 保存配置
        config_save_path = run_dir / "config.yaml"
        config.to_yaml(str(config_save_path))
        print(f"📁 结果保存在: {run_dir}")
        print(f"   查看结果: {run_dir}/exports/")
        print(f"   查看报告: {run_dir}/exports/*.html")
        print(f"   查看模型: {run_dir}/models/")

        # 训练阶段耗时记录（summary + detail）
        try:
            # 尝试向 logger 写入 timing（如果可用）
            if 'training' in locals() or 'pipeline' in locals():
                # pipeline 内部 logger 在运行时已存在（按训练目标写入），这里我们只追加全局 timing 到 summary 文件
                pass
            timing_summary = {
                'startup_to_end': total_secs,
                'config_prepare': conf_secs,
                'validate': validate_secs,
                'training_all': train_secs,
                'comparison_tables': table_secs,
            }
            import json as __json
            with open(_Path(run_dir) / 'timing_summary.json', 'w') as f:
                __json.dump(timing_summary, f, indent=2)
            print(f"   ⏱️ 时间统计保存: {run_dir}/timing_summary.json")

            # 细粒度: 汇总每个实验写入 timing_detail.json（若存在logger导出的实验JSON）
            try:
                detail = {}
                exp_dir = _Path(run_dir) / 'training_logs' / run_dir.name / 'experiments'
                if exp_dir.exists():
                    for p in exp_dir.glob('*_complete.json'):
                        try:
                            with open(p, 'r') as f:
                                exp = __json.load(f)
                            exp_id = exp.get('experiment_id', p.stem.replace('_complete', ''))
                            detail[exp_id] = exp.get('timing', {})
                        except Exception:
                            continue
                with open(_Path(run_dir) / 'timing_detail.json', 'w') as f:
                    __json.dump(detail, f, indent=2, ensure_ascii=False)
                print(f"   ⏱️ 细粒度时间统计保存: {run_dir}/timing_detail.json")
            except Exception:
                pass
        except Exception:
            pass

        # 论文完整资料包整合（仅在 paper_comparison 或显式开启 comparison.enable 时启用）
        try:
            is_paper_mode = (config.name.lower().startswith('paper_comparison') if hasattr(config, 'name') else False)
        except Exception:
            is_paper_mode = False

        should_make_paper_package = False
        try:
            if hasattr(config, 'comparison') and hasattr(config.comparison, 'enable'):
                should_make_paper_package = bool(config.comparison.enable)
        except Exception:
            pass
        should_make_paper_package = should_make_paper_package or is_paper_mode

        if should_make_paper_package:
            try:
                from utils.comparison_table import ComparisonTableGenerator
                from pathlib import Path as _Path
                import shutil as _shutil
                import json as _json
                paper_dir = _Path(run_dir) / 'paper_complete'
                paper_dir.mkdir(parents=True, exist_ok=True)

                # 1) 表格导出（四种格式）
                generator = ComparisonTableGenerator(str(run_dir))
                exported = generator.export_all_formats(output_dir=str(paper_dir), formats=['markdown','html','latex','csv'])

                # 2) 生成论文图（含数据）
                try:
                    from scripts.generate_paper_figures import generate_all_figures
                    data_path = config.data.data_path if hasattr(config, 'data') else 'data/Database_normalized.csv'
                    generate_all_figures(str(run_dir), data_path, str(paper_dir))
                except Exception as e:
                    print(f"⚠️ 生成论文图表失败: {e}")

                # 3) 保留测试集原始预测与真值（若有）到 paper_complete
                try:
                    from pathlib import Path as __Path
                    exports_dir = __Path(run_dir) / 'exports'
                    if exports_dir.exists():
                        for f in exports_dir.glob('test_predictions_*.csv'):
                            __shutil.copy(f, paper_dir / f.name)
                        for f in exports_dir.glob('test_metrics_*.json'):
                            __shutil.copy(f, paper_dir / f.name)
                except Exception:
                    pass

                # 4) 汇总文件与配置
                from datetime import datetime as _datetime
                import numpy as ___np

                # 定义一个JSON编码器来处理numpy类型
                class NumpyEncoder(_json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, (___np.integer, ___np.int64)):
                            return int(obj)
                        elif isinstance(obj, (___np.floating, ___np.float64)):
                            return float(obj)
                        elif isinstance(obj, ___np.ndarray):
                            return obj.tolist()
                        return super().default(obj)

                summary = {
                    'project': str(run_dir.name),
                    'path': str(run_dir),
                    'timestamp': _datetime.now().isoformat(),
                    'comparison_tables': {k: _Path(v).name for k, v in exported.items()},
                    'best_models': generator.get_best_models() if exported else {},
                }
                with open(paper_dir / 'summary.json', 'w', encoding='utf-8') as f:
                    _json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

                # 保存最终配置副本
                try:
                    (_Path(run_dir) / 'config.yaml').replace(paper_dir / 'config.yaml')
                except Exception:
                    try:
                        import shutil as __shutil
                        __shutil.copy(_Path(run_dir) / 'config.yaml', paper_dir / 'config.yaml')
                    except Exception:
                        pass

                # 5) 可选首页 index.html
                try:
                    index_path = paper_dir / 'index.html'
                    index_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head><meta charset='utf-8'><title>Paper Complete Package</title></head>
                    <body>
                      <h1>Paper Complete Package</h1>
                      <ul>
                        <li><a href="{_Path(exported.get('html','')).name if exported else ''}">Comparison Table (HTML)</a></li>
                        <li><a href="summary.json">Summary (JSON)</a></li>
                        <li><a href="../timing_summary.json">Timing Summary</a></li>
                        <li><a href="../timing_detail.json">Timing Detail</a></li>
                        <li><a href="figure_c_wavelength_plqy.png">Figure C</a></li>
                        <li><a href="figure_d_plqy_distribution.png">Figure D</a></li>
                        <li><a href="figure_e_f_predictions.png">Figure E-F</a></li>
                        <li><a href="figure_g_plqy_accuracy.png">Figure G</a></li>
                      </ul>
                    </body>
                    </html>
                    """
                    with open(index_path, 'w', encoding='utf-8') as f:
                        f.write(index_html)
                except Exception:
                    pass

                # 追加 timing 到 summary
                try:
                    import json as ___json
                    import numpy as ___np
                    
                    # 定义一个JSON编码器来处理numpy类型
                    class NumpyEncoder(___json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, (___np.integer, ___np.int64)):
                                return int(obj)
                            elif isinstance(obj, (___np.floating, ___np.float64)):
                                return float(obj)
                            elif isinstance(obj, ___np.ndarray):
                                return obj.tolist()
                            return super().default(obj)
                    
                    s_path = paper_dir / 'summary.json'
                    if s_path.exists():
                        data = ___json.load(open(s_path, 'r'))
                    else:
                        data = {}
                    data['timing'] = timing_summary if 'timing_summary' in locals() else {}
                    with open(s_path, 'w') as f:
                        ___json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                except Exception:
                    pass

                # 6) 可选自动发布到后端（通过环境变量控制）
                try:
                    import os as ___os
                    from utils.publisher import ResultsPublisher
                    api_url = ___os.getenv('RESULTS_API_URL', '').strip()
                    if api_url:
                        print("\n🌐 发布论文资料包到后端...")
                        publisher = ResultsPublisher()
                        resp = publisher.publish_package(
                            str(paper_dir),
                            metadata={'project': run_dir.name, 'path': str(run_dir)}
                        )
                        if resp:
                            print(f"✅ 发布成功: {resp}")
                        else:
                            print("⚠️ 发布未返回成功响应")
                except Exception as e:
                    print(f"⚠️ 发布过程异常: {e}")

                print(f"\n📦 论文资料包已生成: {paper_dir}")
            except Exception as e:
                print(f"⚠️ 整合论文资料包失败: {e}")

        
        # 打印示例预测指令（为本次训练产生的所有模型逐一打印：单配体/多配体）
        try:
            models_dir = run_dir / "models"
            model_paths = []
            if models_dir.exists():
                model_paths = sorted(
                    [p for p in models_dir.glob("*.joblib")],
                    key=lambda p: p.stat().st_mtime
                )
            # 回退：查找 run_dir 下所有 joblib
            if not model_paths:
                model_paths = sorted(
                    [p for p in run_dir.glob("**/*.joblib")],
                    key=lambda p: p.stat().st_mtime
                )
            if model_paths:
                print("\n📌 示例预测指令（复制后可直接运行，按模型列出）：")
                for mp in model_paths:
                    print(f"  # {mp.name}")
                    # 检查文件名是否包含特殊字符，如果有则用引号包裹
                    model_param = f"model={mp}"
                    if any(char in str(mp) for char in ['(', ')', '[', ']', '{', '}', ' ', '*', '?']):
                        model_param = f'"model={mp}"'
                    
                    # 单样本：使用数据集中真实示例的 L1/L2/L3
                    print(f"  python automl.py predict {model_param} input='[[\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"C1=CN=C(C2=CN(CCCCCCN3C4=CC=CC=C4C4=C3C=CC=C4)N=N2)C=C1\"]]' feature=combined")
                    # 双样本：重复该三联体作为第二个样本
                    print(f"  python automl.py predict {model_param} input='[[\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"C1=CN=C(C2=CN(CCCCCCN3C4=CC=CC=C4C4=C3C=CC=C4)N=N2)C=C1\"],[\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"C1=CN=C(C2=CN(CCCCCCN3C4=CC=CC=C4C4=C3C=CC=C4)N=N2)C=C1\"]]' feature=combined")
        except Exception:
            pass
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ 检测到中断，保存已完成的部分并输出示例预测命令...")
        try:
            # 尝试保存运行信息与配置
            run_manager.save_run_info(
                run_dir,
                config.to_dict(),
                command=' '.join(['automl', 'train'] + args)
            )
            RunManager.create_symlink(run_dir, "last")
            config_save_path = run_dir / "config.yaml"
            config.to_yaml(str(config_save_path))
        except Exception:
            pass
        # 尝试打印当前已有模型的预测命令
        try:
            models_dir = run_dir / "models"
            model_paths = []
            if models_dir.exists():
                model_paths = sorted(
                    [p for p in models_dir.glob("*.joblib")],
                    key=lambda p: p.stat().st_mtime
                )
            if model_paths:
                print("\n📌 已完成模型的示例预测指令：")
                for mp in model_paths:
                    print(f"  # {mp.name}")
                    print(f"  python automl.py predict model={mp} input='[[\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"C1=CN=C(C2=CN(CCCCCCN3C4=CC=CC=C4C4=C3C=CC=C4)N=N2)C=C1\"]]' feature=combined")
            else:
                print("⚠️ 尚未产生模型文件。")
        except Exception:
            pass
        return 130  # 常见中断退出码

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ========================================
#           预测命令
# ========================================

def predict_command(args: List[str]):
    """预测命令"""
    print("\n" + "="*60)
    print("AutoML Prediction System")
    print("="*60)
    
    # 解析参数
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    
    # 检查必要参数
    if 'model' not in params:
        print("❌ 缺少模型参数: model=path/to/model.joblib")
        return 1
    
    if 'data' not in params and 'input' not in params:
        print("❌ 需要提供数据: data=path/to/data.csv 或 input=[\"CCO\",\"c1ccccc1\"]")
        return 1
    
    # 加载模型
    print(f"\n📦 加载模型: {params['model']}")
    try:
        model = load_model(params['model'])
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return 1
    
    # 推断训练配置（用于自动对齐特征类型与 SMILES 列）
    training_feature_type = None
    training_smiles_columns = None
    training_morgan_bits = None
    training_morgan_radius = None
    try:
        model_path = Path(params['model']).resolve()
        # 常见保存位置: runs/.../models/*.joblib → runs/.../config.yaml
        run_dir = model_path.parent.parent if model_path.parent.name == 'models' else model_path.parent
        config_candidates = [run_dir / 'config.yaml', run_dir / 'experiment_config.yaml']
        for cfg in config_candidates:
            if cfg.exists():
                try:
                    from config.system import ExperimentConfig
                    cfg_obj = ExperimentConfig.from_yaml(str(cfg)) if cfg.suffix in ['.yml', '.yaml'] else ExperimentConfig.from_json(str(cfg))
                    training_feature_type = str(getattr(cfg_obj.feature, 'feature_type', None)).lower()
                    training_smiles_columns = list(getattr(cfg_obj.data, 'smiles_columns', []))
                    training_morgan_bits = getattr(cfg_obj.feature, 'morgan_bits', None)
                    training_morgan_radius = getattr(cfg_obj.feature, 'morgan_radius', None)
                    break
                except Exception:
                    pass
    except Exception:
        pass
    
    # 解析/决策特征类型
    feature_param = params.get('feature')
    if feature_param is None or str(feature_param).lower() == 'auto':
        feature_type = (training_feature_type or 'combined').lower()
        if training_feature_type:
            print(f"🔁 按训练配置自动设置特征类型: {feature_type}")
    else:
        feature_type = str(feature_param).lower()
    
    # 解析/决策 SMILES 列
    smiles_param = params.get('smiles_columns')
    if smiles_param:
        resolved_smiles_cols = [c.strip() for c in smiles_param.split(',') if c.strip()]
        print(f"📌 使用指定的 SMILES 列: {','.join(resolved_smiles_cols)}")
    else:
        resolved_smiles_cols = training_smiles_columns or ['L1', 'L2', 'L3']
        if training_smiles_columns:
            print(f"🔁 按训练配置自动设置 SMILES 列: {','.join(resolved_smiles_cols)}")
    expected_ligand_count = len(resolved_smiles_cols)
    
    # 解析输出列名
    output_column = params.get('output_column', 'Prediction')
    
    # 解析批处理参数
    batch_size = int(params.get('batch_size', '1000'))
    show_progress = params.get('show_progress', 'true').lower() in ['true', '1', 'yes']
    skip_errors = params.get('skip_errors', 'true').lower() in ['true', '1', 'yes']
    
    # 准备特征
    print("\n🔧 准备特征...")
    from core.feature_extractor import FeatureExtractor
    X = None
    df = None
    
    # 允许通过命令指定 morgan_bits/morgan_radius（兼容别名 bits/radius）
    morgan_bits = params.get('morgan_bits', params.get('bits'))
    morgan_radius = params.get('morgan_radius', params.get('radius'))
    try:
        morgan_bits = int(morgan_bits) if morgan_bits is not None else None
    except ValueError:
        morgan_bits = None
    try:
        morgan_radius = int(morgan_radius) if morgan_radius is not None else None
    except ValueError:
        morgan_radius = None
    # 若未显式提供，则按训练配置自动设置
    if morgan_bits is None and training_morgan_bits is not None:
        morgan_bits = int(training_morgan_bits)
        print(f"🔁 按训练配置自动设置 morgan_bits: {morgan_bits}")
    if morgan_radius is None and training_morgan_radius is not None:
        morgan_radius = int(training_morgan_radius)
        print(f"🔁 按训练配置自动设置 morgan_radius: {morgan_radius}")
    feature_extractor = FeatureExtractor(use_cache=True, morgan_bits=morgan_bits, morgan_radius=morgan_radius)
    
    if 'input' in params:
        raw_input = params['input']
        user_input = None
        if isinstance(raw_input, list):
            user_input = raw_input
        else:
            try:
                user_input = json.loads(raw_input)
            except Exception:
                # 退化处理：按逗号拆分字符串
                user_input = [s.strip() for s in str(raw_input).split(',') if s.strip()]
        
        print("📥 使用 inline input 进行预测")
        if feature_type in ['morgan', 'descriptors', 'combined']:
            # 规范化为每个样本一个 SMILES 列表
            samples = []
            if all(isinstance(x, str) for x in user_input):
                samples = [[s] for s in user_input]
            elif all(isinstance(x, (list, tuple)) for x in user_input):
                samples = [list(sample) for sample in user_input]
            else:
                print("❌ input 格式不支持。对于分子特征，使用 ['SMI', ...] 或 [['L1','L2'], ...]")
                return 1
            
            # 按训练需要的配体数自动补齐/截断
            if expected_ligand_count > 0:
                adjusted = False
                for i in range(len(samples)):
                    if len(samples[i]) < expected_ligand_count:
                        samples[i] = samples[i] + [None] * (expected_ligand_count - len(samples[i]))
                        adjusted = True
                    elif len(samples[i]) > expected_ligand_count:
                        samples[i] = samples[i][:expected_ligand_count]
                        adjusted = True
                if adjusted:
                    print(f"ℹ️ 已按训练配置对齐配体数: 期望 {expected_ligand_count}，已自动补齐/截断")

            features = []
            for smiles_list in samples:
                feat = feature_extractor.extract_combination(
                    smiles_list,
                    feature_type=feature_type,
                    combination_method='mean'
                )
                features.append(feat)
            X = np.array(features)
            # 为了导出结果，保留一个最小 df
            df = pd.DataFrame({'L1_L2_L3': [','.join([s for s in sm if s is not None]) for sm in samples]})
        else:
            # tabular/auto: 直接使用数值/数组
            arr = np.array(user_input, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X = arr
            df = pd.DataFrame({'row': list(range(len(X)))})
    else:
        # 从 CSV 读取 - 使用批处理优化
        print(f"📊 加载数据: {params['data']}")
        try:
            df = pd.read_csv(params['data'])
            print(f"   数据形状: {df.shape}")
            print(f"   列名: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return 1
        
        # 检查SMILES列是否存在
        missing_cols = [col for col in resolved_smiles_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  警告: 缺少列 {missing_cols}, 将使用 None 值")
        
        if feature_type in ['morgan', 'descriptors', 'combined']:
            # 使用增强版批处理预测（带文件缓存）
            print(f"\n🚀 使用批处理模式 (batch_size={batch_size})")
            
            # 检查是否使用文件缓存
            use_file_cache = params.get('use_file_cache', 'true').lower() in ['true', '1', 'yes']
            file_cache_dir = params.get('file_cache_dir', 'file_feature_cache')
            
            # 使用V2版本的批处理器
            from utils.batch_predictor_v2 import BatchPredictorV2
            
            predictor = BatchPredictorV2(
                batch_size=batch_size,
                show_progress=show_progress,
                skip_errors=skip_errors,
                use_file_cache=use_file_cache,
                file_cache_dir=file_cache_dir
            )
            
            predictions, failed_indices = predictor.predict_with_cache(
                df=df,
                model=model,
                feature_extractor=feature_extractor,
                smiles_columns=resolved_smiles_cols,
                feature_type=feature_type,
                combination_method='mean',
                input_file=params['data']  # 传递文件路径用于缓存
            )
            
            # 添加预测列到原始数据框
            df[output_column] = predictions
            
            # 显示统计信息
            stats = predictor.get_statistics(predictions)
            print(f"\n📊 预测统计:")
            print(f"   成功: {stats['count']} / {len(df)} ({stats['success_rate']:.1f}%)")
            if stats['count'] > 0:
                print(f"   最小值: {stats['min']:.4f}")
                print(f"   最大值: {stats['max']:.4f}")
                print(f"   平均值: {stats['mean']:.4f}")
                print(f"   标准差: {stats['std']:.4f}")
            
            # 保存错误日志
            if failed_indices and skip_errors:
                error_file = params.get('output', 'predictions.csv').replace('.csv', '_errors.log')
                predictor.save_error_log(error_file)
            
            # 跳过后续的预测步骤，直接保存
            output_path = params.get('output', None)
            
            # 如果没有指定输出文件，使用固定文件名并覆盖
            if output_path is None:
                output_path = 'predictions.csv'
            
            df.to_csv(output_path, index=False)
            
            # 获取绝对路径
            from pathlib import Path
            abs_path = Path(output_path).absolute()
            
            print(f"\n💾 预测结果已保存:")
            print(f"   文件: {output_path}")
            print(f"   完整路径: {abs_path}")
            print(f"   保留了所有 {len(df.columns)} 列")
            
            # 显示预览
            print(f"\n📋 预测结果预览:")
            preview_df = df.copy()
            
            # 限制SMILES显示长度
            for col in resolved_smiles_cols:
                if col in preview_df.columns:
                    preview_df[col] = preview_df[col].apply(
                        lambda x: str(x)[:30] + '...' if isinstance(x, str) and len(str(x)) > 30 else x
                    )
            
            # 显示前后几行
            print("-" * 80)
            if len(preview_df) <= 20:
                print(preview_df.to_string(index=False))
            else:
                print("前5行:")
                print(preview_df.head(5).to_string(index=False))
                print("\n后5行:")
                print(preview_df.tail(5).to_string(index=False))
                print(f"\n(共 {len(preview_df)} 行)")
            print("-" * 80)
            
            return 0
        else:
            # tabular 或 auto 模式
            target_cols = []
            if 'target' in params:
                target_cols = [t.strip() for t in str(params['target']).split(',') if t.strip()]
            X = feature_extractor.extract_from_dataframe(
                df,
                target_columns=target_cols or None,
                feature_type=feature_type
            )
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"   特征维度: {X.shape}")
    
    # 预测
    print("\n🎯 执行预测...")
    try:
        predictions = model.predict(X)
        print(f"   预测完成: {len(predictions)} 个样本")
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return 1
    
    # 保存结果 - 保留所有原始列
    output_path = params.get('output', None)
    
    # 如果没有指定输出文件，使用固定文件名并覆盖
    if output_path is None:
        output_path = 'predictions.csv'
    
    if df is None:
        df = pd.DataFrame()
    
    # 使用用户指定的输出列名
    df[output_column] = predictions
    df.to_csv(output_path, index=False)
    
    # 获取绝对路径
    from pathlib import Path
    abs_path = Path(output_path).absolute()
    
    print(f"\n💾 预测结果已保存:")
    print(f"   文件: {output_path}")
    print(f"   完整路径: {abs_path}")
    print(f"   保留了所有 {len(df.columns)} 列")
    
    # 显示统计
    print(f"\n📊 预测统计:")
    print(f"   最小值: {predictions.min():.4f}")
    print(f"   最大值: {predictions.max():.4f}")
    print(f"   平均值: {predictions.mean():.4f}")
    print(f"   标准差: {predictions.std():.4f}")
    
    # 显示预测结果表格
    print(f"\n📋 预测结果预览:")
    # 如果有原始数据的标识信息，一起显示
    preview_df = df.copy() if df is not None else pd.DataFrame()
    
    # 选择要显示的列（如果存在）
    display_cols = []
    for col in ['Unnamed: 0', 'Abbreviation_in_the_article', 'L1', 'L2', 'L3']:
        if col in preview_df.columns:
            display_cols.append(col)
    
    # 限制SMILES显示长度
    if display_cols:
        preview_df = preview_df[display_cols].copy()
        for col in ['L1', 'L2', 'L3']:
            if col in preview_df.columns:
                preview_df[col] = preview_df[col].apply(lambda x: str(x)[:30] + '...' if isinstance(x, str) and len(str(x)) > 30 else x)
    
    preview_df['Prediction'] = predictions
    preview_df['Prediction'] = preview_df['Prediction'].round(4)
    
    # 打印表格
    print("-" * 80)
    if len(preview_df) <= 20:
        print(preview_df.to_string(index=False))
    else:
        print("前10行:")
        print(preview_df.head(10).to_string(index=False))
        print("\n后10行:")
        print(preview_df.tail(10).to_string(index=False))
        print(f"\n(共 {len(preview_df)} 行)")
    print("-" * 80)
    
    return 0


# ========================================
#           验证命令
# ========================================

def validate_command(args: List[str]):
    """验证命令 - 支持验证配置文件或数据文件"""
    print("\n" + "="*60)
    print("AutoML Validator")
    print("="*60)
    
    # 解析参数
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    
    # 检查是验证数据还是配置
    data_path = params.get('data')
    config_path = params.get('config')
    
    if data_path:
        # 验证数据文件
        print(f"\n📊 验证数据文件: {data_path}")
        
        # 检查文件是否存在
        if not Path(data_path).exists():
            print(f"❌ 数据文件不存在: {data_path}")
            return 1
        
        try:
            # 加载数据
            import pandas as pd
            df = pd.read_csv(data_path)
            print(f"✅ 数据加载成功")
            
            # 显示数据信息
            print("\n数据信息:")
            print("-" * 40)
            print(f"行数: {len(df)}")
            print(f"列数: {len(df.columns)}")
            print(f"列名: {', '.join(df.columns[:10])}")
            if len(df.columns) > 10:
                print(f"      ... 还有 {len(df.columns)-10} 列")
            
            # 检查必要的列
            smiles_cols = ['L1', 'L2', 'L3']
            target_cols = ['Max_wavelength(nm)', 'PLQY', 'tau(s*10^-6)']
            
            print("\n🔍 检查必要列...")
            
            # 检查SMILES列
            has_smiles = any(col in df.columns for col in smiles_cols)
            if has_smiles:
                found_smiles = [col for col in smiles_cols if col in df.columns]
                print(f"✅ SMILES列: {', '.join(found_smiles)}")
            else:
                print(f"⚠️  未找到SMILES列 (期望: {', '.join(smiles_cols)})")
            
            # 检查目标列
            has_targets = any(col in df.columns for col in target_cols)
            if has_targets:
                found_targets = [col for col in target_cols if col in df.columns]
                print(f"✅ 目标列: {', '.join(found_targets)}")
            else:
                print(f"⚠️  未找到目标列 (期望: {', '.join(target_cols)})")
            
            # 检查数据质量
            print("\n📈 数据质量检查:")
            print(f"缺失值总数: {df.isnull().sum().sum()}")
            print(f"重复行数: {df.duplicated().sum()}")
            
            # 如果有SMILES列，检查SMILES有效性
            if has_smiles:
                try:
                    from rdkit import Chem
                    invalid_count = 0
                    for col in found_smiles:
                        if col in df.columns:
                            # 取样检查（最多100个）
                            sample = df[col].dropna().head(100)
                            for smiles in sample:
                                if pd.notna(smiles) and smiles != '':
                                    mol = Chem.MolFromSmiles(str(smiles))
                                    if mol is None:
                                        invalid_count += 1
                    if invalid_count > 0:
                        print(f"⚠️  发现 {invalid_count} 个无效SMILES")
                    else:
                        print(f"✅ SMILES格式检查通过")
                except ImportError:
                    print("ℹ️  RDKit未安装，跳过SMILES验证")
            
            print("\n✅ 数据验证完成!")
            return 0
            
        except Exception as e:
            print(f"❌ 数据验证失败: {e}")
            return 1
    
    elif config_path:
        # 验证配置文件
        print(f"\n📋 验证配置文件: {config_path}")
        
        if not Path(config_path).exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return 1
        
        try:
            if config_path.endswith('.yaml'):
                config = ExperimentConfig.from_yaml(config_path)
            else:
                config = ExperimentConfig.from_json(config_path)
            
            # 显示配置
            print("\n配置内容:")
            print("-" * 40)
            print(f"名称: {config.name}")
            print(f"描述: {config.description}")
            print(f"模型: {config.model.model_type}")
            print(f"特征: {config.feature.feature_type}")
            print(f"数据: {config.data.data_path}")
            print(f"目标: {config.data.target_columns}")
            print(f"交叉验证: {config.training.n_folds}折")
            
            # 验证配置
            print("\n🔍 验证配置...")
            if ConfigValidator.validate_all(config):
                print("✅ 配置验证通过!")
                return 0
            else:
                print("❌ 配置验证失败!")
                return 1
                
        except Exception as e:
            print(f"❌ 配置加载失败: {e}")
            return 1
    
    else:
        # 默认查找配置文件
        if Path('config.yaml').exists():
            return validate_command(['config=config.yaml'])
        elif Path('config.json').exists():
            return validate_command(['config=config.json'])
        else:
            print("❌ 请指定要验证的文件:")
            print("   验证数据: automl validate data=<数据文件>")
            print("   验证配置: automl validate config=<配置文件>")
            return 1


# ========================================
#           导出命令
# ========================================

def export_command(args: List[str]):
    """导出命令"""
    print("\n" + "="*60)
    print("AutoML Model Export System")
    print("="*60)
    
    # 解析参数
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    
    model_path = params.get('model')
    format_type = params.get('format', 'onnx')
    output_path = params.get('output', 'exported_model')
    
    if not model_path:
        print("❌ 缺少模型参数: model=path/to/model.joblib")
        return 1
    
    # 加载模型
    print(f"\n📦 加载模型: {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return 1
    
    # 导出模型
    print(f"📤 导出为 {format_type} 格式...")
    
    if format_type == 'onnx':
        try:
            import skl2onnx
            from skl2onnx import convert_sklearn
            
            # 需要输入形状信息
            n_features = int(params.get('n_features', 1109))
            initial_type = [('float_input', skl2onnx.common.data_types.FloatTensorType([None, n_features]))]
            
            onx = convert_sklearn(model, initial_types=initial_type)
            
            with open(f"{output_path}.onnx", "wb") as f:
                f.write(onx.SerializeToString())
            
            print(f"✅ 模型已导出: {output_path}.onnx")
            
        except ImportError:
            print("❌ 需要安装 skl2onnx: pip install skl2onnx")
            return 1
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return 1
    
    elif format_type == 'pmml':
        print("❌ PMML导出暂未实现")
        return 1
    
    elif format_type == 'pickle':
        import pickle
        with open(f"{output_path}.pkl", 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ 模型已导出: {output_path}.pkl")
    
    else:
        print(f"❌ 不支持的格式: {format_type}")
        return 1
    
    return 0


# ========================================
#           分析命令
# ========================================

def analyze_command(args: List[str]):
    """分析实验结果"""
    print("\n" + "="*60)
    print("AutoML Results Analysis")
    print("="*60)
    
    # 解析参数
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    
    # 获取运行目录
    run_dir = params.get('run_dir', params.get('dir', 'runs/train'))
    output_format = params.get('format', 'text')
    output_path = params.get('output')
    print_results = params.get('print', 'true').lower() == 'true'
    
    # 转换为Path对象
    run_dir = Path(run_dir)
    
    # 如果使用 'last' 关键字，查找最新的运行
    if str(run_dir) == 'last':
        # 查找最新的运行目录
        if Path('runs/train').exists():
            run_dirs = sorted([d for d in Path('runs/train').iterdir() if d.is_dir() and d.name != 'last'])
            if run_dirs:
                run_dir = run_dirs[-1]
            else:
                print("❌ 没有找到训练运行记录")
                return 1
        else:
            print("❌ 没有找到训练运行记录")
            return 1
    
    # 检查目录是否存在
    if not run_dir.exists():
        print(f"❌ 运行目录不存在: {run_dir}")
        print("\n可用的运行目录:")
        
        # 列出可用的运行目录
        for base_dir in ['runs', '.']:
            base_path = Path(base_dir)
            if base_path.exists():
                for task_dir in base_path.iterdir():
                    if task_dir.is_dir() and not task_dir.name.startswith('.'):
                        sub_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name != 'last']
                        if sub_dirs:
                            print(f"  {task_dir}:")
                            for d in sorted(sub_dirs)[-5:]:  # 显示最近5个
                                print(f"    - {d}")
        return 1
    
    print(f"\n📂 分析目录: {run_dir}")
    
    # 创建分析器
    try:
        analyzer = ResultsAnalyzer(run_dir)
    except Exception as e:
        print(f"❌ 创建分析器失败: {e}")
        return 1
    
    # 生成报告
    print(f"📊 生成{output_format.upper()}格式报告...")
    
    try:
        # 保存报告
        if output_path:
            output_path = Path(output_path)
        analyzer.save_report(output_path=output_path, output_format=output_format)
        
        # 打印到控制台
        if print_results:
            print("\n" + "="*60)
            print(analyzer.generate_report('text'))
            print("="*60)
        
        print("\n✅ 分析完成!")
        return 0
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ========================================
#           信息命令
# ========================================

def info_command(args: List[str]):
    """显示系统信息"""
    print("\n" + "="*60)
    print("AutoML System Information")
    print("="*60)
    
    # 系统信息
    print("\n📊 系统信息:")
    print(f"   Python版本: {sys.version.split()[0]}")
    print(f"   平台: {sys.platform}")
    
    # 可用模型
    from models import ModelFactory
    print("\n🤖 可用模型:")
    for model in ModelFactory.get_supported_models():
        print(f"   - {model}")
    
    # 可用模板
    manager = ConfigManager()
    print("\n📋 配置模板:")
    for template in manager.list_templates():
        desc = manager.templates[template].description
        print(f"   - {template}: {desc}")
    
    # 特征类型
    print("\n🔧 特征类型:")
    print("   - morgan: Morgan指纹")
    print("   - descriptors: 分子描述符")
    print("   - combined: 组合特征")
    
    # 使用示例
    print("\n💡 使用示例:")
    print("   训练: automl train model=xgboost data=data.csv config=config.yaml")
    print("   分析: automl analyze dir=quick_test format=html")
    print("   预测: automl predict model=model.joblib data=test.csv")
    print("   验证: automl validate config=config.yaml")
    print("   导出: automl export model=model.joblib format=onnx")
    
    return 0


# ========================================
#           NUMA和并行支持
# ========================================

def setup_cpu_affinity(task_id: int, cores_per_task: int, bind_cpu: bool = False):
    """
    设置CPU亲和性和NUMA绑定
    
    Args:
        task_id: 任务ID
        cores_per_task: 每个任务使用的核心数
        bind_cpu: 是否绑定CPU
    """
    if not bind_cpu:
        return
    
    try:
        # 获取系统CPU信息
        cpu_count = psutil.cpu_count(logical=True)
        
        # 计算核心范围
        core_start = (task_id * cores_per_task) % cpu_count
        core_end = min(core_start + cores_per_task, cpu_count)
        cores = list(range(core_start, core_end))
        
        # 设置CPU亲和性
        p = psutil.Process()
        p.cpu_affinity(cores)
        
        print(f"   ✅ CPU亲和性设置: 任务{task_id} -> 核心{cores}")
        
    except Exception as e:
        print(f"   ⚠️ 无法设置CPU亲和性: {e}")


def get_numa_info():
    """获取NUMA信息"""
    try:
        import subprocess
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'available:' in line and 'nodes' in line:
                    numa_nodes = int(line.split()[1])
                    return numa_nodes
    except:
        pass
    return 1


# ========================================
#           预热缓存命令
# ========================================

def warmup_command(args: List[str]):
    """预计算并写入特征缓存（支持分子/表格），避免训练阶段并发提取开销"""
    print("\n" + "="*60)
    print("AutoML Cache Warmup")
    print("="*60)

    # 解析参数（key=value）
    params = {}
    for arg in args:
        if '=' in arg:
            k, v = arg.split('=', 1)
            params[k] = v

    # 必要参数
    data_path = params.get('data')
    if not data_path:
        print("❌ 缺少参数: data=path/to.csv")
        return 1

    feature_type = str(params.get('feature', 'auto')).lower()
    smiles_columns = params.get('smiles_columns')
    if smiles_columns:
        smiles_columns = [c.strip() for c in smiles_columns.split(',') if c.strip()]
    morgan_bits = params.get('morgan_bits', params.get('bits'))
    morgan_radius = params.get('morgan_radius', params.get('radius'))
    try:
        morgan_bits = int(morgan_bits) if morgan_bits is not None else None
        morgan_radius = int(morgan_radius) if morgan_radius is not None else None
    except Exception:
        morgan_bits = None
        morgan_radius = None

    # 并发参数（预热阶段本命令内部串行写缓存，避免竞争；可加 n_jobs 做行内并行）
    n_jobs = int(params.get('n_jobs', 0))

    # 加载数据
    import pandas as pd
    import numpy as np
    from core.feature_extractor import FeatureExtractor

    print(f"\n📊 加载数据: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return 1
    print(f"   形状: {df.shape}")

    # 构建提取器
    extractor = FeatureExtractor(
        feature_type=feature_type,
        use_cache=True,
        morgan_bits=morgan_bits,
        morgan_radius=morgan_radius
    )

    # 自动识别 smiles 列
    if feature_type in ['morgan', 'descriptors', 'combined', 'auto']:
        if not smiles_columns:
            # 若 auto/molecular，尝试从 DF 猜测
            guessed = [col for col in df.columns if any(ind in col.lower() for ind in ['smiles','l1','l2','l3'])]
            smiles_columns = guessed or ['L1','L2','L3']

    print(f"   特征类型: {feature_type}")
    if smiles_columns:
        print(f"   SMILES列: {','.join(smiles_columns)}")
    if morgan_bits:
        print(f"   morgan_bits: {morgan_bits}")
    if morgan_radius:
        print(f"   morgan_radius: {morgan_radius}")

    # 预热：逐行提取（必要时可加入 tqdm）
    from tqdm import tqdm
    total = len(df)
    errors = 0

    if feature_type in ['morgan', 'descriptors', 'combined'] or (
        feature_type == 'auto' and extractor.detect_data_type(df) == 'molecular'
    ):
        # 分子路径
        for _, row in tqdm(df.iterrows(), total=total, desc='预热分子特征缓存'):
            smiles_list = [row[col] if col in row and pd.notna(row[col]) else None for col in smiles_columns]
            try:
                _ = extractor.extract_combination(smiles_list, feature_type=feature_type if feature_type!='auto' else 'combined')
            except Exception:
                errors += 1
                continue
    else:
        # 表格路径：一次性写入（内部会缓存列级特征名，不逐行）
        try:
            _ = extractor.extract_from_dataframe(df, target_columns=[] if 'target' not in params else [params['target']])
        except Exception:
            errors += 1

    print(f"\n✅ 预热完成: {total - errors}/{total} 条记录已写入/命中缓存")
    return 0

def train_single_model_parallel(args):
    """
    并行训练单个模型的工作函数
    
    Args:
        args: (config, model_type, task_id, numa_enabled, cores_per_task, bind_cpu)
    """
    config, model_type, task_id, numa_enabled, cores_per_task, bind_cpu = args
    
    # 设置CPU亲和性
    if numa_enabled and cores_per_task:
        setup_cpu_affinity(task_id, cores_per_task, bind_cpu)
    
    # 重建配置对象（从字典或配置对象）
    from config.system import ExperimentConfig
    if isinstance(config, dict):
        config = ExperimentConfig.from_dict(config)
    else:
        config = ExperimentConfig.from_dict(config.to_dict())  # 深拷贝
    
    config.model.model_type = model_type
    
    # 重要：重置超参数为模型特定的默认值，避免使用其他模型的参数
    from models.base import MODEL_PARAMS
    if model_type in MODEL_PARAMS:
        config.model.hyperparameters = MODEL_PARAMS[model_type].copy()
    else:
        config.model.hyperparameters = {}
    
    config.logging.project_name = f"{config.logging.project_name}_{model_type}"
    
    # 设置n_jobs
    if cores_per_task and 'n_jobs' in config.model.hyperparameters:
        config.model.hyperparameters['n_jobs'] = cores_per_task
    
    # 执行训练
    try:
        from training.pipeline import TrainingPipeline
        pipeline = TrainingPipeline(config)
        results = pipeline.run()
        return {'model': model_type, 'success': True, 'results': results}
    except Exception as e:
        return {'model': model_type, 'success': False, 'error': str(e)}


def parallel_train_models(config, run_dir, numa_enabled=False, 
                         cores_per_task=None, parallel_tasks=8, bind_cpu=False):
    """
    并行训练多个模型
    
    Args:
        config: 实验配置
        run_dir: 运行目录
        numa_enabled: 是否启用NUMA优化
        cores_per_task: 每个任务的核心数
        parallel_tasks: 并行任务数
        bind_cpu: 是否绑定CPU
    """
    models = config.models_to_train if hasattr(config, 'models_to_train') else []
    
    # 准备任务参数（序列化配置为字典）
    tasks = []
    config_dict = config.to_dict()  # 转换为字典以便序列化
    for i, model in enumerate(models):
        task_args = (config_dict, model, i, numa_enabled, cores_per_task, bind_cpu)
        tasks.append(task_args)
    
    # 显示NUMA信息
    if numa_enabled:
        numa_nodes = get_numa_info()
        print(f"   NUMA节点数: {numa_nodes}")
        print(f"   CPU总核心数: {psutil.cpu_count(logical=True)}")
    
    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=parallel_tasks) as executor:
        # 提交所有任务
        future_to_model = {
            executor.submit(train_single_model_parallel, task): task[1]
            for task in tasks
        }
        
        # 收集结果
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
                if result['success']:
                    print(f"   ✅ {model} 训练完成")
                else:
                    print(f"   ❌ {model} 训练失败: {result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"   ❌ {model} 执行异常: {e}")
                results.append({'model': model, 'success': False, 'error': str(e)})
    
    # 汇总结果
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n📊 并行训练结果:")
    print(f"   成功: {len(successful)}/{len(models)}")
    if failed:
        print(f"   失败: {', '.join([r['model'] for r in failed])}")
    
    return results


# ========================================
#           主入口
# ========================================

def config_command(args: List[str]):
    """配置管理命令"""
    print("\n" + "="*60)
    print("AutoML Configuration Manager")
    print("="*60)
    
    # 解析子命令
    if not args or args[0] == 'list':
        # 列出所有可用配置
        manager = DynamicConfigManager()
        manager.print_config_summary()
        return 0
    
    elif args[0] == 'show':
        # 显示特定配置详情
        if len(args) < 2:
            print("❌ 请指定配置名称: config show <name>")
            return 1
        
        config_name = args[1]
        manager = DynamicConfigManager()
        config = manager.get_config(config_name)
        
        if not config:
            print(f"❌ 配置不存在: {config_name}")
            return 1
        
        print(f"\n📋 配置: {config_name}")
        print("-" * 40)
        print(f"描述: {config.description}")
        print(f"模型: {config.model.model_type}")
        print(f"特征: {config.feature.feature_type}")
        print(f"折数: {config.training.n_folds}")
        print(f"优化: {'启用' if config.optimization.enable else '禁用'}")
        
        if config.model.hyperparameters:
            print("\n超参数:")
            for k, v in config.model.hyperparameters.items():
                print(f"  {k}: {v}")
        
        return 0
    
    else:
        print(f"❌ 未知子命令: {args[0]}")
        print("可用子命令: list, show")
        return 1


def cache_command(args: List[str]):
    """缓存管理命令"""
    print("\n" + "="*60)
    print("Cache Management System")
    print("="*60)
    
    # 导入缓存管理器
    from utils.file_feature_cache import FileFeatureCache
    
    # 解析子命令
    if not args or args[0] == 'stats':
        # 显示缓存统计
        cache = FileFeatureCache()
        stats = cache.get_cache_stats()
        
        print("\n📊 缓存统计:")
        print(f"   缓存目录: {stats['cache_dir']}")
        print(f"   缓存文件数: {stats['total_files']}")
        print(f"   总大小: {stats['total_size_mb']:.2f} MB")
        print(f"   总访问次数: {stats['total_accesses']}")
        
        if stats['most_accessed']:
            print("\n🔥 最常访问:")
            for item in stats['most_accessed']:
                print(f"   - {item['file']}: {item['accesses']} 次 ({item['feature_type']})")
        
        if stats['largest_files']:
            print("\n💾 最大文件:")
            for item in stats['largest_files']:
                print(f"   - {item['file']}: {item['size_mb']:.2f} MB ({item['feature_type']})")
        
        return 0
    
    elif args[0] == 'clear':
        # 清理缓存
        cache = FileFeatureCache()
        
        # 检查是否有参数
        if len(args) > 1 and args[1].isdigit():
            days = int(args[1])
            print(f"\n🗑️  清理 {days} 天前的缓存...")
            count, size = cache.clear_cache(older_than_days=days)
        else:
            print("\n🗑️  清理所有缓存...")
            confirm = input("确认清理所有缓存? (y/n): ")
            if confirm.lower() != 'y':
                print("取消清理")
                return 0
            count, size = cache.clear_cache()
        
        print(f"✅ 已清理 {count} 个文件 ({size / 1024 / 1024:.2f} MB)")
        return 0
    
    elif args[0] == 'verify':
        # 验证缓存完整性
        cache = FileFeatureCache()
        print("\n🔍 验证缓存完整性...")
        valid, invalid = cache.verify_cache()
        print(f"   有效: {valid} 个文件")
        print(f"   无效: {invalid} 个文件")
        if invalid > 0:
            print(f"   已自动清理无效缓存")
        return 0
    
    else:
        print(f"❌ 未知子命令: {args[0]}")
        print("\n可用子命令:")
        print("  stats  - 显示缓存统计")
        print("  clear  - 清理缓存")
        print("  verify - 验证缓存完整性")
        print("\n示例:")
        print("  automl cache stats")
        print("  automl cache clear")
        print("  automl cache clear 30  # 清理30天前的缓存")
        print("  automl cache verify")
        return 1


def project_command(args: List[str]):
    """
    项目管理命令
    
    使用示例:
        automl project list                        # 列出所有项目
        automl project info project=test           # 项目详情
        automl project predict project=test data=test.csv mode=best  # 批量预测
        automl project export project=test format=zip  # 导出项目
    """
    if not args:
        print("📦 项目管理命令")
        print("\n子命令:")
        print("  list    - 列出所有项目")
        print("  info    - 显示项目信息")
        print("  predict - 使用项目模型进行批量预测")
        print("  export  - 导出项目")
        print("  report  - 生成项目报告")
        print("\n示例:")
        print("  automl project list")
        print("  automl project info project=TestPaperComparison")
        print("  automl project predict project=test data=test.csv mode=best")
        print("  automl project export project=test format=zip")
        return 0
    
    subcommand = args[0].lower()
    params = MLArgumentParser.parse_args_string(' '.join(args[1:]))
    
    # 导入项目管理器
    from utils.project_manager import ProjectManager
    from utils.project_predictor import ProjectPredictor
    
    manager = ProjectManager()
    
    if subcommand == 'list':
        # 列出所有项目
        projects = manager.list_projects()
        if projects:
            print("\n📁 项目列表:")
            for p in projects:
                print(f"\n  📦 {p['name']}")
                print(f"     路径: {p['path']}")
                print(f"     创建: {p['created']}")
                print(f"     模型: {p['models']}, 运行: {p['runs']}")
        else:
            print("❌ 未找到任何项目")
        return 0
    
    elif subcommand == 'info':
        # 显示项目信息
        project = params.get('project')
        if not project:
            print("❌ 请指定项目: project=<name>")
            return 1
        
        try:
            info = manager.get_project_info(project)
            predictor = ProjectPredictor(project, verbose=False)
            
            print(f"\n📦 项目信息: {info['project_name']}")
            print(f"   创建时间: {info.get('created_at', 'Unknown')}")
            print(f"   路径: {info['path']}")
            
            # 显示模型列表
            df = predictor.list_models()
            
            # 显示最佳模型
            if info.get('best_models'):
                print("\n🏆 最佳模型:")
                for target, best in info['best_models'].items():
                    print(f"   {target}: {best['model']} (R²={best['r2']:.4f})")
            
        except Exception as e:
            print(f"❌ 无法获取项目信息: {e}")
            return 1
        
        return 0
    
    elif subcommand == 'predict':
        # 批量预测
        project = params.get('project')
        data = params.get('data')
        mode = params.get('mode', 'all')  # all, best, ensemble
        output = params.get('output')
        
        if not project:
            print("❌ 请指定项目: project=<name>")
            return 1
        if not data:
            print("❌ 请指定数据文件: data=<file>")
            return 1
        
        try:
            predictor = ProjectPredictor(project)
            
            if mode == 'all':
                predictor.predict_all_models(
                    data_path=data,
                    output_dir=output
                )
            elif mode == 'best':
                predictor.predict_best_models(
                    data_path=data,
                    output_path=output
                )
            elif mode == 'ensemble':
                method = params.get('method', 'mean')
                predictor.predict_ensemble(
                    data_path=data,
                    output_path=output,
                    method=method
                )
            else:
                print(f"❌ 未知预测模式: {mode}")
                print("   可用模式: all, best, ensemble")
                return 1
                
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return 1
        
        return 0
    
    elif subcommand == 'export':
        # 导出项目
        project = params.get('project')
        output = params.get('output')
        format = params.get('format', 'zip')
        
        if not project:
            print("❌ 请指定项目: project=<name>")
            return 1
        
        try:
            manager.export_project(project, output, format)
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return 1
        
        return 0
    
    elif subcommand == 'report':
        # 生成项目报告
        project = params.get('project')
        output = params.get('output')
        
        if not project:
            print("❌ 请指定项目: project=<name>")
            return 1
        
        try:
            manager.generate_project_report(project, output)
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
            return 1
        
        return 0
    
    else:
        print(f"❌ 未知子命令: {subcommand}")
        return 1


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("AutoML - 自动化机器学习命令行工具")
        print("\n使用方式:")
        print("  automl <command> [options]")
        print("\n可用命令:")
        print("  train       - 训练模型")
        print("  analyze     - 分析实验结果")
        print("  predict     - 执行预测")
        print("  project     - 项目管理（批量预测）")
        print("  interactive - 🎯 交互式管理界面")
        print("  validate    - 验证配置")
        print("  config      - 管理配置模板")
        print("  cache       - 管理特征缓存")
        print("  export      - 导出模型")
        print("  warmup      - 预计算并写入特征缓存")
        print("  info        - 显示系统信息")
        print("\n示例:")
        print("  automl interactive                    # 启动交互式界面")
        print("  automl train model=xgboost data=data.csv")
        print("  automl analyze dir=runs/train format=html")
        print("  automl project list")
        print("  automl project predict project=test data=test.csv mode=best")
        print("  automl config list")
        print("  automl train config=xgboost_standard")
        print("  automl predict model=model.joblib data=test.csv")
        print("\n更多信息: automl info")
        return 0
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]
    
    # 路由到对应命令
    if command == 'train':
        return train_command(args)
    elif command == 'analyze':
        return analyze_command(args)
    elif command == 'predict':
        return predict_command(args)
    elif command == 'project':
        return project_command(args)
    elif command == 'interactive':
        # 启动交互式界面
        from interactive_cli import InteractiveCLI
        cli = InteractiveCLI()
        cli.run()
        return 0
    elif command == 'validate':
        return validate_command(args)
    elif command == 'config':
        return config_command(args)
    elif command == 'cache':
        return cache_command(args)
    elif command == 'export':
        return export_command(args)
    elif command == 'warmup':
        return warmup_command(args)
    elif command == 'info':
        return info_command(args)
    else:
        print(f"❌ 未知命令: {command}")
        print("使用 'automl info' 查看帮助")
        return 1


if __name__ == "__main__":
    sys.exit(main())

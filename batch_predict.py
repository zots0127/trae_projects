#!/usr/bin/env python3
"""
批量预测脚本
使用训练好的项目模型对新数据进行批量预测
"""

import argparse
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.project_predictor import ProjectPredictor
from utils.project_manager import ProjectManager


def main():
    parser = argparse.ArgumentParser(description='使用项目模型进行批量预测')
    parser.add_argument('project', help='项目名称或路径')
    parser.add_argument('--data', required=True, help='测试数据文件')
    parser.add_argument('--mode', default='best', 
                       choices=['all', 'best', 'ensemble'],
                       help='预测模式 (default: best)')
    parser.add_argument('--output', help='输出路径')
    parser.add_argument('--method', default='weighted',
                       choices=['mean', 'median', 'weighted'],
                       help='集成方法 (for ensemble mode)')
    parser.add_argument('--list-models', action='store_true',
                       help='列出项目中的所有模型')
    parser.add_argument('--info', action='store_true',
                       help='显示项目信息')
    
    args = parser.parse_args()
    
    # 检查项目是否存在
    project_path = Path(args.project)
    if not project_path.exists():
        print(f"❌ 项目不存在: {args.project}")
        return 1
    
    # 创建预测器
    print(f"\n📦 加载项目: {args.project}")
    predictor = ProjectPredictor(args.project, verbose=True)
    
    # 显示项目信息
    if args.info or args.list_models:
        manager = ProjectManager()
        info = manager.get_project_info(args.project)
        
        print(f"\n📊 项目信息:")
        print(f"   名称: {info['project_name']}")
        print(f"   创建时间: {info.get('created_at', 'Unknown')}")
        print(f"   模型数: {len(predictor.models)}")
        
        if args.list_models:
            print("\n📋 模型列表:")
            predictor.list_models()
        
        if info.get('best_models'):
            print("\n🏆 最佳模型:")
            for target, best in info['best_models'].items():
                print(f"   {target}: {best['model']} (R²={best['r2']:.4f})")
        
        if not args.data:
            return 0
    
    # 检查数据文件
    if not Path(args.data).exists():
        print(f"❌ 数据文件不存在: {args.data}")
        return 1
    
    # 执行预测
    print(f"\n🚀 开始预测 (模式: {args.mode})...")
    
    try:
        if args.mode == 'all':
            # 使用所有模型
            results = predictor.predict_all_models(
                data_path=args.data,
                output_dir=args.output
            )
            print(f"\n✅ 完成! 预测了 {len(results)} 个模型")
            
        elif args.mode == 'best':
            # 使用最佳模型
            result = predictor.predict_best_models(
                data_path=args.data,
                output_path=args.output
            )
            print(f"\n✅ 完成! 预测了 {len(result.columns) - len(['L1', 'L2', 'L3'])} 个目标")
            
        elif args.mode == 'ensemble':
            # 集成预测
            result = predictor.predict_ensemble(
                data_path=args.data,
                output_path=args.output,
                method=args.method
            )
            print(f"\n✅ 完成! 使用 {args.method} 方法集成预测")
        
    except Exception as e:
        print(f"\n❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
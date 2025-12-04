#!/usr/bin/env python3
"""
交互式CLI功能演示
展示主要功能的截图和说明
"""

from interactive_cli import InteractiveCLI
from utils.project_manager import ProjectManager
from utils.project_predictor import ProjectPredictor
import pandas as pd
from pathlib import Path

def demo_features():
    """演示交互式CLI的主要功能"""
    
    print("\n" + "="*80)
    print(" "*20 + "AutoML Interactive CLI Demo")
    print("="*80)
    
    # 1. 项目列表
    print("\n📋 Feature 1: Project List")
    print("-"*40)
    manager = ProjectManager()
    projects = manager.list_projects()
    if projects:
        df_projects = pd.DataFrame(projects)
        print(df_projects[['name', 'models', 'runs', 'created']].to_string(index=False))
    
    # 2. 项目信息和模型性能（包含标准差）
    print("\n\n📊 Feature 2: Project Information with Standard Deviations")
    print("-"*40)
    
    test_project = 'TestPaperComparison'
    predictor = ProjectPredictor(test_project, verbose=False)
    
    print(f"Project: {test_project}")
    print(f"Models: {len(predictor.models)}")
    
    # 显示带标准差的模型性能
    print("\nModel Performance (mean±std):")
    for i, (key, info) in enumerate(predictor.models.items()):
        if i >= 3:  # 只显示前3个
            print("  ...")
            break
        perf = info.get('performance', {})
        target = info.get('original_target', info['target'])
        r2 = perf.get('r2', 0)
        r2_std = perf.get('r2_std', 0)
        print(f"  {info['type']:8} → {target:20} R²={r2:.4f}±{r2_std:.4f}")
    
    # 3. 对比表格
    print("\n\n📈 Feature 3: Comparison Table")
    print("-"*40)
    
    table_files = list(Path(test_project).glob("comparison_table_*.csv"))
    if table_files:
        latest_table = sorted(table_files)[-1]
        df_comp = pd.read_csv(latest_table)
        
        print(f"File: {latest_table.name}")
        print("\nModel Performance Comparison:")
        
        # 显示简化版对比表
        for target in df_comp['Target'].unique()[:2]:  # 显示前2个目标
            print(f"\n  Target: {target}")
            target_df = df_comp[df_comp['Target'] == target][['Algorithm', 'R2_mean', 'R2_std']]
            for _, row in target_df.iterrows():
                print(f"    {row['Algorithm']:20} R²={row['R2_mean']:.4f}±{row['R2_std']:.4f}")
    
    # 4. 批量预测配置
    print("\n\n🚀 Feature 4: Batch Prediction Options")
    print("-"*40)
    print("Available Modes:")
    print("  1. Best Models   - Use only the best model for each target")
    print("  2. All Models    - Use all available models")
    print("  3. Ensemble      - Combine predictions from all models")
    print("\nEnsemble Methods:")
    print("  • Mean     - Simple average")
    print("  • Median   - Median value")
    print("  • Weighted - Weighted by R² scores")
    
    # 5. 项目管理功能
    print("\n\n💼 Feature 5: Project Management")
    print("-"*40)
    print("Available Operations:")
    print("  ✓ Export Project  - Package as zip/tar")
    print("  ✓ Generate Report - Create Markdown report")
    print("  ✓ Clean Project   - Remove temporary files")
    print("  ✓ Train New Models - Launch training pipeline")
    
    # 6. 交互式界面特性
    print("\n\n✨ Feature 6: Interactive Interface Features")
    print("-"*40)
    print("Rich Library Enhancements:")
    print("  • Colored output for better readability")
    print("  • Formatted tables with borders")
    print("  • Progress indicators for long operations")
    print("  • Emoji icons for visual clarity")
    print("  • Smart defaults and auto-completion")
    
    print("\n" + "="*80)
    print(" "*25 + "Demo Complete!")
    print("="*80)
    
    print("\n📝 Usage Instructions:")
    print("-"*40)
    print("1. Start Interactive CLI:")
    print("   python automl.py interactive")
    print("   OR")
    print("   python interactive_cli.py")
    print("\n2. Select a project (option 2)")
    print("3. View project info (option 3)")
    print("4. Run batch prediction (option 4)")
    print("5. Generate report (option 8)")
    
    print("\n🎯 Key Benefits:")
    print("-"*40)
    print("• No need to remember complex commands")
    print("• Guided workflow with prompts")
    print("• Visual feedback and validation")
    print("• Error handling and recovery")
    print("• Persistent session state")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    demo_features()
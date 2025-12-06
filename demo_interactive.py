#!/usr/bin/env python3
"""
Interactive CLI feature demo
Screenshots and explanations of main features
"""

from interactive_cli import InteractiveCLI
from utils.project_manager import ProjectManager
from utils.project_predictor import ProjectPredictor
import pandas as pd
from pathlib import Path

def demo_features():
    """Demonstrate the main features of the interactive CLI"""
    
    print("\n" + "="*80)
    print(" "*20 + "AutoML Interactive CLI Demo")
    print("="*80)
    
    # 1. Project list
    print("\nFeature 1: Project List")
    print("-"*40)
    manager = ProjectManager()
    projects = manager.list_projects()
    if projects:
        df_projects = pd.DataFrame(projects)
        print(df_projects[['name', 'models', 'runs', 'created']].to_string(index=False))
    
    # 2. Project info and model performance (with standard deviations)
    print("\n\nFeature 2: Project Information with Standard Deviations")
    print("-"*40)
    
    test_project = 'TestPaperComparison'
    predictor = ProjectPredictor(test_project, verbose=False)
    
    print(f"Project: {test_project}")
    print(f"Models: {len(predictor.models)}")
    
    # Show model performance with standard deviation
    print("\nModel Performance (mean+/-std):")
    for i, (key, info) in enumerate(predictor.models.items()):
        if i >= 3:  # Show only first 3
            print("  ...")
            break
        perf = info.get('performance', {})
        target = info.get('original_target', info['target'])
        r2 = perf.get('r2', 0)
        r2_std = perf.get('r2_std', 0)
        print(f"  {info['type']:8} -> {target:20} R^2={r2:.4f}+/-{r2_std:.4f}")
    
    # 3. Comparison table
    print("\n\nFeature 3: Comparison Table")
    print("-"*40)
    
    # Compatible with fixed naming and legacy timestamp naming
    table_files = list(Path(test_project).glob("comparison_table.csv"))
    if not table_files:
        table_files = list(Path(test_project).glob("comparison_table_*.csv"))
    if table_files:
        latest_table = sorted(table_files)[-1]
        df_comp = pd.read_csv(latest_table)
        
        print(f"File: {latest_table.name}")
        print("\nModel Performance Comparison:")
        
        # Show simplified comparison table
        for target in df_comp['Target'].unique()[:2]:  # Show first 2 targets
            print(f"\n  Target: {target}")
            target_df = df_comp[df_comp['Target'] == target][['Algorithm', 'R2_mean', 'R2_std']]
            for _, row in target_df.iterrows():
                print(f"    {row['Algorithm']:20} R^2={row['R2_mean']:.4f}+/-{row['R2_std']:.4f}")
    
    # 4. Batch prediction options
    print("\n\nFeature 4: Batch Prediction Options")
    print("-"*40)
    print("Available Modes:")
    print("  1. Best Models   - Use only the best model for each target")
    print("  2. All Models    - Use all available models")
    print("  3. Ensemble      - Combine predictions from all models")
    print("\nEnsemble Methods:")
    print("  - Mean     - Simple average")
    print("  - Median   - Median value")
    print("  - Weighted - Weighted by R^2 scores")
    
    # 5. Project management features
    print("\n\nFeature 5: Project Management")
    print("-"*40)
    print("Available Operations:")
    print("  - Export Project  - Package as zip/tar")
    print("  - Generate Report - Create Markdown report")
    print("  - Clean Project   - Remove temporary files")
    print("  - Train New Models - Launch training pipeline")
    
    # 6. Interactive interface features
    print("\n\nFeature 6: Interactive Interface Features")
    print("-"*40)
    print("Rich Library Enhancements:")
    print("  - Colored output for better readability")
    print("  - Formatted tables with borders")
    print("  - Progress indicators for long operations")
    print("  - Plain text output without emojis")
    print("  - Smart defaults and auto-completion")
    
    print("\n" + "="*80)
    print(" "*25 + "Demo Complete!")
    print("="*80)
    
    print("\nUsage Instructions:")
    print("-"*40)
    print("1. Start Interactive CLI:")
    print("   python automl.py interactive")
    print("   OR")
    print("   python interactive_cli.py")
    print("\n2. Select a project (option 2)")
    print("3. View project info (option 3)")
    print("4. Run batch prediction (option 4)")
    print("5. Generate report (option 8)")
    
    print("\nKey Benefits:")
    print("-"*40)
    print("- No need to remember complex commands")
    print("- Guided workflow with prompts")
    print("- Visual feedback and validation")
    print("- Error handling and recovery")
    print("- Persistent session state")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    demo_features()

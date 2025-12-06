#!/usr/bin/env python3
"""
Demonstrate using the paper_comparison config to train all models and generate comparison tables
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Paper Comparison Demo')
    parser.add_argument('--data', default='../data/Database_normalized.csv',
                       help='Training data file')
    parser.add_argument('--test-data', default=None,
                       help='Test data file (optional)')
    parser.add_argument('--project', default='PaperDemo',
                       help='Project name')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (train only 3 models)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("       Paper-level Model Comparison Demo")
    print("="*60)
    print()
    
    # Build command
    cmd_parts = [
        'python', 'automl.py', 'train',
        'config=paper_comparison',
        f'data={args.data}',
        f'project={args.project}'
    ]
    
    if args.test_data:
        cmd_parts.append(f'test_data={args.test_data}')
    
    if args.quick:
        # Quick mode: train only several key models
        cmd_parts.extend([
            'optimization.automl_models=[xgboost,catboost,lightgbm]',
            'training.n_folds=5'
        ])
        print("Quick mode: train XGBoost, CatBoost, LightGBM only")
    else:
        print("Full mode: train all 13 models")
    
    # Show command
    print("\nExecute command:")
    print(" ".join(cmd_parts))
    print()
    
    # Execute command
    import subprocess
    result = subprocess.run(cmd_parts, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\nTraining completed!")
        
        # Find results directory
        project_dir = Path(args.project)
        if project_dir.exists():
            # Find the latest training directory
            train_dirs = sorted(project_dir.glob('train*'), key=lambda x: x.stat().st_mtime)
            if train_dirs:
                latest_dir = train_dirs[-1]
                print(f"\nResults directory: {latest_dir}")
                
                # Generate comparison table
                print("\nGenerating comparison table...")
                try:
                    sys.path.append('.')
                    from utils.comparison_table import ComparisonTableGenerator
                    
                    generator = ComparisonTableGenerator(str(latest_dir))
                    exported = generator.export_all_formats()
                    
                    # Show best models
                    print("\n" + "="*60)
                    print("Best Models Summary")
                    print("="*60)
                    best_models = generator.get_best_models()
                    for target, info in best_models.items():
                        print(f"\n{target}:")
                        print(f"  Best Model: {info['algorithm']}")
                        print(f"  R^2: {info['r2']}")
                        print(f"  RMSE: {info['rmse']}")
                    
                    print("\nGenerated table files:")
                    for fmt, path in exported.items():
                        print(f"  - {fmt.upper()}: {path}")
                    
                except Exception as e:
                    print(f"Warning: error generating tables: {e}")
    else:
        print("\nTraining failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

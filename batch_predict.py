#!/usr/bin/env python3
"""
Batch prediction script
Use trained project models to batch predict on new data
"""

import argparse
import sys
from pathlib import Path

# Add current directory to import path
sys.path.insert(0, str(Path(__file__).parent))

from utils.project_predictor import ProjectPredictor
from utils.project_manager import ProjectManager


def main():
    parser = argparse.ArgumentParser(description='Batch prediction using project models')
    parser.add_argument('project', help='Project name or path')
    parser.add_argument('--data', required=True, help='Test data file')
    parser.add_argument('--mode', default='best', 
                       choices=['all', 'best', 'ensemble'],
                       help='Prediction mode (default: best)')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--method', default='weighted',
                       choices=['mean', 'median', 'weighted'],
                       help='Ensemble method (for ensemble mode)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all models in the project')
    parser.add_argument('--info', action='store_true',
                       help='Show project information')
    
    args = parser.parse_args()
    
    # Check whether project exists
    project_path = Path(args.project)
    if not project_path.exists():
        print(f"Project not found: {args.project}")
        return 1
    
    # Create predictor
    print(f"\nLoading project: {args.project}")
    predictor = ProjectPredictor(args.project, verbose=True)
    
    # Show project info
    if args.info or args.list_models:
        manager = ProjectManager()
        info = manager.get_project_info(args.project)
        
        print(f"\nProject Information:")
        print(f"   Name: {info['project_name']}")
        print(f"   Created: {info.get('created_at', 'Unknown')}")
        print(f"   Models: {len(predictor.models)}")
        
        if args.list_models:
            print("\nModel List:")
            predictor.list_models()
        
        if info.get('best_models'):
            print("\nBest Models:")
            for target, best in info['best_models'].items():
                print(f"   {target}: {best['model']} (R^2={best['r2']:.4f})")
        
        if not args.data:
            return 0
    
    # Check data file
    if not Path(args.data).exists():
        print(f"Data file not found: {args.data}")
        return 1
    
    # Execute prediction
    print(f"\nStart predicting (mode: {args.mode})...")
    
    try:
        if args.mode == 'all':
            # Use all models
            results = predictor.predict_all_models(
                data_path=args.data,
                output_dir=args.output
            )
            print(f"\nCompleted! Predicted {len(results)} models")
            
        elif args.mode == 'best':
            # Use best models
            result = predictor.predict_best_models(
                data_path=args.data,
                output_path=args.output
            )
            print(f"\nCompleted! Predicted {len(result.columns) - len(['L1', 'L2', 'L3'])} targets")
            
        elif args.mode == 'ensemble':
            # Ensemble prediction
            result = predictor.predict_ensemble(
                data_path=args.data,
                output_path=args.output,
                method=args.method
            )
            print(f"\nCompleted! Ensemble prediction using {args.method} method")
        
    except Exception as e:
        print(f"\nPrediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

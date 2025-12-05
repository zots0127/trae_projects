#!/usr/bin/env python3
"""
Unify n_folds setting to 10 in all configuration files
"""

import os
import yaml
from pathlib import Path

def fix_yaml_file(file_path):
    """Fix n_folds setting in a single YAML file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace all n_folds settings to 10
        import re
        modified = False
        
        # Match n_folds: number
        pattern = r'(n_folds:\s*)(\d+)'
        
        def replace_func(match):
            if match.group(2) != '10':
                nonlocal modified
                modified = True
                return match.group(1) + '10'
            return match.group(0)
        
        new_content = re.sub(pattern, replace_func, content)
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"INFO: Fixed: {file_path.relative_to(Path.cwd())}")
            return True
    except Exception as e:
        print(f"ERROR: Failed to process {file_path}: {e}")
    return False

def main():
    """Main function"""
    config_dir = Path(__file__).parent.parent / 'config'
    
    fixed_count = 0
    total_count = 0
    
    for yaml_file in config_dir.glob('**/*.yaml'):
        if '__pycache__' not in str(yaml_file):
            total_count += 1
            if fix_yaml_file(yaml_file):
                fixed_count += 1
    
    print(f"\nINFO: Processing completed: Checked {total_count} files, fixed {fixed_count} files")
    print("INFO: All configuration files have n_folds unified to 10")

if __name__ == "__main__":
    main()

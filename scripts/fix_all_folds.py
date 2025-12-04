#!/usr/bin/env python3
"""
将所有配置文件的n_folds统一设置为10
"""

import os
import yaml
from pathlib import Path

def fix_yaml_file(file_path):
    """修复单个YAML文件的n_folds设置"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换所有n_folds设置为10
        import re
        modified = False
        
        # 匹配 n_folds: 数字
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
            print(f"✅ 修复: {file_path.relative_to(Path.cwd())}")
            return True
    except Exception as e:
        print(f"❌ 错误处理 {file_path}: {e}")
    return False

def main():
    """主函数"""
    config_dir = Path(__file__).parent.parent / 'config'
    
    fixed_count = 0
    total_count = 0
    
    for yaml_file in config_dir.glob('**/*.yaml'):
        if '__pycache__' not in str(yaml_file):
            total_count += 1
            if fix_yaml_file(yaml_file):
                fixed_count += 1
    
    print(f"\n📊 处理完成：检查了 {total_count} 个文件，修复了 {fixed_count} 个文件")
    print("✅ 所有配置文件的 n_folds 已统一设置为 10")

if __name__ == "__main__":
    main()
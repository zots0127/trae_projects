#!/bin/bash

# 打包当前目录 - 忽略隐藏文件和不必要的文件

echo "==========================================="
echo "打包当前目录"
echo "==========================================="

# 设置版本和包名
VERSION="v1.0.0"
PACKAGE_NAME="ir0921_${VERSION}_$(date +%Y%m%d).tar.gz"

echo ""
echo "准备打包..."

# 创建排除文件列表
cat > .tar_exclude << EOF
.DS_Store
.git
.venv
.claude
.pytest_cache
__pycache__
*.pyc
*.pyo
*.log
*.tar.gz
.tar_exclude
feature_cache/*
file_feature_cache/*
catboost_info
*.ipynb_checkpoints
MLP_Test_*
Paper_*
demos
docs
package*.sh
test_*.py
test_*.sh
monitor_*.sh
manage.py
demo*.py
demo*.ipynb
usage*.md
EOF

echo "创建压缩包: $PACKAGE_NAME"
echo "忽略隐藏目录和测试文件..."

# 使用 tar 打包，排除指定文件
tar -czf $PACKAGE_NAME --exclude-from=.tar_exclude .

# 清理临时文件
rm -f .tar_exclude

echo ""
echo "✅ 打包完成: $PACKAGE_NAME"
echo "文件大小: $(du -h $PACKAGE_NAME | cut -f1)"

# 显示包内容摘要
echo ""
echo "包内容摘要:"
tar -tzf $PACKAGE_NAME | head -20
echo "..."
echo "总文件数: $(tar -tzf $PACKAGE_NAME | wc -l)"

echo ""
echo "使用方法:"
echo "1. 上传到服务器: scp $PACKAGE_NAME user@server:/path/"
echo "2. 在服务器解压: tar -xzf $PACKAGE_NAME"
echo "3. 运行部署脚本: ./uv_deploy.sh"
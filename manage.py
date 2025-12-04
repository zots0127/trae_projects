#!/usr/bin/env python3
"""
AutoML管理界面快速启动脚本
"""

import sys
from interactive_cli import InteractiveCLI

def main():
    """主函数"""
    print("🚀 Starting AutoML Interactive Manager...")
    cli = InteractiveCLI()
    
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
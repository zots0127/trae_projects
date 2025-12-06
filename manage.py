#!/usr/bin/env python3
"""
AutoML management interface quick start script
"""

import sys
from interactive_cli import InteractiveCLI

def main():
    """Main function"""
    print("Starting AutoML Interactive Manager...")
    cli = InteractiveCLI()
    
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

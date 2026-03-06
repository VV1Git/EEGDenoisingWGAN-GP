#!/usr/bin/env python3
"""
Quick launcher for EEG Denoising Demo
Simply double-click this file or run: python run_demo.py
"""

if __name__ == "__main__":
    import subprocess
    import sys
    
    print("Launching EEG Denoising Demo...")
    print("=" * 70)
    
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\nDemo closed.")
    except Exception as e:
        print(f"\nError launching demo: {e}")
        input("Press Enter to exit...")

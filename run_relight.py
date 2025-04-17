#!/usr/bin/env python
"""
Run Relight

This script provides a simple way to run the Relight project from the command line.
"""

import os
import sys
import subprocess
from pathlib import Path

# Get the path to Blender
def get_blender_path():
    """Get the path to Blender executable."""
    # Check if BLENDER_PATH environment variable is set
    blender_path = os.environ.get("BLENDER_PATH")
    if blender_path:
        return blender_path
    
    # Try to find Blender in common locations
    if sys.platform == "win32":
        # Windows
        common_paths = [
            r"C:\Program Files\Blender Foundation\Blender\blender.exe",
            r"C:\Program Files (x86)\Blender Foundation\Blender\blender.exe",
        ]
    elif sys.platform == "darwin":
        # macOS
        common_paths = [
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ]
    else:
        # Linux
        common_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
        ]
    
    # Check if Blender is in the PATH
    try:
        subprocess.run(["blender", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "blender"
    except FileNotFoundError:
        pass
    
    # Check common paths
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # If Blender is not found, ask the user
    print("Blender not found. Please enter the path to the Blender executable:")
    blender_path = input().strip()
    if os.path.exists(blender_path):
        return blender_path
    else:
        print(f"Error: {blender_path} does not exist.")
        sys.exit(1)


def main():
    """Main function."""
    # Get the path to Blender
    blender_path = get_blender_path()
    
    # Get the path to the main script
    script_path = Path(__file__).parent / "relight" / "main.py"
    
    # Run Blender with the main script and pass all arguments
    cmd = [blender_path, "--background", "--python", str(script_path), "--"] + sys.argv[1:]
    subprocess.run(cmd)


if __name__ == "__main__":
    main() 
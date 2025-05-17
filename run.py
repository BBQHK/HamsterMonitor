#!/usr/bin/env python3
"""Launcher script for the Hamster Monitor system."""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the main application
from src.main import main

if __name__ == '__main__':
    main() 
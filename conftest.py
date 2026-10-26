"""Make the project root importable in tests (src, app, service)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

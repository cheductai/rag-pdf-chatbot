"""Init file for tests package."""

# Test configuration
import os
import sys
from pathlib import Path

# Add src to Python path for testing
test_dir = Path(__file__).parent
project_root = test_dir.parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

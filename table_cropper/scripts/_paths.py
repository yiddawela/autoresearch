"""Path setup for scripts that import from sibling directories.

Add project root, scripts/, and src/ to sys.path so cross-imports work
regardless of working directory.

Usage (at the top of any script):
    import _paths  # noqa: F401
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_SRC_DIR = _PROJECT_ROOT / "src"

for p in (_PROJECT_ROOT, _SCRIPTS_DIR, _SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

"""Shared pytest configuration for the duxxdb Python tests.

Makes the in-tree ``python/`` source importable without needing to
build the maturin wheel first. The native PyO3 module (``_native``)
is only imported by the embedded primitives — the server facade
under test here is pure Python and does not depend on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add bindings/python/python/ to sys.path so ``import duxxdb.server``
# works against the source tree.
_HERE = Path(__file__).resolve().parent
_SOURCE_ROOT = _HERE.parent / "python"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

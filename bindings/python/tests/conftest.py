"""Shared pytest configuration for the duxxdb Python tests.

This conftest exists so the pure-Python server-facade tests can run
without a built wheel: it makes the in-tree ``python/`` source
importable from the test files.

It must NOT shadow an installed wheel — wheel tests
(``test_prompt_registry.py``) need the native ``_native`` PyO3
module which only lives inside the installed wheel. The source tree
contains an ``__init__.py`` that exposes ``_missing_native`` stubs
for that case, and those stubs raise ImportError on any attempt to
construct a ``PromptRegistry`` / ``MemoryStore`` etc.

Resolution: only add the source path to ``sys.path`` when the
installed ``duxxdb`` package is absent (i.e. nothing has been
pip-installed). When a wheel IS installed, leave the import path
alone so the wheel's compiled ``_native`` is found.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# If a wheel-installed ``duxxdb`` is already importable, do NOT add
# the source tree to sys.path — the source ``__init__.py`` would
# shadow the wheel and hide the native ``_native`` module.
_installed = importlib.util.find_spec("duxxdb")
if _installed is None or "site-packages" not in (_installed.origin or ""):
    _HERE = Path(__file__).resolve().parent
    _SOURCE_ROOT = _HERE.parent / "python"
    if str(_SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(_SOURCE_ROOT))

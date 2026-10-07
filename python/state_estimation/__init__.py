"""state_estimation — probabilistic state estimation and SLAM for mobile robots.

The algorithms live in C++ (``cpp/``) and are exposed here through the compiled ``_core`` module.
``state_estimation.plotting`` holds small matplotlib helpers used by the notebooks.
"""

from ._core import *  # noqa: F401,F403
from . import _core

__all__ = [name for name in dir(_core) if not name.startswith("_")]
__version__ = "0.1.0"

"""pytest 配置：把项目根加入 sys.path，便于直接 `import remap_labels` 等。"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

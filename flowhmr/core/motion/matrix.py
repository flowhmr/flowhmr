# Compatibility shim: re-exports ..math.matrix so existing imports of core.motion.matrix keep working.
from ..math.matrix import * # noqa: F401,F403
from ..math import matrix as _matrix

# Ensure all attributes are accessible via `matrix.xxx`
import sys as _sys
_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _matrix.__dict__.items() if not k.startswith("_")}
)

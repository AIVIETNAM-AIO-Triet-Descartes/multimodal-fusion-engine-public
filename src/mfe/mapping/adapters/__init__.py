from .base import get_adapter, register
from .by_frame import ByFrameAdapter
from .by_frame_range import ByFrameRangeAdapter
from .by_timestamp import ByTimestampAdapter
from .raw_keyframe import RawKeyframeAdapter

# Importing this package registers every adapter (side effect below), so a
# caller only needs `import mfe.mapping.adapters` to populate the registry.
__all__ = ["get_adapter", "register"]

register(ByFrameAdapter)
register(ByFrameRangeAdapter)
register(ByTimestampAdapter)
register(RawKeyframeAdapter)

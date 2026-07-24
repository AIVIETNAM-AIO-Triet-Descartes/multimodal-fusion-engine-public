from .base import get_adapter, register
from .by_frame import ByFrameAdapter
from .by_frame_range import ByFrameRangeAdapter
from .by_timestamp import ByTimestampAdapter
from .raw_keyframe import RawKeyframeAdapter

register(ByFrameAdapter)
register(ByFrameRangeAdapter)
register(ByTimestampAdapter)
register(RawKeyframeAdapter)

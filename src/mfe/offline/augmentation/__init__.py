"""Audit-gated augmentation of the immutable BTC keyframe bundle.

The package deliberately owns its pilot-only schemas.  Shared contracts remain
unchanged until an augmentation policy has passed DEV and held-out gates.
"""

from mfe.offline.augmentation.config import AugmentationConfig, load_augmentation_config
from mfe.offline.augmentation.models import (
    AugmentationManifestRow,
    BtcFrame,
    CandidateRow,
    CoverageAuditRow,
    CoverageClass,
    OracleTargetRow,
    Selector,
    ShotProposal,
)

__all__ = [
    "AugmentationConfig",
    "AugmentationManifestRow",
    "BtcFrame",
    "CandidateRow",
    "CoverageAuditRow",
    "CoverageClass",
    "OracleTargetRow",
    "Selector",
    "ShotProposal",
    "load_augmentation_config",
]

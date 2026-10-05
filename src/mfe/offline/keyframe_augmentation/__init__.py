"""BTC-first, audit-gated keyframe augmentation.

The package deliberately keeps research artifacts separate from the immutable
BTC bundle.  Public functions are re-exported from the focused gate modules;
the CLI in :mod:`mfe.offline.keyframe_augmentation.cli` is the supported entry
point for end-to-end runs.
"""

from mfe.offline.keyframe_augmentation.diagnostics import (
    compare_feature_parity,
    crowding_report,
)
from mfe.offline.keyframe_augmentation.models import (
    AugmentationConfig,
    BtcFrame,
    CandidateFrame,
    CoverageClass,
    ShotProposal,
)
from mfe.offline.keyframe_augmentation.oracle import (
    OracleEvidenceRow,
    blocked_oracle_row,
    validate_oracle_rows,
)

__all__ = (
    "AugmentationConfig",
    "BtcFrame",
    "CandidateFrame",
    "CoverageClass",
    "ShotProposal",
    "OracleEvidenceRow",
    "blocked_oracle_row",
    "compare_feature_parity",
    "crowding_report",
    "validate_oracle_rows",
)

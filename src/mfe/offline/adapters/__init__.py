"""Boundary adapters — translate a foreign artifact's identifiers into ours.

Everything here runs **once, at load time**. Past this package there is exactly
one spelling of `keyframe_id` in the codebase (`mfe.contracts.ids`). An adapter
that leaks into the middle of the pipeline has failed at its job.
"""

from mfe.offline.adapters.sota_ids import SotaIdMap, parse_sota_id

__all__ = ["SotaIdMap", "parse_sota_id"]

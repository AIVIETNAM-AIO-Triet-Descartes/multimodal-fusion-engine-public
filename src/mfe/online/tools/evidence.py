"""Evidence ID generation helper — docs/00 §5, docs/04 §2.

Format: "{tool_name}:{run_id}:{ordinal}"
Example: "search_text:run01:0001"
"""

from __future__ import annotations

from mfe.contracts.tools import EvidenceId


def create_evidence_id(tool_name: str, run_id: str, ordinal: int) -> EvidenceId:
    """Generate a structured evidence_id string.

    Args:
        tool_name: Name of the tool emitting the candidate (e.g. 'search_text').
        run_id: Execution/query run identifier (e.g. 'run01').
        ordinal: Zero-based or 1-based index of the hit.

    Returns:
        Structured EvidenceId string.
    """
    return f"{tool_name}:{run_id}:{ordinal:04d}"

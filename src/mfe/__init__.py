"""Multimodal Fusion Engine — video retrieval for AIC 2026.

Package map (one directory = one owner, see docs/00 §2):

    contracts/  [Vỹ Triết] the code form of docs/00. Changing it is a contract change.
    core/       [Vỹ Triết] config, storage backend, logging.
    offline/    [Quốc Đạt]  ingestion: video -> 4 stores.           docs/02
    stores/     [Công Quang]  vector / inverted / asset search stores. docs/03
    online/     [Phạm Trọng]  cascade + tool implementations.          docs/04
    browser/    [Phạm Trọng]  minimal grid + timeline UI.              docs/04 §4
    mapping/    [Nhật Bảo]  relational map, validator, submit (P0).  docs/05
    eval/       [Vỹ Triết] benchmark harness.                       docs/01 §2
    agent/      [later] Buổi 3 planner. Reuses online/tools, does not replace it.
"""

__version__ = "0.1.0"

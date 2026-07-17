"""Multimodal Fusion Engine — video retrieval for AIC 2026.

Package map (one directory = one owner, see docs/00 §2):

    contracts/  [Lead] the code form of docs/00. Changing it is a contract change.
    core/       [Lead] config, storage backend, logging.
    offline/    [TV1]  ingestion: video -> 4 stores.           docs/02
    stores/     [TV2]  vector / inverted / asset search stores. docs/03
    online/     [TV3]  cascade + tool implementations.          docs/04
    browser/    [TV3]  minimal grid + timeline UI.              docs/04 §4
    mapping/    [TV4]  relational map, validator, submit (P0).  docs/05
    eval/       [Lead] benchmark harness.                       docs/01 §2
    agent/      [later] Buổi 3 planner. Reuses online/tools, does not replace it.
"""

__version__ = "0.1.0"

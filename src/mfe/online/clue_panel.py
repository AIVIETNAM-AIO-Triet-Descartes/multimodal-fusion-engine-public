"""Operator-entered oracle clues, scoped to the team or authenticated user.

This is a notepad for an external oracle conversation, not an oracle/chat API.
State lasts for the lifetime of the server; clients reload the current revision.
"""
from collections import deque
from copy import deepcopy
from threading import RLock
import hashlib
import json


class ClueConflict(ValueError):
    pass


class CluePanels:
    def __init__(self):
        self._lock = RLock()
        self._panels = {}

    def snapshot(self, scope, owner):
        with self._lock:
            panel = self._panels.setdefault(self._key(scope, owner),
                {"revision": 0, "task": "", "clues": [], "history": deque(maxlen=256)})
            return self._public(panel)

    @staticmethod
    def _key(scope, owner):
        if scope not in {"shared", "private"} or not owner:
            raise ValueError("Authenticated scope required")
        return scope, owner if scope == "private" else "team"

    @staticmethod
    def _public(panel):
        return {"revision": panel["revision"], "task": panel["task"],
                "clues": deepcopy(panel["clues"]),
                "query": "\n".join(c["text"] for c in panel["clues"] if c["enabled"]),
                "history": deepcopy(list(panel["history"]))}

    def mutate(self, scope, owner, payload):
        with self._lock:
            old = self._panels.setdefault(self._key(scope, owner),
                {"revision": 0, "task": "", "clues": [], "history": deque(maxlen=256)})
            expected = payload.get("revision")
            if type(expected) is not int or expected != old["revision"]:
                raise ClueConflict("Clues changed; reload the current revision")
            new = deepcopy(old)
            op = payload.get("op")
            if op == "reset":
                task = payload.get("task", "")
                if not isinstance(task, str) or len(task) > 200:
                    raise ValueError("Task label must be at most 200 characters")
                new["task"], new["clues"] = task, []
            elif op == "append":
                new["clues"].append({"id": old["revision"] + 1,
                    "text": payload.get("text"), "enabled": True})
            elif op in {"edit", "delete", "toggle"}:
                if type(payload.get("id")) is not int:
                    raise ValueError("Integer clue id required")
                clue = next((c for c in new["clues"] if c["id"] == payload["id"]), None)
                if clue is None:
                    raise ValueError("Clue does not exist")
                if op == "delete":
                    new["clues"].remove(clue)
                elif op == "edit":
                    clue["text"] = payload.get("text")
                else:
                    if type(payload.get("enabled")) is not bool:
                        raise ValueError("enabled must be boolean")
                    clue["enabled"] = payload["enabled"]
            else:
                raise ValueError("Unknown clue operation")
            if len(new["clues"]) > 32:
                raise ValueError("At most 32 clues")
            for clue in new["clues"]:
                if not isinstance(clue["text"], str) or not clue["text"].strip():
                    raise ValueError("A clue needs nonempty text")
                clue["text"] = clue["text"].strip()
            if sum(len(c["text"]) + 1 for c in new["clues"]) > 4000:
                raise ValueError("Combined clues exceed 4000 characters")
            new["revision"] += 1
            digest = hashlib.sha256(json.dumps({k: new[k] for k in
                ("revision", "task", "clues")}, sort_keys=True).encode()).hexdigest()
            new["history"].append({"revision": new["revision"], "op": op,
                "state_sha256": digest})
            self._panels[self._key(scope, owner)] = new
            return self._public(new)

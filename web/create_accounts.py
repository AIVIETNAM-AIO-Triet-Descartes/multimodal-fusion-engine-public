#!/usr/bin/env python3
"""Create fixed web accounts; there is intentionally no registration route."""

from __future__ import annotations

import hashlib
import json
import secrets
import os
from pathlib import Path

ROOT = Path(os.getenv("MFE_WEB_ROOT", Path(__file__).resolve().parent))
ACCOUNTS = ROOT / "accounts.json"
CREDENTIALS = ROOT / "CREDENTIALS.txt"
SESSION_SECRET = ROOT / ".session_secret"


def hash_password(password: str) -> dict[str, object]:
    salt = secrets.token_bytes(16)
    n, r, p = 2**14, 8, 1
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p)
    return {"salt": salt.hex(), "hash": digest.hex(), "n": n, "r": r, "p": p}


def main() -> None:
    if ACCOUNTS.exists():
        raise SystemExit(f"{ACCOUNTS} da ton tai; xoa file nay neu muon tao lai")
    users: dict[str, dict[str, object]] = {}
    credentials: list[str] = [
        "MFE web credentials — keep this file mode 600",
        "",
    ]
    for username, role in (("admin", "admin"), ("operator", "operator"), ("viewer", "viewer")):
        password = secrets.token_urlsafe(15)
        users[username] = {"role": role, **hash_password(password)}
        credentials.append(f"{username}\t{password}\trole={role}")
    ACCOUNTS.write_text(
        json.dumps({"version": 1, "users": users}, indent=2) + "\n",
        encoding="utf-8",
    )
    ACCOUNTS.chmod(0o600)
    CREDENTIALS.write_text("\n".join(credentials) + "\n", encoding="utf-8")
    CREDENTIALS.chmod(0o600)
    if not SESSION_SECRET.exists():
        SESSION_SECRET.write_text(secrets.token_urlsafe(48) + "\n", encoding="utf-8")
        SESSION_SECRET.chmod(0o600)
    print(f"created {len(users)} fixed accounts")
    print(f"credentials: {CREDENTIALS}")


if __name__ == "__main__":
    main()

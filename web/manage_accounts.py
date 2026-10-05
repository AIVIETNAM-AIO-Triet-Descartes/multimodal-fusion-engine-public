#!/usr/bin/env python3
"""Server-side account management for the fixed-login web application.

There is deliberately no public registration endpoint.  Run this command over
SSH on the server when an operator needs another account.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import tempfile
from pathlib import Path

from create_accounts import ACCOUNTS, CREDENTIALS, hash_password

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
ROLES = {"admin", "operator", "viewer"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Them tai khoan MFE, khong co dang ky cong khai")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="tao mot tai khoan moi")
    add.add_argument("username")
    add.add_argument("--role", choices=sorted(ROLES), default="viewer")
    args = parser.parse_args()

    if not USERNAME_RE.fullmatch(args.username):
        raise SystemExit("username chi duoc gom chu, so, dau cham, gach ngang, gach duoi")
    if not ACCOUNTS.is_file():
        raise SystemExit(f"chua co {ACCOUNTS}; chay create_accounts.py truoc")

    payload = json.loads(ACCOUNTS.read_text(encoding="utf-8"))
    users = payload.setdefault("users", {})
    if args.username in users:
        raise SystemExit(f"tai khoan {args.username!r} da ton tai")

    password = secrets.token_urlsafe(15)
    users[args.username] = {"role": args.role, **hash_password(password)}

    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=ACCOUNTS.parent, delete=False, suffix=".tmp"
    )
    try:
        with handle as out:
            json.dump(payload, out, indent=2)
            out.write("\n")
        os.chmod(handle.name, 0o600)
        os.replace(handle.name, ACCOUNTS)
    finally:
        Path(handle.name).unlink(missing_ok=True)

    with CREDENTIALS.open("a", encoding="utf-8") as out:
        out.write(f"{args.username}\t{password}\trole={args.role}\n")
    CREDENTIALS.chmod(0o600)
    print(f"created username={args.username} role={args.role}")
    print(f"password={password}")
    print(f"saved credentials in {CREDENTIALS}")


if __name__ == "__main__":
    main()

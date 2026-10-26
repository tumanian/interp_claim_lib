"""claimci command line."""
from __future__ import annotations

import argparse
import sys

from .loader import load_repo
from .validate import validate


def cmd_validate(args) -> int:
    repo = load_repo(args.root)
    errors = validate(repo)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if errors:
        return 1
    print(f"ok: {len(repo.claims)} claims, {len(repo.questions)} questions")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="claimci")
    p.add_argument("--root", default=".", help="repo root containing claims/ and questions/")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="check claims and questions; no model loads")
    args = p.parse_args(argv)
    return {"validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

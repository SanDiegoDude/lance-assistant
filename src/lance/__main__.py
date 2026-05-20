"""`python -m lance ...` dispatcher."""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m lance {download,verify} [args...]", file=sys.stderr)
        return 2
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "download":
        from . import download

        return download.main(rest)
    if cmd == "verify":
        from . import verify

        return verify.main(rest)
    if cmd == "extract_understanding":
        from . import extract_understanding

        return extract_understanding.main(rest)
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

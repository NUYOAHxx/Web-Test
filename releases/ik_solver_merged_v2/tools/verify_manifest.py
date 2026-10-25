#!/usr/bin/env python3
"""Verify the delivered source package hashes; does not import the IK project."""
import hashlib
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    failures=[]
    rows=(root/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for row in rows:
        digest,name=row.split("  ",1)
        path=root/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual!=digest:
            failures.append(name)
    if failures:
        print("FAILED:",", ".join(failures))
        return 1
    print(f"Verified {len(rows)} file hashes.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())

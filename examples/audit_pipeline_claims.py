"""Audit `docs/pipeline-feasibility.md` references to local report artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OUT_PATH_RE = re.compile(r"out/[A-Za-z0-9._/@+-]+")


def extract_out_paths(text: str) -> list[str]:
    paths = []
    for match in OUT_PATH_RE.finditer(text):
        path = match.group(0).rstrip(").,;:`")
        if path not in paths:
            paths.append(path)
    return paths


def audit_document(doc: Path, root: Path) -> dict:
    text = doc.read_text()
    paths = extract_out_paths(text)
    missing = [path for path in paths if not (root / path).exists()]
    return {
        "document": str(doc),
        "referenced_out_paths": paths,
        "referenced_out_path_count": len(paths),
        "missing_out_paths": missing,
        "ok": not missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doc", default="docs/pipeline-feasibility.md")
    parser.add_argument("--out", default="out/pipeline-feasibility/audit.json")
    args = parser.parse_args()

    root = Path.cwd()
    report = audit_document(root / args.doc, root)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

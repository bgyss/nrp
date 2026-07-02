"""Download an official Mitsuba 3 gallery scene into examples/scenes/<name>/.

Scene assets are never vendored into this repository (see .gitignore next to this
script) — this downloads them on demand from the Mitsuba 3 gallery
(https://mitsuba.readthedocs.io/en/stable/src/gallery.html), most of which originate
from Benedikt Bitterli's rendering resources. Check the LICENSE.txt inside each
downloaded scene folder for its terms.

Usage:
  uv run python examples/scenes/download_scene.py kitchen
  uv run python examples/scenes/download_scene.py --list
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://d38rqfq1h7iukm.cloudfront.net/scenes"
SCENES = [
    "cornell-box",
    "kitchen",
    "bedroom",
    "living-room",
    "living-room-2",
    "living-room-3",
    "veach-bidir",
    "veach-mis",
    "veach-ajar",
]


def download(name: str, dest_root: Path) -> Path:
    if name not in SCENES:
        raise SystemExit(f"unknown scene {name!r}; known: {', '.join(SCENES)}")
    dest = dest_root / name
    scene_xml = dest / "scene.xml"
    if scene_xml.exists():
        print(f"{scene_xml} already present, skipping download")
        return scene_xml
    url = f"{BASE_URL}/{name}.zip"
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    print(f"unpacking {len(data) / 1e6:.1f} MB into {dest} ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            # Zips wrap everything in a top-level <name>/ folder; strip it.
            rel = Path(member)
            if rel.parts and rel.parts[0] == name:
                rel = Path(*rel.parts[1:])
            if not rel.parts or member.endswith("/"):
                continue
            target = dest / rel
            if not target.resolve().is_relative_to(dest.resolve()):
                raise SystemExit(f"zip member escapes destination: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))
    if not scene_xml.exists():
        raise SystemExit(f"downloaded archive contained no scene.xml under {dest}")
    print(f"done: {scene_xml}")
    return scene_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", nargs="?", help=f"one of: {', '.join(SCENES)}")
    parser.add_argument("--list", action="store_true", help="list known scenes")
    args = parser.parse_args()
    if args.list or not args.scene:
        print("\n".join(SCENES))
        sys.exit(0)
    download(args.scene, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()

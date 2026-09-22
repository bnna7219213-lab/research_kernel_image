#!/usr/bin/env python3
"""
T2Image — dataset fingerprint tool.

Produces a reproducible SHA-256 hash over:
  - sorted list of image paths (path + size + mtime)
  - caption CSV contents (if present)

Used as training cache key — staleness triggers re-ingestion.

Usage:
  python compute_data_fingerprint.py \
      --image_dir /data/t2image/raw/laion_aesthetics \
      --caption_csv /data/t2image/raw/laion_aesthetics/metadata.csv \
      --out /data/t2image/_spec/laion_aesthetics.fingerprint.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_images(root: Path) -> list[dict]:
    entries = [{"path": str(p.relative_to(root)),
                "size": p.stat().st_size,
                "mtime": int(p.stat().st_mtime)}
               for p in root.rglob("*")
               if p.suffix.lower() in IMG_EXTS]
    entries.sort(key=lambda e: e["path"])
    return entries


def hash_csv(csv_path: Optional[Path]) -> str:
    if csv_path is None or not csv_path.exists():
        return sha256("")
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--caption_csv", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", default="t2image-dataset")
    args = parser.parse_args()

    if not args.image_dir.exists():
        print(f"[ERROR] image_dir not found: {args.image_dir}", file=sys.stderr)
        return 1

    entries = scan_images(args.image_dir)
    manifest_str = json.dumps(entries, sort_keys=True).encode("utf-8")
    man_hash = hashlib.sha256(manifest_str).hexdigest()
    csv_hash = hash_csv(args.caption_csv)
    combined = sha256(man_hash + csv_hash)

    fp = {
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_dir": str(args.image_dir.resolve()),
        "caption_csv": str(args.caption_csv.resolve()) if args.caption_csv else None,
        "num_files": len(entries),
        "total_gb": round(sum(e["size"] for e in entries) / (1024**3), 3),
        "manifest_sha256": man_hash,
        "csv_sha256": csv_hash,
        "dataset_fingerprint": combined,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fp, f, indent=2, ensure_ascii=False)
    print(json.dumps(fp, indent=2, ensure_ascii=False))
    print(f"\n[FINGERPRINT] {combined}")
    print(f"[OUT] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

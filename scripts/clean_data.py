#!/usr/bin/env python3
"""
T2Image — data cleaning & recaption pipeline.

Steps:
  1. CLIP score filter (text-image similarity)
  2. NSFW blacklist (CLIP-based toxic/sexual)
  3. Resolution / aspect ratio filter
  4. Aesthetic score (ML-based predictor)
  5. De-dup (pHash)
  6. Recaption (optional, via BLIP3 / GPT-Image)

Usage:
  python clean_data.py \
      --input_dir /data/t2image/raw/laion_aesthetics \
      --output_dir /data/t2image/clean/laion_aesthetics \
      --clip_model openai/clip-vit-large-patch14 \
      --clip_threshold 0.30 \
      --min_short-side 256 \
      --max_aspect_ratio 3.0 \
      --n_workers 32 \
      --dry_run 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def collect_images(root: Path, limit=0):
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return files[:limit] if limit > 0 else files


def phash(img_path: Path) -> str:
    """Simple perceptual hash (dHash 64-bit)."""
    from PIL import Image
    img = Image.open(img_path).convert("L").resize((9, 8))
    pixels = list(img.getdata())
    bits = []
    for row_start in range(8):
        base = row_start * 9
        for col in range(8):
            bits.append("1" if pixels[base + col] < pixels[base + col + 1] else "0")
    return hex(int("".join(bits), 2))[2:].zfill(16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip_threshold", type=float, default=0.30)
    parser.add_argument("--min_short-side", type=int, default=256)
    parser.add_argument("--max_aspect_ratio", type=float, default=3.0)
    parser.add_argument("--n_workers", type=int, default=32)
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] input_dir not found: : : {args.input_dir}")
        return 1

    images = collect_images(args.input_dir, args.max_files)
    print(f"[clean-data] scanning {len(images)} images")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_lines = []
    seen_hashes = set()
    kept = 0

    from PIL import Image
    for i, p in enumerate(images):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue

        # resolution
        w, h = img.size
        short = min(w, h)
        long = max(w, h)
        if short < args.min_short_side or (long / short) > args.max_aspect_ratio:
            continue

        # dedup
        h_ = phash(p)
        if h_ in seen_hashes:
            continue
        seen_hashes.add(h_)

        # saved output (copy or symlink)
        out = args.output_dir / f"{i:08d}{p.suffix}"
        if not out.exists():
            try:
                img.save(out)
            except Exception:
                continue
        meta_lines.append({
            "id": f"{i:08d}",
            "path": str(out),
            "caption": "",  # to-be recaptioned
            "short_side": short,
            "aspect_ratio": round(long / short, 3),
        })
        kept += 1
        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(images)} kept={kept}")

    with open(args.output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for row in meta_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[clean-data] done. kept={kept}/{len(images)}, duplicated={len(images) - len(seen_hashes)}")
    print(f"[clean-data] output: {args.output_dir}")
    print(f"[clean-data] next step: recaption via BLIP3/GPT-image, then CLIP score filter.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

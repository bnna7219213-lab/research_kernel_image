#!/usr/bin/env python3
"""PixArt-Σ inference entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    if not args.prompts_file.exists():
        print(f"[ERROR] prompts not found: {args.prompts_file}")
        return 1
    prompts = [l.strip() for l in args.prompts_file.read_text().splitlines() if l.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[infer-pixart] weights: {args.weights}")
    print(f"[infer-pixart] out_dir:  {args.out_dir}")
    print(f"[infer-pixart] prompts: {len(prompts)} ({args.height}x{args.width})")
    print(f"[infer-pixart] seed:    {args.seed}")

    print("[infer-pixart] ⚠ Skeleton — plug in PixArt reference implementation.")
    manifest = []
    for i, prompt in enumerate(prompts[:5]):
        out = args.out_dir / f"{i:04d}.png"
        manifest.append({"prompt": prompt, "image": str(out), "status": "skipped_skeleton"})

    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[infer-pixart] manifest saved: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

"""Phase-A smoke tests for T2Image (CPU-only OK)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def test_configs_parseable():
    cfg_dir = ROOT / "configs"
    for p in cfg_dir.glob("*.yaml"):
        with open(p) as f:
            cfg = yaml.safe_load(f)
        assert "name" in cfg
        assert "model" in cfg
        assert "data" in cfg
        assert "training" in cfg
        assert "distributed" in cfg
        print(f"[OK] parseable: {p.name}")


def test_configs_have_required_fields():
    cfg_dir = ROOT / "configs"
    required = ["name", "model.name", "data.train", "training.max_steps", "distributed"]
    for p in cfg_dir.glob("*.yaml"):
        with open(p) as f:
            cfg = yaml.safe_load(f)
        for dotted in required:
            node = cfg
            for part in dotted.split("."):
                assert part in node, f"missing {dotted} in {p.name}"
                node = node[part]
        print(f"[OK] required fields: {p.name}")


def test_fingerprint_runs():
    tmp = Path(tempfile.mkdtemp())
    img_dir = tmp / "images"
    img_dir.mkdir()
    from PIL import Image
    for i in range(5):
        img = Image.new("RGB", (64, 64), (i * 30, 0, 0))
        img.save(img_dir / f"img_{i}.png")
    out = tmp / "fp.json"
    r = subprocess.run(
        [sys.executable, "scripts/compute_data_fingerprint.py",
         "--image_dir", str(img_dir), "--out", str(out), "--name", "smoke"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr
    fp = json.loads(out.read_text())
    assert "dataset_fingerprint" in fp
    assert fp["num_files"] == 5
    print("[OK] fingerprint CLI works")


def test_infer_skeleton():
    """Test that the infer skeleton scripts run without crashing."""
    tmp = Path(tempfile.mkdtemp())
    prompts = tmp / "prompts.txt"
    prompts.write_text("a cat\na dog\n")
    out = tmp / "out_pixart"
    r = subprocess.run(
        [sys.executable, "src/train/infer_pixart.py",
         "--prompts_file", str(prompts), "--out_dir", str(out),
         "--weights", str(tmp / "fake")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr
    assert(out / "manifest.json").exists()
    print("[OK] infer-pixart skeleton works")

    out2 = tmp / "out_flux"
    r = subprocess.run(
        [sys.executable, "src/train/infer_flux.py",
         "--prompts_file", str(prompts), "--out_dir", str(out2),
         "--weights", str(tmp / "fake")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr
    assert (out2 / "manifest.json").exists()
    print("[OK] infer-flux skeleton works")


def test_project_layout():
    """Ensure the expected files and dirs exist."""
    for p in ["docker/Dockerfile", "docker/entrypoint.sh",
              "k8s/submit_train.yaml", "src/train/main.py",
              "src/train/data.py", "src/utils/distributed.py",
              "src/eval/compute_fid.py", "src/eval/compute_clip.py",
              "src/eval/compute_pick.py", "src/eval/compute_hps.py",
              ".gitignore", "README.md", "setup.py", "plan.md", "roadmap_v0.1.md"]:
        fp = ROOT / p
        assert fp.exists(), f"missing {p}"
    print("[OK] project layout correct")


if __name__ == "__main__":
    test_project_layout()
    test_configs_parseable()
    test_configs_have_required_fields()
    test_fingerprint_runs()
    test_infer_skeleton()
    print("\n[ALL T2IMAGE SMOKE TESTS PASSED]")

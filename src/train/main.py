#!/usr/bin/env python3
"""
T2Image — unified training entrypoint.

Dispatches to the selected model's training loop based on `model.name`.
Phase A supports: pixart_sigma, flux1_dev.

Usage:
  torchrun --nproc_per_node=8 src/train/main.py --config configs/pixart_sigma.yaml
  torchrun --nproc_per_node=8 src/train/main.py --config configs/flux1_dev.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig

from src.utils.logger import setup_logger
from src.utils.distributed import setup_distributed, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="T2Image training")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_config(path, overrides):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg.merge_with_dotlist(overrides)
    return cfg


def train(cfg):
    model_name = cfg.model.name
    if model_name == "pixart_sigma":
        from src.train.train_pixart import train_pixart as fn
    elif model_name in ("flux1_dev", "flux_dev"):
        from src.train.train_flux import train_flux as fn
    else:
        raise ValueError(f"Unknown model: {model_name}")
    fn(cfg)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    setup_distributed(cfg.distributed)
    setup_logger(cfg)

    from src.utils.seed import set_seed
    set_seed(cfg.seed, getattr(cfg.reproducibility, 'deterministic_algorithms', False))

    if is_main_process():
        print(OmegaConf.to_yaml(cfg))

    train(cfg)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

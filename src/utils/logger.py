"""Unified W&B / MLflow logger for T2Image."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.distributed import is_main_process


_LAZY = {}


def setup_logger(cfg, dist_cfg=None) -> None:
    log_dir = Path(getattr(cfg.logging, "log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    rank = dist_cfg.rank if dist_cfg else 0
    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank {rank}] %(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%m-%d %H:%M",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / f"rank{rank}.log")],
    )

    if not is_main_process():
        return

    backend = getattr(cfg.logging, "backend", "wandb")
    if backend == "wandb":
        try:
            import wandb
            wandb.init(
                project=cfg.logging.project,
                entity=getattr(cfg.logging, "entity", None),
                name=cfg.name,
                config=_flatten_omegaconf(cfg),
            )
            _LAZY["wandb"] = wandb
            logging.info(f"[W&B] {cfg.logging.project}/{cfg.name}")
        except Exception as e:
            logging.warning(f"[W&B] failed, fallback MLflow: {e}")
            _setup_mlflow(cfg, log_dir)
    else:
        _setup_mlflow(cfg, log_dir)


def _setup_mlflow(cfg, log_dir):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{log_dir}/mlruns")
        mlflow.set_experiment(cfg.name)
        mlflow.start_run(run_name=cfg.name)
        mlflow.log_params(_flatten_omegaconf(cfg))
        _LAZY["mlflow"] = mlflow
    except Exception as e:
        logging.warning(f"[MLflow] failed: {e}")


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if "wandb" in _LAZY:
        _LAZY["wandb"].log(metrics, step=step)
    if "mlflow" in _LAZY:
        import mlflow
        mlflow.log_metrics(metrics, step=step)


def log_images(name: str, images_dir: str, max_n: int = 16) -> None:
    """Log a directory of PNGs to W&B."""
    if "wandb" not in _LAZY:
        return
    import wandb
    from pathlib import Path
    files = sorted(Path(images_dir).glob("*.png"))[:max_n]
    _LAZY["wandb"].log({name: [wandb.Image(str(f)) for f in files]})


def finish():
    if "wandb" in _LAZY:
        _LAZY["wandb"].finish()
    if "mlflow" in _LAZY:
        import mlflow
        mlflow.end_run()


def _flatten_omegaconf(cfg) -> dict:
    try:
        from omegaconf import OmegaConf
        return _flat_dot(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        return {}


def _flat_dot(d: dict, pk: str = "") -> dict:
    r = {}
    for k, v in d.items():
        nk = f"{pk}.{k}" if pk else k
        if isinstance(v, dict):
            r.update(_flat_dot(v, nk))
        else:
            r[nk] = v
    return r

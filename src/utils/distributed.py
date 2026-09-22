"""
T2Image distributed training utilities (FSDP-2 on 8×A100).
Clone of research_kernel_music variant, tuned for 512px image training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


@dataclass
class DistConfig:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    use_fsdp: bool = True
    sharding: str = "full_shard"
    mixed_precision: str = "bf16"
    activation_checkpoint: bool = True
    compile_model: bool = True


def setup_distributed(cfg) -> DistConfig:
    """Initialize process group from config / torchrun env."""
    import os

    dist_cfg = DistConfig(
        backend=getattr(cfg, "backend", "nccl"),
        use_fsdp=getattr(cfg, "fsdp", True),
        sharding=getattr(cfg, "fsdp_sharding", "full_shard"),
        mixed_precision=getattr(cfg, "fsdp_mixed_precision", "bf16"),
        activation_checkpoint=getattr(cfg, "activation_checkpointing", True),
        compile_model=getattr(cfg, "compile", True),
    )

    if "WORLD_SIZE" in os.environ:
        dist_cfg.rank = int(os.environ.get("RANK", 0))
        dist_cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist_cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        dist_cfg.rank = 0
        dist_cfg.world_size = 1

    if dist_cfg.world_size > 1:
        dist.init_process_group(
            backend=dist_cfg.backend,
            init_method=f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT', '29500')}",
            world_size=dist_cfg.world_size,
            rank=dist_cfg.rank,
        )
        torch.cuda.set_device(dist_cfg.local_rank)
    return dist_cfg


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def wrap_fsdp(model: torch.nn.Module, dist_cfg: DistConfig,
              auto_wrap_min_params: int = 1e6) -> torch.nn.Module:
    mp_map = {
        "bf16": MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        "fp16": MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16),
    }
    mp_policy = mp_map.get(dist_cfg.mixed_precision)

    sharding_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }

    wrapped = FSDP(
        model,
        sharding_strategy=sharding_map.get(dist_cfg.sharding, ShardingStrategy.FULL_SHARD),
        mixed_precision=mp_policy,
        auto_wrap_policy=size_based_auto_wrap_policy(min_num_params=int(auto_wrap_min_params)),
        device_id=torch.cuda.current_device(),
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    if dist_cfg.compile_model and hasattr(torch, "compile"):
        wrapped = torch.compile(wrapped, mode="reduce-overhead", fullgraph=False)
    return wrapped

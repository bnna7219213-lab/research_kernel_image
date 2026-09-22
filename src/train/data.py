"""
T2Image dataset loader + fingerprint-lock utilities.

Phase A supports:
  - csv_image   : image path + caption CSV (LAION, JourneyDB)
  - dir_image   : image directory (no caption, synthetic)
  - jsonl_image : JSONL with base64 / url + caption (common in LAION dumps)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from dataclasses import dataclass, field


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


@dataclass
class CsvImageDataset(Dataset):
    image_dir: Path
    csv_path: Path
    caption_column: str = "caption"
    id_column: str = "id"
    ext: str = ".jpg"
    resolution: int = 512
    entries: List[Dict[str, str]] = field(default_factory=list)
    transform: Any = None

    def __post_init__(self):
        self.image_dir = Path(self.image_dir)
        import csv
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.entries.append(row)
        self.transform = self.transform or transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        import csv
        row = self.entries[idx]
        img_path = self.image_dir / f"{row[self.id_column]}{self.ext}"
        if not img_path.exists():
            return None
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            return None
        return {
            "image": tensor,
            "caption": row.get(self.caption_column, ""),
            "id": row[self.id_column],
        }


@dataclass
class DirImageDataset(Dataset):
    image_dir: Path
    resolution: int = 512
    exts: set = field(default_factory=lambda: IMAGE_EXTS)
    files: list = field(default_factory=list)
    transform: Any = None

    def __post_init__(self):
        self.image_dir = Path(self.image_dir)
        self.files = sorted(
            p for p in self.image_dir.rglob("*") if p.suffix.lower() in self.exts
        )
        self.transform = self.transform or transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            return None
        return {"image": tensor, "caption": "", "id": str(self.files[idx].stem)}


@dataclass
class JsonlImageDataset(Dataset):
    jsonl_path: Path
    image_column: str = "image"
    caption_column: str = "caption"
    resolution: int = 512
    entries: list = field(default_factory=list)
    transform: Any = None

    def __post_init__(self):
        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.entries.append(json.loads(line.strip()))
        self.transform = self.transform or transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        # Expect decoded PIL image or tensor; in practice pre-extract to disk
        img = row.get(self.image_column)
        if img is None:
            return None
        if isinstance(img, Image.Image):
            tensor = self.transform(img)
        elif isinstance(img, torch.Tensor):
            tensor = img
        else:
            return None
        return {"image": tensor, "caption": row.get(self.caption_column, ""), "id": str(idx)}


def build_dataloaders(data_cfg, dist_cfg, split: str = "train") -> DataLoader:
    dataset_cfgs = getattr(data_cfg, split, [])
    datasets: List[Dataset] = []
    for ds_spec in dataset_cfgs:
        ds_type = ds_spec.get("type", "dir_image")
        res = ds_spec.get("resolution", 512)

        if ds_type == "csv_image":
            ds = CsvImageDataset(
                image_dir=Path(ds_spec["image_dir"]),
                csv_path=Path(ds_spec["csv_path"]),
                caption_column=ds_spec.get("csv_caption_column", "caption"),
                id_column=ds_spec.get("csv_id_column", "id"),
                ext=ds_spec.get("ext", ".jpg"),
                resolution=res,
            )
        elif ds_type == "jsonl_image":
            ds = JsonlImageDataset(
                jsonl_path=Path(ds_spec["jsonl_path"]),
                image_column=ds_spec.get("image_column", "image"),
                caption_column=ds_spec.get("caption_column", "caption"),
                resolution=res,
            )
        else:
            ds = DirImageDataset(
                image_dir=Path(ds_spec["image_dir"]),
                resolution=res,
            )
        datasets.append(ds)

    combined = ConcatDataset(datasets)
    sampler = DistributedSampler(
        combined,
        num_replicas=dist_cfg.world_size,
        rank=dist_cfg.rank,
        shuffle=(split == "train"),
        seed=42,
    )
    return DataLoader(
        combined,
        batch_size=int(getattr(data_cfg, "batch_size_per_gpu", 4)),
        sampler=sampler,
        num_workers=int(getattr(data_cfg, "num_workers", 8)),
        prefetch_factor=int(getattr(data_cfg, "prefetch_factor", 2)),
        pin_memory=bool(getattr(data_cfg, "pin_memory", True)),
        drop_last=(split == "train"),
        collate_fn=_collate,
    )


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {"image": torch.empty(0, 3, 512, 512), "caption": []}
    images = torch.stack([b["image"] for b in batch])
    captions = [b.get("caption", "") for b in batch]
    ids = [b.get("id", "") for b in batch]
    return {"image": images, "caption": captions, "id": ids}


def check_fingerprint(expected_paths: List[Path]) -> Dict[str, Any]:
    result = {"checked": True, "entries": []}
    for p in expected_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Dataset fingerprint not found: {p}")
        with open(p) as f:
            fp = json.load(f)
        result["entries"].append({
            "name": fp.get("name", ""),
            "fingerprint": fp["dataset_fingerprint"],
        })
    return result

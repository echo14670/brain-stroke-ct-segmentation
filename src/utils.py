from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def resolve_path(value: str | Path, *, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def resolve_data_root(value: str | Path) -> Path:
    env_value = os.environ.get("BSC_DATA_ROOT")
    root = resolve_path(env_value or value)
    if (root / "Brain_Stroke_CT_Dataset").is_dir():
        root = root / "Brain_Stroke_CT_Dataset"
    required = ["Bleeding", "Ischemia", "Normal", "External_Test"]
    if not all((root / name).is_dir() for name in required):
        raise FileNotFoundError(
            f"Dataset root must contain {required}; received: {root}"
        )
    return root


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

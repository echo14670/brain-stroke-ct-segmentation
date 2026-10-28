from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_provisional_split, discover_cases  # noqa: E402
from src.utils import (  # noqa: E402
    load_config,
    resolve_data_root,
    resolve_path,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a provisional fixed baseline split"
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(resolve_path(args.config))
    data_root = resolve_data_root(config["data"]["root"])
    output_path = resolve_path(config["data"]["split_path"])
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Split already exists: {output_path}. Use --overwrite only for "
            "a provisional split."
        )
    payload = build_provisional_split(
        discover_cases(data_root),
        seed=int(config["project"]["seed"]),
        train_fraction=float(config["data"]["train_fraction"]),
        val_fraction=float(config["data"]["val_fraction"]),
        test_fraction=float(config["data"]["test_fraction"]),
    )
    write_json(output_path, payload)
    print(f"Wrote provisional split: {output_path}")
    print(payload["counts"])


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:  # Allows data/mask tests without a local PyTorch install.
    Dataset = object  # type: ignore[assignment,misc]

MAIN_GROUPS = ("Bleeding", "Ischemia", "Normal")
ALL_GROUPS = (*MAIN_GROUPS, "External_Test")


@dataclass(frozen=True)
class CaseRecord:
    image_id: str
    group: str

    @property
    def key(self) -> str:
        return f"{self.group}/{self.image_id}"

    def as_dict(self) -> dict[str, str]:
        return {"image_id": self.image_id, "group": self.group}


def discover_cases(root: str | Path) -> list[CaseRecord]:
    root = Path(root)
    cases: list[CaseRecord] = []
    for group in ALL_GROUPS:
        image_dir = root / group / "PNG"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        for image_path in sorted(image_dir.glob("*.png")):
            cases.append(CaseRecord(image_path.stem, group))
    return cases


def _stable_rank(group: str, image_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group}:{image_id}".encode("utf-8")).hexdigest()


def build_provisional_split(
    cases: Iterable[CaseRecord],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, Any]:
    fraction_sum = train_fraction + val_fraction + test_fraction
    if not np.isclose(fraction_sum, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {fraction_sum}")

    main_cases = [case for case in cases if case.group in MAIN_GROUPS]
    external_cases = [case for case in cases if case.group == "External_Test"]
    splits: dict[str, list[CaseRecord]] = {"train": [], "validation": [], "test": []}

    for group in MAIN_GROUPS:
        group_cases = sorted(
            (case for case in main_cases if case.group == group),
            key=lambda case: _stable_rank(group, case.image_id, seed),
        )
        count = len(group_cases)
        val_count = max(1, round(count * val_fraction))
        test_count = max(1, round(count * test_fraction))
        train_count = count - val_count - test_count
        if train_count <= 0:
            raise ValueError(f"Not enough {group} cases to create a split")
        splits["train"].extend(group_cases[:train_count])
        splits["validation"].extend(
            group_cases[train_count : train_count + val_count]
        )
        splits["test"].extend(group_cases[train_count + val_count :])

    for name in splits:
        splits[name].sort(key=lambda case: case.key)

    split_payload = {
        "version": f"baseline-provisional-v1-seed{seed}",
        "provisional": True,
        "owner_note": (
            "Temporary baseline split. Replace with the data owner's frozen "
            "main_split.json before formal comparison experiments."
        ),
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": val_fraction,
            "test": test_fraction,
        },
        "counts": {name: len(items) for name, items in splits.items()}
        | {"external_test": len(external_cases)},
        "splits": {
            name: [case.as_dict() for case in items]
            for name, items in splits.items()
        },
        "external_test": [case.as_dict() for case in external_cases],
    }
    _validate_split_payload(split_payload, cases)
    return split_payload


def _iter_split_items(payload: dict[str, Any]) -> Iterable[dict[str, str]]:
    for name in ("train", "validation", "test", "external_test"):
        for item in payload.get(name, []):
            if isinstance(item, str):
                yield {"image_id": item, "group": ""}
            else:
                yield item


def _validate_split_payload(
    payload: dict[str, Any], cases: Iterable[CaseRecord]
) -> None:
    case_by_key = {case.key: case for case in cases}
    seen: set[str] = set()
    split_blocks = payload.get("splits")
    if not isinstance(split_blocks, dict):
        raise ValueError("Split JSON must contain a 'splits' mapping")

    normalized: dict[str, Any] = dict(payload)
    normalized.update(split_blocks)
    for item in _iter_split_items(normalized):
        image_id = str(item.get("image_id", ""))
        group = str(item.get("group", ""))
        key = f"{group}/{image_id}" if group else image_id
        if key not in case_by_key:
            raise ValueError(f"Split references unknown case: {key}")
        if key in seen:
            raise ValueError(f"Duplicate case across split files: {key}")
        seen.add(key)

    expected = {case.key for case in cases}
    missing = expected - seen
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Split does not cover all cases; missing {preview}")

    train_keys = {
        f"{item['group']}/{item['image_id']}"
        for item in split_blocks["train"]
        if isinstance(item, dict)
    }
    external_keys = {
        f"{item['group']}/{item['image_id']}"
        for item in payload.get("external_test", [])
        if isinstance(item, dict)
    }
    if train_keys & external_keys:
        raise ValueError("External_Test cases overlap with training data")


def ensure_split(
    *,
    root: str | Path,
    split_path: str | Path,
    seed: int,
    generate_if_missing: bool,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, Any]:
    split_path = Path(split_path)
    cases = discover_cases(root)
    if split_path.is_file():
        import json

        with split_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        _validate_split_payload(payload, cases)
        return payload
    if not generate_if_missing:
        raise FileNotFoundError(
            f"Missing frozen split: {split_path}. The data owner must provide it."
        )
    payload = build_provisional_split(
        cases,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    from src.utils import write_json

    write_json(split_path, payload)
    return payload


def load_split(
    split_path: str | Path, cases: Iterable[CaseRecord] | None = None
) -> dict[str, Any]:
    import json

    split_path = Path(split_path)
    with split_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if cases is not None:
        _validate_split_payload(payload, cases)
    return payload


def _case_map(payload: dict[str, Any], split_name: str) -> list[CaseRecord]:
    if split_name == "external_test":
        items = payload.get("external_test", [])
    else:
        items = payload["splits"].get(split_name, [])
    result: list[CaseRecord] = []
    for item in items:
        if isinstance(item, str):
            result.append(CaseRecord(item, "External_Test"))
        else:
            result.append(CaseRecord(str(item["image_id"]), str(item["group"])))
    return result


def extract_overlay_mask(
    overlay_path: str | Path, mask_mode: str = "red"
) -> np.ndarray:
    with Image.open(overlay_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red_mask = (red >= 80) & ((red - green) >= 20) & ((red - blue) >= 20)
    if mask_mode == "red":
        mask = red_mask
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")
    return mask.astype(np.uint8)


def _load_mask(
    root: Path,
    case: CaseRecord,
    image_shape: tuple[int, int],
    mask_mode: str,
) -> np.ndarray:
    if case.group == "Normal":
        return np.zeros(image_shape, dtype=np.uint8)
    if case.group == "External_Test":
        mask_path = root / case.group / "MASKS" / f"{case.image_id}.png"
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"))
        return (mask > 0).astype(np.uint8)
    overlay_path = root / case.group / "OVERLAY" / f"{case.image_id}.png"
    return extract_overlay_mask(overlay_path, mask_mode=mask_mode)


class StrokeCTDataset(Dataset):
    def __init__(
        self,
        *,
        root: str | Path,
        cases: Iterable[CaseRecord],
        image_size: int,
        mask_mode: str,
        training: bool = False,
        horizontal_flip: bool = False,
    ) -> None:
        self.root = Path(root)
        self.cases = list(cases)
        self.image_size = int(image_size)
        self.mask_mode = mask_mode
        self.training = training
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.cases[index]
        image_path = self.root / case.group / "PNG" / f"{case.image_id}.png"
        with Image.open(image_path) as source:
            image = source.convert("L")
            mask = _load_mask(
                self.root,
                case,
                (image.height, image.width),
                self.mask_mode,
            )
            image = image.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
        mask_image = Image.fromarray(mask * 255, mode="L")
        mask_image = mask_image.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.NEAREST,
        )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask_image) > 0).astype(np.float32)

        if self.training and self.horizontal_flip and random.random() < 0.5:
            image_array = np.fliplr(image_array).copy()
            mask_array = np.fliplr(mask_array).copy()

        import torch

        return {
            "image": torch.from_numpy(image_array).unsqueeze(0),
            "mask": torch.from_numpy(mask_array).unsqueeze(0),
            "image_id": case.image_id,
            "group": case.group,
        }


def dataset_for_split(
    *,
    root: str | Path,
    split_payload: dict[str, Any],
    split_name: str,
    image_size: int,
    mask_mode: str,
    training: bool,
    horizontal_flip: bool = False,
) -> StrokeCTDataset:
    return StrokeCTDataset(
        root=root,
        cases=_case_map(split_payload, split_name),
        image_size=image_size,
        mask_mode=mask_mode,
        training=training,
        horizontal_flip=horizontal_flip,
    )

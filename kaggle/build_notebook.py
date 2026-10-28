from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = Path(__file__).with_name("baseline_kaggle.ipynb")


def _source(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def _archive_sources() -> str:
    paths: list[Path] = []
    for directory in ("configs", "scripts", "src", "data/splits"):
        paths.extend(
            path
            for path in (PROJECT_ROOT / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=path.relative_to(PROJECT_ROOT))
    return base64.b64encode(stream.getvalue()).decode("ascii")


def build_notebook() -> dict[str, object]:
    payload = _archive_sources()
    setup = f"""
from pathlib import Path
import base64
import io
import os
import shutil
import subprocess
import sys
import tarfile

repo = Path('/kaggle/working/brain-stroke-ct-segmentation')
if repo.exists():
    shutil.rmtree(repo)
repo.mkdir(parents=True)

payload = {payload!r}
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz') as archive:
    archive.extractall(repo, filter='data')

candidates = [
    path for path in Path('/kaggle/input').rglob('Brain_Stroke_CT_Dataset')
    if all((path / name).is_dir() for name in ('Bleeding', 'Ischemia', 'Normal', 'External_Test'))
]
if not candidates:
    raise FileNotFoundError('Brain_Stroke_CT_Dataset was not mounted')
data_root = candidates[0]
os.environ['BSC_DATA_ROOT'] = str(data_root)
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
print('repository:', repo)
print('dataset:', data_root)
print('python:', sys.version)
"""
    train = """
subprocess.run(
    [sys.executable, 'scripts/train.py', '--config', 'configs/baseline.yaml'],
    cwd=repo,
    check=True,
    env=os.environ.copy(),
)
"""
    evaluate = """
subprocess.run(
    [sys.executable, 'scripts/evaluate.py', '--config', 'configs/baseline.yaml'],
    cwd=repo,
    check=True,
    env=os.environ.copy(),
)
subprocess.run(
    [sys.executable, 'scripts/infer.py', '--config', 'configs/baseline.yaml'],
    cwd=repo,
    check=True,
    env=os.environ.copy(),
)
"""
    finish = """
results = repo / 'reports' / 'experiments' / 'baseline'
print((results / 'summary.md').read_text(encoding='utf-8'))
(Path('/kaggle/working') / 'BASELINE_RUN_COMPLETE').write_text('complete\\n', encoding='utf-8')
print('result directory:', results)
"""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "baseline-overview",
                "metadata": {},
                "source": _source(
                    """
# Brain Stroke CT Baseline

The notebook is self-contained: it extracts the embedded source bundle into
`/kaggle/working`, then runs training, evaluation, and external inference.

Progress appears live in the training cell and is persisted in `run.log` and
`progress.json`.
"""
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "runtime-setup",
                "metadata": {},
                "outputs": [],
                "source": _source(setup),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "train-baseline",
                "metadata": {},
                "outputs": [],
                "source": _source(train),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "evaluate-baseline",
                "metadata": {},
                "outputs": [],
                "source": _source(evaluate),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "show-results",
                "metadata": {},
                "outputs": [],
                "source": _source(finish),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Notebook size: {NOTEBOOK_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

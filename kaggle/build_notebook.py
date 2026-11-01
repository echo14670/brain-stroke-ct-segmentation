from __future__ import annotations

import base64
import io
import json
import tarfile
import argparse
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


def build_notebook(config_path: str = "configs/baseline.yaml") -> dict[str, object]:
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
os.environ['REQUIRE_CUDA'] = '1'
config_path = os.environ.get('BSC_CONFIG', {config_path!r})
import torch
print('torch:', torch.__version__)
print('torch_cuda:', torch.version.cuda)
print('cuda_device_count:', torch.cuda.device_count())
print('cuda_available:', torch.cuda.is_available())
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
try:
    nvidia_output = subprocess.run(
        ['nvidia-smi'], check=False, capture_output=True, text=True, timeout=15
    )
    print('nvidia-smi rc:', nvidia_output.returncode)
    print(nvidia_output.stdout)
    print(nvidia_output.stderr)
except Exception as error:
    print('nvidia-smi error:', repr(error))
device_nodes = sorted(str(path) for path in Path('/dev').glob('nvidia*'))
print('nvidia device nodes:', device_nodes)
if not device_nodes and shutil.which('nvidia-smi') is None:
    raise RuntimeError(
        'No NVIDIA device was attached to this session; refusing CPU fallback'
    )

wheel_candidates = sorted(Path('/kaggle/input').rglob('torch-*cu128*.whl'))
if not wheel_candidates:
    raise FileNotFoundError('The offline CUDA PyTorch wheel was not mounted')
torch_wheel = wheel_candidates[0]
print('torch wheel:', torch_wheel)
valid_name = torch_wheel.name.replace('2.10.0cu128', '2.10.0+cu128')
valid_wheel = repo.parent / valid_name
shutil.copy2(torch_wheel, valid_wheel)
print('normalized wheel:', valid_wheel)

nvidia_wheels = sorted(
    path
    for path in Path('/kaggle/input').rglob('nvidia_*.whl')
    if 'cu12' in path.name and '-py3-none-' in path.name
)
if not nvidia_wheels:
    raise FileNotFoundError(
        'Offline CUDA 12 NVIDIA dependency wheels were not mounted'
    )
print('nvidia wheels:', len(nvidia_wheels))
subprocess.run(
    [
        sys.executable, '-m', 'pip', 'install', '--no-deps',
        *[str(path) for path in nvidia_wheels],
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable, '-m', 'pip', 'install', '--no-deps',
        '--force-reinstall', str(valid_wheel),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable, '-c',
        'import torch; print("installed torch:", torch.__version__); '
        'print("installed cuda:", torch.version.cuda); '
        'print("installed cuda available:", torch.cuda.is_available())',
    ],
    check=True,
)
print('repository:', repo)
print('dataset:', data_root)
print('config:', config_path)
print('python:', sys.version)
"""
    train = """
subprocess.run(
    [sys.executable, 'scripts/train.py', '--config', config_path],
    cwd=repo,
    check=True,
    env=os.environ.copy(),
)
"""
    evaluate = """
subprocess.run(
    [sys.executable, 'scripts/evaluate.py', '--config', config_path],
    cwd=repo,
    check=True,
    env=os.environ.copy(),
)
subprocess.run(
    [sys.executable, 'scripts/infer.py', '--config', config_path],
    cwd=repo,
    check=True,
    env=os.environ.copy(),
)
"""
    finish = """
import yaml
config_data = yaml.safe_load((repo / config_path).read_text(encoding='utf-8'))
results = repo / config_data['project']['output_dir']
print((results / 'summary.md').read_text(encoding='utf-8'))
marker = Path('/kaggle/working') / f"{config_data['project']['name']}_RUN_COMPLETE"
marker.write_text('complete\\n', encoding='utf-8')
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
# Brain Stroke CT Experiment

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    notebook = build_notebook(args.config)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Notebook size: {NOTEBOOK_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

"""Build the E3 handoff notebook without importing PyTorch or using the network."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

from build_notebook import build_notebook

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/E3_focal_tversky.yaml"


def cell(kind: str, identifier: str, source: str) -> dict:
    result = {"cell_type": kind, "id": identifier, "metadata": {},
              "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(formal: bool = False, batch: bool = False) -> dict:
    notebook = build_notebook(CONFIG)
    setup = "".join(notebook["cells"][1]["source"])
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for directory in ("configs", "scripts", "src", "data/splits", "tests"):
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.add(path, arcname=path.relative_to(ROOT).as_posix())
        for name in ("metrics.json", "summary.md", "submission_manifest.json"):
            path = ROOT / "reports/experiments/baseline" / name
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    payload = base64.b64encode(stream.getvalue()).decode("ascii")
    # Replace only the embedded archive assignment; leave the team's CUDA setup.
    for node in ast.parse(setup).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "payload" for target in node.targets
        ):
            lines = setup.splitlines()
            lines[node.lineno - 1:node.end_lineno] = [f"payload = {payload!r}"]
            setup = "\n".join(lines)
            break
    else:
        raise RuntimeError("Upstream notebook no longer has a payload assignment")
    old = "repo = Path('/kaggle/working/brain-stroke-ct-segmentation')\nif repo.exists():\n    shutil.rmtree(repo)\nrepo.mkdir(parents=True)"
    new = "import uuid\nrepo = Path('/kaggle/working') / ('brain-stroke-e3-' + uuid.uuid4().hex[:12])\nrepo.mkdir(parents=True)"
    if old not in setup:
        raise RuntimeError("Upstream setup changed; review the extraction directory")
    setup = setup.replace(old, new, 1)
    setup = setup.replace(
        f"config_path = os.environ.get('BSC_CONFIG', {CONFIG!r})",
        f"config_path = {CONFIG!r}", 1,
    )
    # Freeze official hashes locally so the cloud run can remain offline.
    official_hashes = json.loads((ROOT / 'kaggle/E3_verified_wheel_hashes.json').read_text(encoding='utf-8'))['sha256']
    verification_before_install = 'OFFICIAL_WHEEL_HASHES = ' + repr(official_hashes) + '\n' + '''
import hashlib

def wheel_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

vendor_hash = OFFICIAL_WHEEL_HASHES.get(valid_name)
if not vendor_hash or wheel_sha256(torch_wheel) != vendor_hash:
    raise RuntimeError('Offline torch wheel does not match the official PyTorch index')
print('Official PyTorch SHA256 verified:', valid_name, vendor_hash, flush=True)

for path in nvidia_wheels:
    expected_hash = OFFICIAL_WHEEL_HASHES.get(path.name)
    if not expected_hash or wheel_sha256(path) != expected_hash:
        raise RuntimeError('Offline NVIDIA wheel does not match official PyPI: ' + path.name)
    print('Official PyPI SHA256 verified:', path.name, flush=True)
'''
    install_marker = "print('nvidia wheels:', len(nvidia_wheels))"
    if install_marker not in setup:
        raise RuntimeError('Upstream wheel setup changed')
    setup = setup.replace(install_marker, install_marker + '\n' + verification_before_install, 1)
    setup += '''
subprocess.run([
    sys.executable, '-c',
    "import torch; assert torch.__version__ == '2.10.0+cu128', torch.__version__; "
    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
    "print('Baseline torch 2.10.0+cu128 + CUDA: PASS')"
], check=True)
valid_wheel.unlink(missing_ok=True)
'''
    split_hash = hashlib.sha256((ROOT / "data/splits/main_split.json").read_bytes()).hexdigest()
    verification = '''
import hashlib
import json
import textwrap

E3_TESTS_PASSED = False
expected_split_sha256 = EXPECTED_HASH
actual_hash = hashlib.sha256((repo / 'data/splits/main_split.json').read_bytes()).hexdigest()
assert actual_hash == expected_split_sha256, 'Frozen split checksum mismatch'

def run_checked(arguments, log_name):
    result = subprocess.run(
        [sys.executable, *arguments], cwd=repo, env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    (repo / log_name).write_text(result.stdout, encoding='utf-8')
    print(result.stdout)
    result.check_returncode()

run_checked(['-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'e3_tests.log')

smoke = textwrap.dedent("""
    import torch
    import yaml
    from src.losses import FocalTverskyLoss, build_loss
    from src.models import build_model
    from src.utils import seed_everything

    with open('configs/experiments/E3_focal_tversky.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open('configs/baseline.yaml', encoding='utf-8') as f:
        baseline = yaml.safe_load(f)
    for key in ('model', 'train', 'evaluation'):
        assert config[key] == baseline[key], key
    for key, value in baseline['data'].items():
        if key != 'generate_split_if_missing':
            assert config['data'][key] == value, key
    assert config['project']['seed'] == baseline['project']['seed'] == 42
    assert config['data']['generate_split_if_missing'] is False
    assert config['project']['output_dir'] != baseline['project']['output_dir']
    assert torch.cuda.is_available(), 'CUDA unavailable; stop before training'
    seed_everything(42)
    model = build_model(config['model']).cuda()
    criterion = build_loss(config['loss'])
    assert isinstance(criterion, FocalTverskyLoss), 'Wrong loss selected'
    image = torch.randn(1, 1, 256, 256, device='cuda')
    mask = torch.zeros_like(image)
    mask[:, :, 100:110, 100:110] = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    scaler = torch.amp.GradScaler('cuda')
    with torch.autocast(device_type='cuda', enabled=True):
        loss = criterion(model(image), mask)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    scaler.step(optimizer)
    scaler.update()
    print('E3 factory + CUDA AMP backward: PASS; loss=', float(loss.detach()))
""")
run_checked(['-c', smoke], 'e3_cuda_smoke.log')
E3_TESTS_PASSED = True
print('验证通过。尚未正式训练；确认结果后再启用下一格。')
'''.replace("EXPECTED_HASH", repr(split_hash))
    train = '''
# 默认关闭：先检查上一格的测试结果，再决定是否启动完整训练。
RUN_TRAINING = False
if RUN_TRAINING:
    assert E3_TESTS_PASSED, '先完成测试及 CUDA 验证'
    subprocess.run([
        sys.executable, '-c',
        "import json, platform, torch; from pathlib import Path; "
        "p=Path('reports/experiments/E3_focal_tversky'); p.mkdir(parents=True, exist_ok=True); "
        "(p/'environment.json').write_text(json.dumps({'python':platform.python_version(), "
        "'torch':torch.__version__, 'cuda':torch.version.cuda, "
        "'cudnn':torch.backends.cudnn.version(), 'gpu':torch.cuda.get_device_name(0)}, "
        "indent=2), encoding='utf-8')"
    ], cwd=repo, check=True, env=os.environ.copy())
    subprocess.run([sys.executable, 'scripts/train.py', '--config', config_path],
                   cwd=repo, check=True, env=os.environ.copy())
else:
    print('正式训练未启动。')
'''
    evaluate = '''
# 仅在验证集选定配置和最佳权重后开启；不要用测试集或 External_Test 调参。
RUN_FINAL_EVALUATION = False
if RUN_FINAL_EVALUATION:
    assert E3_TESTS_PASSED, '先完成测试'
    assert (repo / 'reports/experiments/E3_focal_tversky/best.pt').is_file()
    for entry in ('evaluate.py', 'infer.py'):
        subprocess.run([sys.executable, 'scripts/' + entry, '--config', config_path],
                       cwd=repo, check=True, env=os.environ.copy())
else:
    print('最终测试与外部评估未启动。')
'''
    export = '''
# 打包当前实验产出（可能只有验证日志），供下载保存。
import zipfile
bundle = Path('/kaggle/working') / (repo.name + '-results.zip')
with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in repo.rglob('*'):
        if path.is_file() and (
            'E3_focal_tversky' in path.parts or path.name in ('e3_tests.log', 'e3_cuda_smoke.log')
        ):
            archive.write(path, path.relative_to(repo))
print('结果包：', bundle)
print('本格不会启动训练或评估。权重可保留在下载包中，但不要提交到 Git。')
'''
    notebook["cells"] = [
        cell("markdown", "e3-overview", """
# E3：Focal Tversky Loss 对照实验

已内嵌代码、配置、冻结切分和测试，无需从 GitHub 下载源码。
先运行环境准备与验证两格，核对测试通过后再启动训练。
正式训练和最终评估默认关闭，Run All 不会直接训练。

## 运行前设置
- GPU：选择 T4，与团队基线尽量一致。
- 挂载数据：ozguraslank/brain-stroke-ct-dataset。
- 沿用基线离线依赖：amitmalakarjoy/torch-2100-cu128、koumeimaki/vllm019-cp311-cu128-wheelhouse。
- 配置固定为 E3；基础红色掩码、无增强、种子 42、256×256。
- alpha=0.3 惩罚 FP，beta=0.7 惩罚 FN，gamma=0.75 为直接指数。
- 环境格按仓库安装挂载的 PyTorch 2.10.0+cu128 和 NVIDIA wheels。
- 保持 Internet off；已内嵌从官方 PyTorch/PyPI 固定的哈希，安装前校验离线包。
- 大型安装包使用已挂载文件，不重新下载。校验或 CUDA 检查失败则停止。
- 正式训练使用与仓库一致的 15 轮上限和连续 4 轮未提升早停规则。
- 每次初始化使用独立目录，已有实验结果不会被删除。
"""),
        cell("code", "e3-runtime-setup", setup),
        cell("code", "e3-verification", verification),
        cell("code", "e3-train", train),
        cell("code", "e3-final-evaluation", evaluate),
        cell("code", "e3-export", export),
    ]
    if formal:
        notebook['cells'][0]['source'] = ''.join(notebook['cells'][0]['source']).replace(
            '正式训练和最终评估默认关闭，Run All 不会直接训练。',
            '这是正式训练版：验证通过后自动训练，训练结束自动打包；最终测试与 External_Test 仍然关闭。'
        ).splitlines(keepends=True)
        training_cell = next(c for c in notebook['cells'] if c['id'] == 'e3-train')
        training_cell['source'] = ''.join(training_cell['source']).replace(
            '# 默认关闭：先检查上一格的测试结果，再决定是否启动完整训练。',
            '# 已获用户授权：验证通过后启动正式训练。'
        ).replace('RUN_TRAINING = False', 'RUN_TRAINING = True', 1).splitlines(keepends=True)
    if batch:
        for current in notebook['cells']:
            source = ''.join(current['source']).replace('E3_focal_tversky', 'E3_focal_tversky_batch')
            if current['cell_type'] == 'markdown':
                source += '\n本次为唯一追加候选：batch=True，整批汇总 TP/FP/FN；其余损失参数及训练条件不变。\n'
            if current['id'] == 'e3-verification':
                source = source.replace("assert isinstance(criterion, FocalTverskyLoss), 'Wrong loss selected'",
                    "assert isinstance(criterion, FocalTverskyLoss) and criterion.batch is True, 'Wrong loss selected'")
            current['source'] = source.splitlines(keepends=True)
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--batch', action='store_true')
    args = parser.parse_args()
    filename = 'E3_focal_tversky_train.ipynb' if args.formal else 'E3_focal_tversky_kaggle.ipynb'
    if args.batch:
        filename = filename.replace('E3_focal_tversky', 'E3_focal_tversky_batch')
    destination = ROOT / 'kaggle' / filename
    destination.write_text(json.dumps(build(args.formal, args.batch), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created: {destination.name}")

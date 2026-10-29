"""构建 E4（320×320 输入分辨率）的 Kaggle Notebook，不导入 PyTorch、不访问网络、不在本机训练。

沿用 build_notebook.py / build_e3_notebook.py 的做法：把 configs、scripts、src、data/splits、
tests 打包成 base64 内嵌到 Notebook 里，云端不需要访问 GitHub。
E4 只改一个变量：data.image_size 从 256 改为 320。
"""
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
KAGGLE = ROOT / "kaggle"
CONFIG = "configs/experiments/E4_res320.yaml"
OUTPUT_DIR = "reports/experiments/E4_res320"
# 与 E3 使用同一批离线 wheel，哈希表直接复用，避免出现两份会漂移的副本。
WHEEL_HASHES = "E3_verified_wheel_hashes.json"


def cell(kind: str, identifier: str, source: str) -> dict:
    result = {
        "cell_type": kind,
        "id": identifier,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def _payload_b64() -> str:
    """把 configs、scripts、src、冻结切分、tests 与基线对照结果打包成 base64 内嵌包。"""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for directory in ("configs", "scripts", "src", "data/splits", "tests"):
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.add(path, arcname=path.relative_to(ROOT).as_posix())
        for name in ("metrics.json", "summary.md", "submission_manifest.json"):
            path = ROOT / "reports/experiments/baseline" / name
            if path.is_file():
                archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _repack_payload(setup: str) -> str:
    """把内嵌包写回上游 setup 单元的 payload 赋值行。"""
    payload = _payload_b64()
    for node in ast.parse(setup).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "payload"
            for target in node.targets
        ):
            lines = setup.splitlines()
            lines[node.lineno - 1 : node.end_lineno] = [f"payload = {payload!r}"]
            return "\n".join(lines)
    raise RuntimeError("上游 Notebook 已不再包含 payload 赋值，需要重新核对生成脚本")


def _patch_setup(setup: str) -> str:
    setup = _repack_payload(setup)
    old_repo = (
        "repo = Path('/kaggle/working/brain-stroke-ct-segmentation')\n"
        "if repo.exists():\n"
        "    shutil.rmtree(repo)\n"
        "repo.mkdir(parents=True)"
    )
    new_repo = (
        "import uuid\n"
        "repo = Path('/kaggle/working') / ('brain-stroke-e4-' + uuid.uuid4().hex[:12])\n"
        "repo.mkdir(parents=True)"
    )
    if old_repo not in setup:
        raise RuntimeError("上游解压目录已变化，需要重新核对生成脚本")
    setup = setup.replace(old_repo, new_repo, 1)
    setup = setup.replace(
        f"config_path = os.environ.get('BSC_CONFIG', {CONFIG!r})",
        f"config_path = {CONFIG!r}",
        1,
    )

    official = json.loads((KAGGLE / WHEEL_HASHES).read_text(encoding="utf-8"))["sha256"]
    verify = (
        "OFFICIAL_WHEEL_HASHES = " + repr(official) + "\n" + '''
import hashlib

def wheel_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

vendor_hash = OFFICIAL_WHEEL_HASHES.get(valid_name)
if not vendor_hash or wheel_sha256(torch_wheel) != vendor_hash:
    raise RuntimeError('离线 torch wheel 与官方 PyTorch 索引不一致')
print('PyTorch 官方 SHA256 校验通过:', valid_name, vendor_hash, flush=True)

for path in nvidia_wheels:
    expected_hash = OFFICIAL_WHEEL_HASHES.get(path.name)
    if not expected_hash or wheel_sha256(path) != expected_hash:
        raise RuntimeError('离线 NVIDIA wheel 与官方 PyPI 不一致: ' + path.name)
    print('PyPI 官方 SHA256 校验通过:', path.name, flush=True)
'''
    )
    marker = "print('nvidia wheels:', len(nvidia_wheels))"
    if marker not in setup:
        raise RuntimeError("上游 wheel 安装段已变化，需要重新核对生成脚本")
    setup = setup.replace(marker, marker + "\n" + verify, 1)
    setup += '''
subprocess.run([
    sys.executable, '-c',
    "import torch; assert torch.__version__ == '2.10.0+cu128', torch.__version__; "
    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
    "print('与基线一致的 torch 2.10.0+cu128 + CUDA: PASS')"
], check=True)
valid_wheel.unlink(missing_ok=True)
'''
    return setup


def _online_setup() -> str:
    """联网版准备格：只依赖 1 个数据数据集，torch 由 pip 安装并锁死 2.10.0+cu128。"""
    template = '''
from pathlib import Path
import base64
import importlib.metadata
import io
import os
import subprocess
import sys
import tarfile
import uuid

repo = Path('/kaggle/working') / ('brain-stroke-e4-' + uuid.uuid4().hex[:12])
repo.mkdir(parents=True)
payload = __PAYLOAD__
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz') as archive:
    try:
        archive.extractall(repo, filter='data')
    except TypeError:  # Python < 3.12 没有 filter 参数
        archive.extractall(repo)

candidates = [
    path for path in Path('/kaggle/input').rglob('Brain_Stroke_CT_Dataset')
    if all((path / name).is_dir() for name in ('Bleeding', 'Ischemia', 'Normal', 'External_Test'))
]
if not candidates:
    raise FileNotFoundError(
        '没有找到 Brain_Stroke_CT_Dataset。请先在 Input 挂载团队自建的私有数据集，'
        '其内部需包含 Brain_Stroke_CT_Dataset/Bleeding|Ischemia|Normal|External_Test。'
    )
data_root = candidates[0]
os.environ['BSC_DATA_ROOT'] = str(data_root)
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['REQUIRE_CUDA'] = '1'
config_path = __CONFIG__
print('仓库:', repo)
print('数据:', data_root)
print('配置:', config_path)


def pip_install(arguments):
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', *arguments],
        check=True,
    )


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


torch_version = installed('torch')
print('预装 torch:', torch_version)
if torch_version != '2.10.0+cu128':
    pip_install(['torch==2.10.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128'])
    torch_version = installed('torch')
    print('安装后 torch:', torch_version)
if torch_version != '2.10.0+cu128':
    raise RuntimeError('torch 未锁定到 2.10.0+cu128，与基线不同栈，停止运行')

missing = [spec for spec in ('numpy', 'pandas', 'Pillow', 'matplotlib', 'tqdm', 'PyYAML') if installed(spec) is None]
if missing:
    pip_install(missing)
    print('补装依赖:', missing)

subprocess.run([
    sys.executable, '-c',
    "import torch; assert torch.__version__ == '2.10.0+cu128', torch.__version__; "
    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
    "print('torch 2.10.0+cu128 + CUDA 就绪:', torch.cuda.get_device_name(0))"
], check=True)
'''
    return (
        template.replace("__PAYLOAD__", repr(_payload_b64()))
        .replace("__CONFIG__", repr(CONFIG))
        .strip()
    )


def _latency_source() -> str:
    """单张推理延迟基准：与训练/评估同环境同权重，纯前向 + 端到端两种口径。

    计划 §2.2-4 与 §2.3 都要求报告推理时间，`scripts/evaluate.py` 不输出这项，
    因此在 E4 自己的执行包里补一格，不修改基线负责人维护的 scripts/。
    """
    return '''
import json
import time
from pathlib import Path

import torch
import yaml

from src.data import dataset_for_split, ensure_split
from src.models import build_model
from src.utils import resolve_data_root

config = yaml.safe_load((repo / config_path).read_text(encoding='utf-8'))
output_dir = repo / config['project']['output_dir']
output_dir.mkdir(parents=True, exist_ok=True)

candidates = [output_dir / 'best.pt'] + sorted(Path('/kaggle/input').rglob('best.pt'))
checkpoint = next((path for path in candidates if path.is_file()), None)

model = build_model(config['model'], pretrained=False).cuda()
weight_info = {'source': 'random_init', 'path': None, 'checkpoint_epoch': None}
if checkpoint is not None:
    state = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(state['state_dict'])
    weight_info = {
        'source': 'best.pt',
        'path': str(checkpoint),
        'checkpoint_epoch': int(state.get('epoch', -1)),
    }
    print('已加载训练权重:', checkpoint, '| epoch =', weight_info['checkpoint_epoch'])
else:
    print(
        '未找到 best.pt：改用同结构随机初始化权重。前向延迟只取决于网络结构与输入尺寸，'
        '与权重数值无关，因此该结果仍可用于报告；若需要记录 epoch，请把结果 zip 挂到 /kaggle/input。'
    )
model.eval()
amp = bool(config['train']['amp'])

def forward_latency(size, batch, repeats, warmup=15):
    sample = torch.randn(batch, 1, size, size, device='cuda')
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=amp):
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats / batch * 1000.0

measurements = []
for size in (320, 256):
    for batch, repeats in ((1, 100), (16, 20)):
        milliseconds = forward_latency(size, batch, repeats)
        measurements.append({
            'kind': 'forward_only',
            'image_size': size,
            'batch': batch,
            'ms_per_image': round(milliseconds, 3),
            'images_per_second': round(1000.0 / milliseconds, 2),
        })
        print(f'纯前向 {size}x{size} batch={batch}: {milliseconds:.2f} ms/图 '
              f'({1000.0 / milliseconds:.1f} 图/秒)')

# 端到端：含 PNG 解码 + resize + 前向，batch=1，模拟逐例出报告的场景
try:
    root = resolve_data_root(config['data']['root'])
    split = ensure_split(
        root=root, split_path=config['data']['split_path'], seed=int(config['project']['seed']),
        generate_if_missing=False, train_fraction=float(config['data']['train_fraction']),
        val_fraction=float(config['data']['val_fraction']), test_fraction=float(config['data']['test_fraction']),
    )
    dataset = dataset_for_split(
        root=root, split_payload=split, split_name='external_test',
        image_size=int(config['data']['image_size']), mask_mode=str(config['data']['mask_mode']),
        training=False,
    )
    sample_count = min(30, len(dataset))
    for _ in range(3):
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=amp):
            model(dataset[0]['image'].unsqueeze(0).cuda())
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=amp):
        for index in range(sample_count):
            image = dataset[index]['image'].unsqueeze(0).cuda()
            model(image)
    torch.cuda.synchronize()
    end_to_end_ms = (time.perf_counter() - started) / sample_count * 1000.0
    measurements.append({
        'kind': 'end_to_end_decode_resize_forward',
        'image_size': int(config['data']['image_size']),
        'batch': 1,
        'samples': sample_count,
        'ms_per_image': round(end_to_end_ms, 3),
    })
    print(f'端到端（解码+resize+前向）{sample_count} 例: {end_to_end_ms:.2f} ms/图')
except Exception as error:  # 数据未挂载时不阻塞纯前向基准
    print('端到端测速跳过（数据不可用）:', repr(error))

benchmark = {
    'experiment': config['project']['name'],
    'gpu': torch.cuda.get_device_name(0),
    'torch': torch.__version__,
    'amp': amp,
    'weights': weight_info,
    'note': 'forward_only 不含 PNG 解码与 resize；end_to_end 为该机的实测上限口径',
    'measurements': measurements,
}
(output_dir / 'latency_benchmark.json').write_text(
    json.dumps(benchmark, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8'
)
print('已写出:', output_dir / 'latency_benchmark.json')
'''.strip()


def _latency_wrapper() -> str:
    """把测速脚本落盘后用子进程执行。

    在 kernel 进程内 import src 会因为 cwd 不是仓库根而失败；而且环境格在 pip 替换 torch
    之前已经 import 过 torch，内核里缓存的是预装版本。子进程同时解决这两个问题：
    `cwd=repo` 让 `src` 可导入，解释器是全新的因此加载的是 2.10.0+cu128。
    """
    script = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            "",
            "# 由 Notebook 写入仓库根后执行：cwd 即仓库根，因此 src 可直接导入",
            "repo = Path.cwd()",
            f"config_path = {CONFIG!r}",
            "",
            _latency_source(),
        ]
    )
    return '''
import subprocess
import sys

bench_path = repo / 'e4_latency_bench.py'
bench_path.write_text(__BENCH_SCRIPT__, encoding='utf-8')
finished = subprocess.run(
    [sys.executable, str(bench_path)], cwd=repo, env=os.environ.copy(),
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
(repo / 'e4_latency_bench.log').write_text(finished.stdout, encoding='utf-8')
print(finished.stdout)
finished.check_returncode()
print('延迟基准文件:', repo / 'reports/experiments/E4_res320/latency_benchmark.json')
'''.replace(
        "__BENCH_SCRIPT__", repr(script)
    ).strip()


def build(
    formal: bool = False,
    online: bool = False,
    full: bool = False,
    latency_only: bool = False,
) -> dict:
    notebook = build_notebook(CONFIG)
    setup = _online_setup() if online else _patch_setup("".join(notebook["cells"][1]["source"]))
    split_sha256 = hashlib.sha256(
        (ROOT / "data/splits/main_split.json").read_bytes()
    ).hexdigest()

    verification = '''
import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

E4_VERIFIED = False
EXPECTED_SPLIT_SHA256 = SPLIT_HASH

actual_split = hashlib.sha256((repo / 'data/splits/main_split.json').read_bytes()).hexdigest()
assert actual_split == EXPECTED_SPLIT_SHA256, '冻结切分校验失败'
print('冻结切分校验通过:', actual_split)

def run_checked(arguments, log_name):
    result = subprocess.run(
        [sys.executable, *arguments], cwd=repo, env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    (repo / log_name).write_text(result.stdout, encoding='utf-8')
    print(result.stdout)
    result.check_returncode()

run_checked(['-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'e4_tests.log')

smoke = textwrap.dedent("""
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from src.data import dataset_for_split, ensure_split
    from src.losses import build_loss
    from src.models import build_model
    from src.utils import resolve_data_root, seed_everything

    with open('configs/baseline.yaml', encoding='utf-8') as stream:
        baseline = yaml.safe_load(stream)
    with open('configs/experiments/E4_res320.yaml', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    # 单变量断言：除 image_size 及命名/输出目录外，其余必须与基线逐项相同
    assert config['data']['image_size'] == 320, config['data']['image_size']
    assert baseline['data']['image_size'] == 256, baseline['data']['image_size']
    for key in ('model', 'loss', 'train', 'evaluation'):
        assert config[key] == baseline[key], key
    for key, value in baseline['data'].items():
        if key in ('image_size', 'generate_split_if_missing'):
            continue
        assert config['data'][key] == value, key
    assert config['data']['generate_split_if_missing'] is False
    assert config['project']['seed'] == baseline['project']['seed'] == 42
    assert config['project']['output_dir'] == 'reports/experiments/E4_res320'
    print('单变量校验通过：仅 image_size 256 -> 320')

    assert torch.cuda.is_available(), 'CUDA 不可用'
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print('GPU:', torch.cuda.get_device_name(0), f'{total_gib:.1f} GiB')

    seed_everything(config['project']['seed'])
    root = resolve_data_root(config['data']['root'])
    split = ensure_split(
        root=root, split_path='data/splits/main_split.json', seed=42,
        generate_if_missing=False, train_fraction=0.70,
        val_fraction=0.15, test_fraction=0.15,
    )
    print('切分:', split['version'], split['counts'])

    dataset = dataset_for_split(
        root=root, split_payload=split, split_name='train',
        image_size=320, mask_mode='red', training=False,
    )
    item = dataset[0]
    assert tuple(item['image'].shape) == (1, 320, 320), item['image'].shape
    assert tuple(item['mask'].shape) == (1, 320, 320), item['mask'].shape
    print('数据样本:', item['image_id'], tuple(item['image'].shape), '| 掩码与图像同步缩放')

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    images = batch['image'].cuda()
    masks = batch['mask'].cuda()

    model = build_model(config['model']).cuda()
    criterion = build_loss(config['loss'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['train']['learning_rate']))
    scaler = torch.amp.GradScaler('cuda', enabled=bool(config['train']['amp']))

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type='cuda', enabled=bool(config['train']['amp'])):
        loss = criterion(model(images), masks)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    peak_train = torch.cuda.max_memory_allocated() / 1024 ** 3
    reserved_train = torch.cuda.max_memory_reserved() / 1024 ** 3
    assert torch.isfinite(loss), float(loss)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.grad is not None
    )
    print('训练步通过 | batch=8 @320x320 | loss=', float(loss.detach()))
    print(f'训练峰值显存 allocated={peak_train:.2f} GiB '
          f'reserved={reserved_train:.2f} GiB / total={total_gib:.2f} GiB')

    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=bool(config['train']['amp'])):
        logits = model(torch.randn(16, 1, 320, 320, device='cuda'))
    assert tuple(logits.shape) == (16, 1, 320, 320), logits.shape
    peak_eval = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f'评估推理通过 | batch=16 @320x320 | 峰值 allocated={peak_eval:.2f} GiB')
    if peak_train > 0.9 * total_gib:
        raise RuntimeError('训练峰值显存超过总显存的 90%，正式训练有 OOM 风险')
    print('显存余量检查通过，可以启动正式训练。')
""")
run_checked(['-c', smoke], 'e4_cuda_smoke.log')
E4_VERIFIED = True
print('验证完成：本轮没有启动训练。确认显存与断言结果后再切正式版 Notebook。')
'''.replace("SPLIT_HASH", repr(split_sha256))

    latency = _latency_wrapper()

    train = '''
# 默认关闭：先核对上一格的测试、单变量与显存结果，再决定是否启动完整训练。
RUN_TRAINING = False
if RUN_TRAINING:
    assert E4_VERIFIED, '先完成验证格'
    subprocess.run([
        sys.executable, '-c',
        "import json, platform, torch; from pathlib import Path; "
        "p = Path('reports/experiments/E4_res320'); p.mkdir(parents=True, exist_ok=True); "
        "(p / 'environment.json').write_text(json.dumps({'python': platform.python_version(), "
        "'torch': torch.__version__, 'cuda': torch.version.cuda, "
        "'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)}, "
        "indent=2), encoding='utf-8')"
    ], cwd=repo, check=True, env=os.environ.copy())
    subprocess.run(
        [sys.executable, 'scripts/train.py', '--config', config_path],
        cwd=repo, check=True, env=os.environ.copy(),
    )
else:
    print('正式训练未启动。')
'''

    evaluate = '''
# 仅在最佳权重生成后开启；不要用测试集或 External_Test 调参、选轮次或改阈值。
RUN_FINAL_EVALUATION = False
if RUN_FINAL_EVALUATION:
    assert E4_VERIFIED, '先完成验证格'
    assert (repo / 'reports/experiments/E4_res320/best.pt').is_file(), '缺少 best.pt'
    for entry in ('evaluate.py', 'infer.py'):
        subprocess.run(
            [sys.executable, 'scripts/' + entry, '--config', config_path],
            cwd=repo, check=True, env=os.environ.copy(),
        )
else:
    print('内部测试与 External_Test 评估未启动。')
'''

    export = '''
# 打包 E4 产出（可能只有验证日志），供下载保存。
import zipfile
bundle = Path('/kaggle/working') / (repo.name + '-e4-results.zip')
with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in repo.rglob('*'):
        if path.is_file() and (
            'E4_res320' in path.parts
            or path.name in ('e4_tests.log', 'e4_cuda_smoke.log', 'e4_latency_bench.log')
        ):
            archive.write(path, path.relative_to(repo))
print('结果包：', bundle)
print('权重可保留在下载包中，但不要提交到 Git。')
'''

    notebook["cells"] = [
        cell("markdown", "e4-overview", """
# E4：320×320 输入分辨率对照实验

已内嵌代码、配置、冻结切分与测试，无需从 GitHub 下载源码，也不在本机运行。
先跑「环境准备」与「验证」两格，核对单变量断言、320×320 数据形状和显存占用，再启动训练。

## 运行前设置
- GPU：选择 T4，与团队基线 `baseline-v1` 保持一致。
- 挂载数据：`ozguraslank/brain-stroke-ct-dataset`。
- 沿用基线离线依赖：`amitmalakarjoy/torch-2100-cu128`、`koumeimaki/vllm019-cp311-cu128-wheelhouse`。
- 保持 Internet off；已内嵌官方 PyTorch/PyPI 固化的 wheel 哈希，安装前逐个校验。
- 实验变量只有 `data.image_size`：256 → 320。模型、损失、优化器、batch size 8、15 轮、
  种子 42、阈值 0.5、冻结切分和统一评估脚本全部与基线一致。
- 图像按 BILINEAR、掩码按 NEAREST 从同一张原图同步缩放到 320×320，不存在图像/掩码错位。
- 显存不足时不允许改 batch size；应由基线负责人评估是否需要梯度累积（`scripts/train.py` 目前没有实现）。
- 推理延迟格输出纯前向与端到端两种口径的 ms/图，写入 `latency_benchmark.json`。
- 正式训练与最终评估默认关闭，Run All 不会直接训练。
"""),
        cell("code", "e4-runtime-setup", setup),
        cell("code", "e4-verification", verification),
        cell("code", "e4-train", train),
        cell("code", "e4-latency", latency),
        cell("code", "e4-final-evaluation", evaluate),
        cell("code", "e4-export", export),
    ]

    if latency_only:
        notebook["cells"] = [
            cell("markdown", "e4-latency-overview", """
# E4：单张推理延迟补测

只做环境准备与推理延迟测量，不训练、不评估。
用途：正式训练已经完成，仅补齐计划 §2.2-4 / §2.3 要求的**单张推理时间**。

## 运行前设置
- GPU：T4（与正式训练同一张卡）。
- 挂载 `ozguraslank/brain-stroke-ct-dataset` 与两个离线 wheel 数据集（环境格强制要求这三个）。
- 权重**可选**：纯前向延迟只取决于网络结构与输入尺寸，与权重数值无关。若想记录 epoch，把结果包
  `brain-stroke-e4-*-e4-results.zip` 上传为数据集（或 *Add Input → Your Work* 选上一次 Notebook 输出），
  环境格会在 `/kaggle/input` 下递归找 `best.pt`；找不到则用同结构随机初始化权重，并在 JSON 里标
  `"weights": {"source": "random_init"}`。
- Internet 保持关闭。
"""),
            cell("code", "e4-runtime-setup", setup),
            cell("code", "e4-latency", latency),
            cell("code", "e4-export", export),
        ]
        return notebook

    if online:
        notebook["cells"][0]["source"] = "".join(notebook["cells"][0]["source"]).replace(
            "- 沿用基线离线依赖：`amitmalakarjoy/torch-2100-cu128`、`koumeimaki/vllm019-cp311-cu128-wheelhouse`。\n"
            "- 保持 Internet off；已内嵌官方 PyTorch/PyPI 固化的 wheel 哈希，安装前逐个校验。",
            "- **本版为联网版**：Settings 里必须打开 Internet，且只需要挂载一个数据数据集。\n"
            "- torch 由 pip 安装并锁死 `2.10.0+cu128`，与基线同栈；版本不一致时直接报错停止。",
        ).splitlines(keepends=True)

    if formal:
        notebook["cells"][0]["source"] = "".join(notebook["cells"][0]["source"]).replace(
            "正式训练与最终评估默认关闭，Run All 不会直接训练。",
            "这是正式训练版：验证通过后自动训练并打包；内部测试与 External_Test 评估仍需手动开启。",
        ).splitlines(keepends=True)
        training_cell = next(c for c in notebook["cells"] if c["id"] == "e4-train")
        training_cell["source"] = "".join(training_cell["source"]).replace(
            "# 默认关闭：先核对上一格的测试、单变量与显存结果，再决定是否启动完整训练。",
            "# 已确认验证结果：验证格通过后直接启动正式训练。",
        ).replace("RUN_TRAINING = False", "RUN_TRAINING = True", 1).splitlines(keepends=True)

    if full:
        # 一次提交内完成 训练 -> 统一评估 -> 外部推理 -> 打包。
        # 最佳权重由验证集 Dice 选出，测试集与 External_Test 不参与任何选择。
        evaluation_cell = next(c for c in notebook["cells"] if c["id"] == "e4-final-evaluation")
        evaluation_cell["source"] = "".join(evaluation_cell["source"]).replace(
            "# 仅在最佳权重生成后开启；不要用测试集或 External_Test 调参、选轮次或改阈值。",
            "# 已确认验证结果：训练结束后用验证集选出的 best.pt 跑统一评估与外部推理。",
        ).replace("RUN_FINAL_EVALUATION = False", "RUN_FINAL_EVALUATION = True", 1).splitlines(keepends=True)
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", action="store_true", help="生成自动训练的正式版")
    parser.add_argument(
        "--full",
        action="store_true",
        help="生成一步到底版：训练 + 统一评估 + 外部推理 + 打包（隐含 --formal）",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="生成联网版：Internet 打开、只挂数据数据集、pip 安装 torch 2.10.0+cu128",
    )
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="生成只测推理延迟的补测版（需要额外挂载 best.pt）",
    )
    args = parser.parse_args()
    formal = args.formal or args.full
    filename = "E4_res320_train.ipynb" if formal else "E4_res320_kaggle.ipynb"
    if args.full:
        filename = "E4_res320_full.ipynb"
    if args.latency_only:
        filename = "E4_res320_latency.ipynb"
    if args.online:
        filename = filename.replace("E4_res320", "E4_res320_online")
    destination = KAGGLE / "E4_res320" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            build(formal, args.online, args.full, args.latency_only),
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"已生成: {destination.relative_to(ROOT).as_posix()}")

"""Build a self-contained E1 Kaggle notebook from this checkout."""
from pathlib import Path
import base64
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = []
    for directory in ('src', 'scripts', 'configs', 'data/splits'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.json'))
    paths.extend([ROOT/'reports/experiments/baseline/metrics.json'])
    files = {p.relative_to(ROOT).as_posix(): base64.b64encode(p.read_bytes()).decode('ascii') for p in sorted(paths)}
    encoded = base64.b64encode(json.dumps(files).encode()).decode('ascii')
    split_hash = hashlib.sha256((ROOT/'data/splits/main_split.json').read_bytes()).hexdigest()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    setup = '''from pathlib import Path
import base64, hashlib, importlib, json, os, platform, subprocess, sys
import torch

assert torch.cuda.is_available(), 'Enable GPU before running this notebook.'
for name in ('numpy', 'PIL', 'matplotlib', 'yaml', 'tqdm', 'pandas'):
    importlib.import_module(name)
repo = Path('/kaggle/working/brain-stroke-e1')
repo.mkdir(parents=True, exist_ok=True)
source_files = json.loads(base64.b64decode(PAYLOAD))
for relative, value in source_files.items():
    target = (repo / relative).resolve()
    assert repo.resolve() in target.parents
    raw = base64.b64decode(value)
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError(f'Existing source differs: {target}; use a fresh session.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)

roots = [p for p in Path('/kaggle/input').rglob('Brain_Stroke_CT_Dataset')
         if p.is_dir() and all((p/g).is_dir() for g in ('Bleeding','Ischemia','Normal','External_Test'))]
assert len(roots) == 1, f'Expected one dataset root, found {roots}'
os.environ['BSC_DATA_ROOT'] = str(roots[0])
os.environ['REQUIRE_CUDA'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
config_path = 'configs/experiments/E1_data_quality.yaml'
sys.path.insert(0, str(repo))
from src.data import discover_cases, load_split
split_file = repo / 'data/splits/main_split.json'
assert hashlib.sha256(split_file.read_bytes()).hexdigest() == SPLIT_HASH
split = load_split(split_file, discover_cases(roots[0]))
import yaml
config = yaml.safe_load((repo/config_path).read_text(encoding='utf-8'))
assert config['data']['mask_mode'] == 'red_blue'
assert config['data']['generate_split_if_missing'] is False
out = repo/config['project']['output_dir']
out.mkdir(parents=True, exist_ok=True)
(out/'environment.json').write_text(json.dumps({
    'python':sys.version, 'platform':platform.platform(), 'torch':torch.__version__,
    'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0),
    'gpu_count':torch.cuda.device_count(), 'base_commit':BASE_COMMIT,
    'split_sha256':SPLIT_HASH,
    'source_sha256':{p:hashlib.sha256(base64.b64decode(v)).hexdigest() for p,v in source_files.items()},
}, indent=2), encoding='utf-8')
print('GPU:',torch.cuda.get_device_name(0))
print('Dataset:',roots[0])
print('Split:',split['version'],split['counts'])
print('Mask mode:',config['data']['mask_mode'])
print('Output:',out)
print('SETUP OK - run the training cell next')
'''.replace('PAYLOAD', repr(encoded)).replace('SPLIT_HASH',repr(split_hash)).replace('BASE_COMMIT',repr(commit))
    train = '''# Full E1 training; no debug batch/epoch limits.
if (out/'best.pt').exists():
    raise RuntimeError('A checkpoint already exists. Do not restart training over it; inspect results first.')
subprocess.run([sys.executable,'scripts/train.py','--config',config_path],
               cwd=repo,env=os.environ.copy(),check=True)
'''
    evaluate = '''# Run after training completes. Keep threshold and external annotations fixed.
subprocess.run([sys.executable,'scripts/evaluate.py','--config',config_path],
               cwd=repo,env=os.environ.copy(),check=True)
subprocess.run([sys.executable,'scripts/infer.py','--config',config_path],
               cwd=repo,env=os.environ.copy(),check=True)
print((out/'summary.md').read_text(encoding='utf-8'))
print('Results:',out)
'''
    show = '''from IPython.display import display, FileLink, Image
print('E1 metrics (internal annotations use red_blue):')
print((out/'metrics.json').read_text(encoding='utf-8'))
if (out/'curves.png').exists():
    display(Image(filename=str(out/'curves.png')))
for filename in ('metrics.json','summary.md','run.log','train_summary.json','environment.json','best.pt'):
    if (out/filename).exists():
        display(FileLink(str(out/filename)))
print('Save a version with outputs after completion. Do not publish weights to Git.')
'''
    markdown = '''# E1 数据质量实验 — 乔震宇

从 baseline-v1 的统一训练、评估和推理代码运行。唯一实验变量为红色掩码改为红蓝并集；冻结病例集合、256 输入、种子 42、Dice+BCE 和训练参数保持一致。

依次运行四个代码单元。使用 Kaggle 已验证可用的 GPU PyTorch，不重装 CUDA，不需要从 GitHub 下载代码。

已发现跨划分完全重复图像，保留冻结切分并在报告披露。外部标签与空掩码冲突保留原样。当前结果属于原冻结切分上的对照实验，不能当成无泄漏泛化证据。

内部评估的掩码定义也随 E1 改变，因此与历史基线指标的差异不完全等同于模型改进；正式结论需要在统一参考掩码上比较两个检查点。External_Test 使用原官方掩码及标签，不参与调参。
'''
    cells=[{'cell_type':'markdown','id':'overview','metadata':{},'source':markdown.splitlines(True)}]
    for i, source in enumerate((setup,train,evaluate,show),1):
        compile(source, f'cell-{i}', 'exec')
        cells.append({'cell_type':'code','id':f'e1-{i}','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(True)})
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/E1_data_quality.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Created:',path)
    print('Embedded source files:',len(files),'Notebook bytes:',path.stat().st_size)
    print('Validated: all code cells compile; split SHA256:',split_hash)


if __name__ == '__main__':
    main()

"""Read-only E1 audit: python -m src.data.audit_quality"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset import discover_cases, extract_overlay_mask, load_split


def main():
    root = Path('data/raw/Brain_Stroke_CT_Dataset')
    split = Path('data/splits/main_split.json')
    output = Path('reports/experiments/E1_data_quality/audit')
    before = split.read_bytes()
    cases = discover_cases(root)
    payload = load_split(split, cases)
    blocks = dict(payload['splits'], external_test=payload['external_test'])
    membership = {f"{x['group']}/{x['image_id']}": name for name, items in blocks.items() for x in items}
    with (root / 'External_Test/labels.csv').open(encoding='utf-8-sig', newline='') as f:
        labels = {r['image_id']: int(r['Stroke']) for r in csv.DictReader(f)}
    output.mkdir(parents=True, exist_ok=True)
    rows, errors, examples = [], [], []
    hashes = defaultdict(list)
    for index, case in enumerate(cases, 1):
        try:
            with Image.open(root / case.group / 'PNG' / f'{case.image_id}.png') as image:
                gray = np.asarray(image.convert('L'))
                w, h = image.size
            digest = hashlib.sha256(f'{w}x{h}:L:'.encode() + gray.tobytes()).hexdigest()
            hashes[digest].append({'key': case.key, 'split': membership[case.key]})
            red_count = union_count = added = ''
            empty_red = empty_union = conflict = False
            mw, mh = w, h
            label = ''
            if case.group in ('Bleeding', 'Ischemia'):
                overlay = root / case.group / 'OVERLAY' / f'{case.image_id}.png'
                red = extract_overlay_mask(overlay, 'red')
                union = extract_overlay_mask(overlay, 'red_blue')
                red_count, union_count = int(red.sum()), int(union.sum())
                added = union_count - red_count
                mh, mw = union.shape
                empty_red, empty_union = red_count == 0, union_count == 0
                if added and membership[case.key] == 'train' and len(examples) < 6:
                    examples.append((case.key, gray.copy(), red, union))
            elif case.group == 'External_Test':
                with Image.open(root / case.group / 'MASKS' / f'{case.image_id}.png') as image:
                    mask = np.asarray(image.convert('L')) > 0
                    mw, mh = image.size
                union_count = int(mask.sum())
                empty_union = union_count == 0
                label = labels[case.image_id]
                conflict = bool(label) != bool(union_count)
            rows.append(dict(key=case.key, group=case.group, split=membership[case.key],
                width=w, height=h, non_512=(w,h)!=(512,512), non_square=w!=h,
                mask_width=mw, mask_height=mh, mask_size_mismatch=(w,h)!=(mw,mh),
                red_pixels=red_count, red_blue_pixels=union_count, added_blue_pixels=added,
                positive_empty_red=empty_red, empty_union=empty_union,
                stroke_label=label, label_mask_conflict=conflict, image_sha256=digest))
        except Exception as e:
            errors.append(dict(key=case.key, error=repr(e)))
        if index % 500 == 0 or index == len(cases):
            print(f'Audited {index}/{len(cases)}', flush=True)
    def save_csv(name, items, fields):
        with (output/name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(items)
    fields = list(rows[0]) if rows else ['key']
    save_csv('cases.csv', rows, fields)
    anomalies = [r for r in rows if r['non_512'] or r['mask_size_mismatch'] or r['positive_empty_red'] or r['label_mask_conflict'] or (r['group'] in ('Bleeding','Ischemia') and r['empty_union'])]
    save_csv('anomalies.csv', anomalies, fields)
    save_csv('errors.csv', errors, ['key','error'])
    duplicates = [v for v in hashes.values() if len(v)>1]
    cross = [v for v in duplicates if len({x['split'] for x in v})>1]
    save_csv('cross_split_duplicates.csv', [dict(duplicate_group=i, **x) for i, group in enumerate(cross,1) for x in group], ['duplicate_group','key','split'])
    main_rows = [r for r in rows if r['group']!='External_Test']
    external = [r for r in rows if r['group']=='External_Test']
    summary = dict(split_version=payload['version'], split_provisional=payload.get('provisional'),
        split_sha256=hashlib.sha256(before).hexdigest(), split_counts={k:len(v) for k,v in blocks.items()},
        discovered=len(cases), audited=len(rows), read_errors=len(errors),
        main_non_512=sum(r['non_512'] for r in main_rows),
        external_non_512=sum(r['non_512'] for r in external),
        external_non_square=sum(r['non_square'] for r in external),
        mask_size_mismatch=sum(r['mask_size_mismatch'] for r in rows),
        main_positive_empty_red=sum(r['positive_empty_red'] for r in main_rows),
        main_positive_empty_union=sum(r['empty_union'] for r in main_rows),
        blue_added_by_group=dict(Counter(r['group'] for r in main_rows if r['added_blue_pixels'] and r['added_blue_pixels']>0)),
        external_positive_empty_mask=sum(r['stroke_label']==1 and r['empty_union'] for r in external),
        external_label_mask_conflict=sum(r['label_mask_conflict'] for r in external),
        exact_duplicate_groups=len(duplicates), cross_split_duplicate_groups=len(cross),
        split_unchanged=split.read_bytes()==before)
    (output/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if examples:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(examples),3,figsize=(10,3*len(examples)),squeeze=False)
        for axrow,(key,gray,red,union) in zip(axes,examples):
            for ax,data,title in zip(axrow,(gray,red,union),(key+' (train)','Red mask','Red + blue mask')):
                ax.imshow(data,cmap='gray',vmin=0,vmax=255 if data is gray else 1)
                ax.set_title(title)
                ax.axis('off')
        fig.tight_layout()
        fig.savefig(output/'mask_comparison.png',dpi=130)
        plt.close(fig)
    notes = ['# E1 数据质量审计','','本次只审计，不修改原始图像、掩码、标签或冻结切分。','','| 项目 | 结果 |','| --- | --- |']
    notes += [f'| `{k}` | `{v}` |' for k,v in summary.items()]
    notes += ['', '## 解释与后续处理', '',
        '- cases.csv 是逐病例记录；anomalies.csv 是异常候选，不能直接据此删除病例。',
        '- 红蓝并集复用基线的颜色提取实现；对照图仅使用训练集，需人工核验新增区域。',
        '- 外部标签与掩码冲突保持原样并单独报告，不补画标签，不用外部结果调参。',
        '- 原始图像与掩码尺寸或坐标不一致时需逐例核查，不能仅通过缩放宣称修复。',
        '- 完全重复检测使用灰度像素和尺寸的 SHA256，不能排除近似切片或同患者泄漏。',
        '- 如发现泄漏需要改变病例集合，须与基线负责人协作创建新切分并重跑基线，不能私改 baseline-v1。',
        '- 本报告为数据审计，不是训练或性能评估结果。', '']
    (output/'audit_summary.md').write_text('\n'.join(notes),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Audit output:',output.resolve())
    if errors:
        raise SystemExit('Read errors found. Inspect errors.csv before training.')


if __name__ == '__main__':
    main()

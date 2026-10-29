"""Read-only validation diagnosis of the saved E3 best/last checkpoints."""
import json
import hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data import dataset_for_split
from src.models import build_model
from src.evaluation.metrics import binary_case_metrics, aggregate_case_metrics
from src.losses import build_loss
from src.utils import resolve_data_root

assert torch.__version__ == '2.10.0+cu128'
paths = [p for p in Path('/kaggle/input').rglob('best.pt')
         if p.parent.name == 'E3_focal_tversky' and (p.parent/'last.pt').is_file()]
assert len(paths) == 1, [str(p) for p in paths]
saved = paths[0].parent
split_file = saved/'split_snapshot.json'
split = json.loads(split_file.read_text())
assert split['splits'] == json.loads(Path('data/splits/main_split.json').read_text())['splits']
summary = json.loads((saved/'train_summary.json').read_text())
report = {'source': str(saved), 'split': 'validation only', 'threshold': .5,
          'torch': torch.__version__, 'training_summary': summary, 'checkpoints': {}}
for label in ('best', 'last'):
    checkpoint_path = saved/(label+'.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    dataset = dataset_for_split(root=resolve_data_root(config['data']['root']),
        split_payload=split, split_name='validation', image_size=256,
        mask_mode='red', training=False)
    loader = DataLoader(dataset, batch_size=16, num_workers=2, shuffle=False)
    model = build_model(config['model']).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    criterion = build_loss(config['loss'])
    rows, losses = [], []
    grouped = {}
    total_prob, total_pixels, finite = 0., 0, True
    max_prob, min_prob, zero_prediction_cases = 0., 1., 0
    all_background_losses = []
    with torch.inference_mode():
        for batch in loader:
            masks = batch['mask'].cuda()
            logits = model(batch['image'].cuda())
            probabilities = logits.sigmoid()
            finite = finite and bool(torch.isfinite(logits).all())
            max_prob = max(max_prob, float(probabilities.max()))
            min_prob = min(min_prob, float(probabilities.min()))
            total_prob += float(probabilities.double().sum())
            total_pixels += probabilities.numel()
            losses.append(float(criterion(logits, masks)))
            for i in range(len(masks)):
                pred = (probabilities[i, 0] >= .5).cpu().numpy()
                target = masks[i, 0].cpu().numpy()
                row = binary_case_metrics(pred, target)
                row['group'] = str(batch['group'][i])
                rows.append(row)
                zero_prediction_cases += int(row['predicted_positive'] == 0)
                n = row['ground_truth_positive']
                background_loss = max(1.-1./(.7*n+1.), 1e-7)**.75
                all_background_losses.append(background_loss)
                category = 'nonempty_mask' if n else 'empty_mask'
                stats = grouped.setdefault(category, {'cases':0,'loss_sum':0.,'mask_pixels':0})
                stats['cases'] += 1
                stats['loss_sum'] += float(criterion(logits[i:i+1], masks[i:i+1]))
                stats['mask_pixels'] += n
    for stats in grouped.values():
        stats['mean_loss'] = stats.pop('loss_sum') / stats['cases']
    report['checkpoints'][label] = {
        'epoch': checkpoint['epoch'], 'sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'metrics': aggregate_case_metrics(rows), 'finite_logits': finite,
        'min_probability':min_prob, 'max_probability':max_prob,
        'mean_probability':total_prob/total_pixels,
        'predicted_positive_pixels':sum(r['predicted_positive'] for r in rows),
        'zero_prediction_cases':zero_prediction_cases,
        'positive_cases_with_overlap':sum(r['tp']>0 for r in rows),
        'validation_loss_mean_of_batches':sum(losses)/len(losses),
        'all_background_loss_mean_of_cases':sum(all_background_losses)/len(all_background_losses),
        'mask_groups':grouped}
    print('CHECKPOINT_DIAGNOSIS', label, json.dumps(report['checkpoints'][label]), flush=True)
    del model, checkpoint
    torch.cuda.empty_cache()
out = Path('/kaggle/working/E3_validation_diagnosis.json')
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('E3_DIAGNOSIS_COMPLETE', json.dumps(report), flush=True)

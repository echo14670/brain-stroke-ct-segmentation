"""Render fixed E1 examples without training or changing evaluation settings."""
from pathlib import Path
import argparse
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/E1_data_quality.yaml')
    args = parser.parse_args()
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.utils import load_config, resolve_path, resolve_data_root
    from src.models import UNet2D
    from src.data.dataset import StrokeCTDataset, CaseRecord
    from src.evaluation.metrics import binary_case_metrics

    cfg = load_config(resolve_path(args.config))
    out = resolve_path(cfg['project']['output_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D(**cfg['model']).to(device)
    checkpoint = torch.load(out/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    cases = [CaseRecord(i,g) for i,g in [
        ('10793','Bleeding'),('15134','Bleeding'),('10024','Ischemia'),
        ('10101','Ischemia'),('15339','Normal'),('11606','Normal')]]
    dataset = StrokeCTDataset(root=resolve_data_root(cfg['data']['root']), cases=cases,
        image_size=cfg['data']['image_size'], mask_mode=cfg['data']['mask_mode'], training=False)
    fig, axes = plt.subplots(6,3,figsize=(10,18))
    records = []
    with torch.inference_mode():
        for i,case in enumerate(cases):
            item = dataset[i]
            image = item['image']
            target = item['mask'][0].numpy().astype('uint8')
            logits = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits)[0,0] >= cfg['evaluation']['threshold']).cpu().numpy().astype('uint8')
            metrics = binary_case_metrics(pred,target)
            records.append(dict(image_id=case.image_id, group=case.group, **metrics))
            titles = [case.key,'Reference mask',f"Prediction | Dice={metrics['dice']:.4f}"]
            for ax,data,title in zip(axes[i],[image[0].numpy(),target,pred],titles):
                ax.imshow(data,cmap='gray',vmin=0,vmax=1)
                ax.set_title(title)
                ax.axis('off')
    fig.tight_layout()
    fig.savefig(out/'samples.png',dpi=150)
    plt.close(fig)
    with (out/'rendered_sample_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('Saved:',out/'samples.png')
    print('These selected extremes illustrate errors; use full-set metrics for overall performance.')


if __name__ == '__main__':
    main()

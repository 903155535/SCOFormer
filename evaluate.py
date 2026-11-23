"""Rollout L2RE of a checkpoint on the test sets.

Pre-trained model on all twelve datasets:
    python evaluate.py --config configs/scoformer_ti.yaml --ckpt runs/scoformer_ti_pretrain/model.pth
Fine-tuned model on its dataset:
    python evaluate.py --config configs/scoformer_ti.yaml --ckpt runs/scoformer_ti_finetune_swe_pdb/model.pth \
        --datasets swe_pdb
The test set of each dataset is its first #Test trajectories (Table 2).
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from models import SCOFormer
from utils import PDEDataset, rollout_error, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--datasets', nargs='+', help='default: the twelve pre-training datasets')
    p.add_argument('--data_root', help='overrides data_root of the config')
    p.add_argument('--batch_size', type=int, default=20)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--out', help='optional JSON file for the results, in the format of results.json')
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SCOFormer(**cfg['model']).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])

    errors = {}
    for name in args.datasets or cfg['pretrain']['datasets']:
        test_set = PDEDataset([name], args.data_root or cfg['data_root'], train=False)
        loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        errors[name] = rollout_error(model, loader, device=device)
        print(f'{name:34s} {errors[name]:.5f}')
    mean = sum(errors.values()) / len(errors)
    print(f'{"mean":34s} {mean:.5f}')
    if args.out:                                             # same format as results.json of a training run
        with open(args.out, 'w') as f:
            json.dump({'test_l2re': errors, 'mean': mean}, f, indent=1)


if __name__ == '__main__':
    main()

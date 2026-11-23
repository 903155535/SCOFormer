"""Pre-training and fine-tuning of SCOFormer (Section 4.1 and Table A.1 of the paper).

Pre-training on the twelve datasets, 8 GPUs x 20 samples:
    accelerate launch --multi_gpu --num_processes 8 train.py --config configs/scoformer_ti.yaml --stage pretrain
Fine-tuning (FT-500) on one dataset, one GPU:
    python train.py --config configs/scoformer_ti.yaml --stage finetune --dataset swe_pdb \
        --init runs/scoformer_ti_pretrain/model.pth
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from models import SCOFormer
from utils import PDEDataset, rel_l2, rollout_error, load_config, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['pretrain', 'finetune'], required=True)
    p.add_argument('--dataset', help='fine-tuning dataset, one of utils.data.DATASETS')
    p.add_argument('--init', help='pre-trained checkpoint to fine-tune')
    p.add_argument('--resume', help='checkpoint of an interrupted run of the same stage')
    p.add_argument('--data_root', help='overrides data_root of the config')
    p.add_argument('--out', help='output folder, default runs/<config>_<stage>[_<dataset>]')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tc = cfg[args.stage]
    data_root = args.data_root or cfg['data_root']
    if args.stage == 'pretrain':
        names, weights = tc['datasets'], tc['weights']
    else:
        assert args.dataset and (args.init or args.resume), 'fine-tuning needs --dataset and --init'
        names, weights = [args.dataset], [1]
    tag = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or os.path.join('runs', f'{tag}_{args.stage}' + (f'_{args.dataset}' if args.dataset else ''))

    accelerator = Accelerator(mixed_precision='no')          # full precision
    if accelerator.is_main_process:
        os.makedirs(out, exist_ok=True)
        log_file = open(os.path.join(out, 'log.txt'), 'a', buffering=1)

    def log(msg):
        if accelerator.is_main_process:
            print(msg)
            log_file.write(msg + '\n')

    train_set = PDEDataset(names, data_root, train=True, weights=weights)
    test_sets = [PDEDataset([n], data_root, train=False) for n in names]
    train_loader = DataLoader(train_set, batch_size=tc['batch_size'], shuffle=True, num_workers=tc['num_workers'])
    test_loaders = [DataLoader(s, batch_size=tc['batch_size'], shuffle=False, num_workers=tc['num_workers'])
                    for s in test_sets]

    model = SCOFormer(**cfg['model'])
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location='cpu')['model'])
    # Adam with L2 weight decay. OneCycleLR also cycles beta1 between 0.85 and 0.95.
    optimizer = torch.optim.Adam(model.parameters(), lr=tc['lr'], betas=(0.9, tc['beta2']),
                                 weight_decay=tc['weight_decay'])
    # steps_per_epoch counts the whole loader: with N processes, accelerate steps the scheduler N times per update.
    scheduler = OneCycleLR(optimizer, max_lr=tc['lr'], div_factor=1e4, final_div_factor=1e4,
                           pct_start=tc['warmup_epochs'] / tc['epochs'], epochs=tc['epochs'],
                           steps_per_epoch=len(train_loader))
    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start = ckpt['epoch'] + 1

    log(f'{args.stage} | datasets {names} | train samples {len(train_set)} | params {count_parameters(model)}'
        f' | processes {accelerator.num_processes} x batch {tc["batch_size"]}')
    model, optimizer, scheduler, train_loader, *test_loaders = accelerator.prepare(
        model, optimizer, scheduler, train_loader, *test_loaders)

    for ep in range(start, tc['epochs']):
        model.train()
        t0, loss_sum, n_seen = time.time(), 0.0, 0
        for x, y, mask in train_loader:
            # Input noise of the denoising objective: per sample and channel, scaled by the L2 norm over the window
            x = x + tc['noise_scale'] * torch.sum(x ** 2, dim=(1, 2, 3), keepdim=True) ** 0.5 * torch.randn_like(x)
            loss = rel_l2(model(x), y, mask).sum()
            optimizer.zero_grad()
            accelerator.backward(loss)
            nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])
            optimizer.step()
            scheduler.step()
            loss_sum, n_seen = loss_sum + loss.item(), n_seen + x.shape[0]
        msg = (f'epoch {ep} | time {time.time() - t0:.1f}s | lr {optimizer.param_groups[0]["lr"]:.2e}'
               f' | train L2RE {loss_sum / n_seen:.5f}')

        last = ep + 1 == tc['epochs']
        if last or (ep + 1) % tc['eval_every'] == 0:
            errors = {n: rollout_error(model, l, gather=accelerator.gather_for_metrics)
                      for n, l in zip(names, test_loaders)}
            msg += ' | test L2RE ' + ' '.join(f'{v:.5f}' for v in errors.values())
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                torch.save({'model': accelerator.unwrap_model(model).state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'epoch': ep, 'config': cfg},
                           os.path.join(out, 'model.pth'))
                if last:
                    with open(os.path.join(out, 'results.json'), 'w') as f:
                        json.dump({'test_l2re': errors, 'mean': sum(errors.values()) / len(errors)}, f, indent=1)
        log(msg)


if __name__ == '__main__':
    main()

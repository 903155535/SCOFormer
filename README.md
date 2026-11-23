# SCOFormer

Code for **SCOFormer: a spectral–coordinate operator transformer for parameter-efficient PDE pre-training**
(Expert Systems with Applications, 2026).

SCOFormer is an auto-regressive neural operator for 2D time-dependent PDEs. Each HybridBlock adds a grouped
factorized Fourier layer (GFFNO) and a coordinate-attention branch. In the first block, a kernel router mixes K basis
spectral kernels per input. The models are pre-trained on twelve datasets with an auto-regressive denoising objective
and then fine-tuned on each dataset.

This repository reproduces the main results (Table 4 of the paper) for SCO-Ti, SCO-S, and SCO-M:
pre-training on the twelve datasets and 500-epoch fine-tuning (FT-500) on each of them.
Pre-trained weights will be released later.

## Installation

```bash
pip install -r requirements.txt     # PyTorch >= 2.0 with CUDA
```

## Data

The twelve datasets come from four public benchmarks. Download the source files:

| Source | Download | Files |
| --- | --- | --- |
| FNO (Li et al., 2021) | [Google Drive](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) | `ns_V1e-5_N1200_T20`, `ns_V1e-4_N10000_T30`, `ns_V1e-3_N5000_T50` (prefix `ns_` or `NavierStokes_`, `.mat` or `.h5`) |
| PDEBench (Takamoto et al., 2022) | [DaRUS](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) | `2D_CFD_Rand_M{1.0,0.1}_Eta{0.1,0.01}_Zeta{0.1,0.01}_periodic_128_Train.hdf5`, `2D_rdb_NA_NA.h5`, `2D_diff-react_NA_NA.h5` |
| PDEArena (Gupta & Brandstetter, 2023) | [PDEArena](https://microsoft.github.io/pdearena/datadownload/) | folders `NavierStokes-2D`, `NavierStokes-2D-conditoned` |
| CFDBench (Luo et al., 2023) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/435413b55dea434297d1/) | `cavity.zip`, `tube.zip`, `cylinder/{bc,geo,prop}.zip` |

Convert them:

```bash
python data_generation/prepare_data.py fno      --raw <FNO files>          --out ./data
python data_generation/prepare_data.py pdebench --raw <PDEBench 2D folder>  --out ./data
python data_generation/prepare_data.py pdearena --raw <PDEArena folder>     --out ./data
python data_generation/prepare_cfdbench.py --source-root <CFDBench folder> --target-root ./data/cfdbench
```

This gives the following layout:

```
data/
├── ns2d_fno/ns2d_{1e-5,1e-4,1e-3}_{train,test}.hdf5          # [N, 64, 64, T]
├── pdebench/<dataset>/{train,test}/data_<i>.hdf5               # [128, 128, T, C], one file per trajectory
├── pdearena/{ns2d_pda,ns2d_cond_pda}/{train,test}/data_<i>.hdf5
└── cfdbench/ns2d_cdb_{train,test}.hdf5                         # [N, 64, 64, 20, 3]
```

Dataset names in the code (`utils/data.py`, Table 2 of the paper):

| Name | Dataset | Train / Test | Rollout | Weight w |
| --- | --- | --- | --- | --- |
| `ns2d_fno_1e-5` | FNO incompressible NS, ν = 1e-5 | 1000 / 200 | 10 | 8 |
| `ns2d_fno_1e-4` | FNO incompressible NS, ν = 1e-4 | 9800 / 200 | 20 | 2 |
| `ns2d_fno_1e-3` | FNO incompressible NS, ν = 1e-3 | 1000 / 200 | 20 | 8 |
| `ns2d_pdb_M1_eta1e-1_zeta1e-1` | PDEBench compressible NS, M = 1, η = ζ = 0.1 | 9000 / 200 | 11 | 1 |
| `ns2d_pdb_M1_eta1e-2_zeta1e-2` | PDEBench compressible NS, M = 1, η = ζ = 0.01 | 9000 / 200 | 11 | 1 |
| `ns2d_pdb_M1e-1_eta1e-1_zeta1e-1` | PDEBench compressible NS, M = 0.1, η = ζ = 0.1 | 9000 / 200 | 11 | 1 |
| `ns2d_pdb_M1e-1_eta1e-2_zeta1e-2` | PDEBench compressible NS, M = 0.1, η = ζ = 0.01 | 9000 / 200 | 11 | 1 |
| `swe_pdb` | PDEBench shallow water (SWE) | 900 / 60 | 91 | 20 |
| `dr_pdb` | PDEBench diffusion–reaction (DR) | 900 / 60 | 91 | 20 |
| `ns2d_pda` | PDEArena incompressible NS (NS) | 6500 / 650 | 4 | 1 |
| `ns2d_cond_pda` | PDEArena force-conditioned NS (NS-c) | 3100 / 200 | 46 | 3 |
| `cfdbench` | CFDBench (CFD) | 9000 / 1000 | 10 | 1 |

Training and testing use the first trajectories of each split. All trajectories are resized to 128 × 128, and the
physical channels are padded with ones to four channels. Predictions of 64 × 64 data are scored at every second
grid point. CFDBench scores u and v; it does not score the flow-region mask.

## Pre-training

```bash
bash scripts/pretrain.sh ti        # or s, m; 8 GPUs x 20 samples = 160 samples per update
```

This runs `accelerate launch --multi_gpu --num_processes 8 --mixed_precision no train.py --config configs/scoformer_ti.yaml --stage pretrain`.
The run trains for 1000 epochs with Adam (weight decay 1e-6, β2 = 0.9). The learning rate follows a one-cycle schedule
with peak 1e-3 and 200 warm-up epochs, and β1 cycles between 0.85 and 0.95. The input noise scale is ε = 5e-4 and
the gradient-clipping threshold is 1e4. Training runs in full precision.

In each epoch, dataset d is repeated w_d times, which gives 132,400 samples. Every `eval_every` epochs, the run
evaluates the twelve test sets and saves `runs/scoformer_<size>_pretrain/model.pth`. `--resume <model.pth>`
continues an interrupted run. With fewer GPUs, set `NGPU` and scale `batch_size` in the config to keep 160 samples
per update.

## Fine-tuning

```bash
GPUS="0 1 2 3" bash scripts/finetune_all.sh ti      # 12 runs, one dataset and one GPU each
```

Each run fine-tunes all parameters of the pre-trained model on one dataset for 500 epochs. The batch size is 32,
with 40 warm-up epochs and no input noise; the optimizer and schedule are the same as in pre-training. A single run is

```bash
python train.py --config configs/scoformer_ti.yaml --stage finetune --dataset swe_pdb \
    --init runs/scoformer_ti_pretrain/model.pth
```

## Evaluation

The test error is the L2RE of the full auto-regressive rollout. The model takes the first 10 frames and predicts the
rollout length of the table above. The error of each channel is ‖û_c − u_c‖₂ / ‖u_c‖₂ over all scored grid points
and frames. It is averaged over the scored channels and then over the test trajectories.

Each training run writes this error at its last epoch to `results.json` in its output folder. `summarize.py`
collects these files:

```bash
python scripts/summarize.py ti     # pre-trained and fine-tuned errors in the column order of Table 4
```

To evaluate a saved checkpoint again (`--out` writes a JSON file in the format of `results.json`):

```bash
python evaluate.py --config configs/scoformer_ti.yaml --ckpt runs/scoformer_ti_pretrain/model.pth
```

## Results

L2RE from Table 4 of the paper. The main runs did not fix random seeds, so new runs differ slightly.

| Model | 1e-5 | 1e-4 | 1e-3 | (1, 0.1) | (1, 0.01) | (0.1, 0.1) | (0.1, 0.01) | DR | SWE | NS | NS-c | CFD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCO-Ti, pre-trained | 0.0625 | 0.0515 | 0.00861 | 0.0147 | 0.0339 | 0.0112 | 0.0169 | 0.0486 | 0.00404 | 0.118 | 0.361 | 0.01008 |
| SCO-S, pre-trained | 0.0458 | 0.0450 | 0.00701 | 0.0121 | 0.0259 | 0.00948 | 0.0135 | 0.0318 | 0.00306 | 0.0973 | 0.315 | 0.00989 |
| SCO-M, pre-trained | 0.0361 | 0.0318 | 0.00485 | 0.0115 | 0.0238 | 0.01015 | 0.0145 | 0.0248 | 0.00301 | 0.0825 | 0.275 | 0.00898 |
| SCO-Ti, FT-500 | 0.0379 | 0.0280 | 0.00539 | 0.00710 | 0.0106 | 0.00844 | 0.00625 | 0.0208 | 0.00242 | 0.0594 | 0.295 | 0.00480 |
| SCO-S, FT-500 | 0.0271 | 0.00963 | 0.00446 | 0.00686 | 0.00848 | 0.00795 | 0.00574 | 0.0173 | 0.00160 | 0.0400 | 0.236 | 0.00480 |
| SCO-M, FT-500 | 0.0231 | 0.00957 | 0.00423 | 0.00722 | 0.00823 | 0.01023 | 0.00550 | 0.0145 | 0.00219 | 0.0308 | 0.200 | 0.00391 |

The first three columns are the FNO viscosities ν, and the next four are the PDEBench compressible NS settings (M, η).

## Model sizes

| Config | C | L | K | G | r | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| `configs/scoformer_ti.yaml` | 256 | 6 | 4 | 4 | 1 | 8.6 M |
| `configs/scoformer_s.yaml` | 512 | 8 | 6 | 8 | 1 | 30 M |
| `configs/scoformer_m.yaml` | 1024 | 12 | 8 | 16 | 1 | 121 M |

C is the embedding width, L the number of HybridBlocks, K the number of basis kernels in the first block, G the
number of GFFNO channel groups, and r the FFN expansion ratio. All sizes keep m = 9 Fourier modes per axis, which
is the full axial spectrum of the 16 × 16 tokens.

## Code structure

```
models/scoformer.py          SCOFormer: patch embedding, temporal aggregation, kernel router, HybridBlock (GFFNO + coordinate attention), head
utils/data.py                dataset list and trajectory dataset
utils/common.py              L2RE, rollout evaluation, config loading
train.py                     pre-training (multi-GPU, accelerate) and fine-tuning
evaluate.py                  rollout evaluation of a checkpoint
configs/                     one config per model size
scripts/                     pre-training, fine-tuning of all twelve datasets, result summary
data_generation/             conversion of the source datasets
```

## Acknowledgements

This code builds on [DPOT](https://github.com/HaoZhongkai/DPOT). We thank the authors of DPOT and of the FNO,
PDEBench, PDEArena, and CFDBench datasets.

## Citation

```bibtex
@article{li2026scoformer,
  title   = {{SCOFormer}: a spectral--coordinate operator transformer for parameter-efficient {PDE} pre-training},
  author  = {Li, Qixuan and Yao, Shunchun and Liang, Xiaojun and Zhang, Xiaoning and Yang, Chunhua and Gui, Weihua and Luo, Weichao},
  journal = {Expert Systems with Applications},
  year    = {2026}
}
```

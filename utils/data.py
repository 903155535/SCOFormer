"""The twelve pre-training datasets (Table 2) and the trajectory dataset used for training and testing.

Each trajectory is stored as `data` with shape [X, Y, T] or [X, Y, T, C], either
all in one HDF5 file ([N, X, Y, T(, C)]) or one file per trajectory (`<folder>/data_<i>.hdf5`).
Paths are relative to `data_root`; data_generation/ writes this layout.
"""
import os
from collections import namedtuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

Spec = namedtuple('Spec', 'train test n_train n_test rollout eval_channels')
# eval_channels: number of leading channels scored at test time (None = all). CFDBench scores u and v, not the mask.
DATASETS = {
    'ns2d_fno_1e-5': Spec('ns2d_fno/ns2d_1e-5_train.hdf5', 'ns2d_fno/ns2d_1e-5_test.hdf5', 1000, 200, 10, None),
    'ns2d_fno_1e-4': Spec('ns2d_fno/ns2d_1e-4_train.hdf5', 'ns2d_fno/ns2d_1e-4_test.hdf5', 9800, 200, 20, None),
    'ns2d_fno_1e-3': Spec('ns2d_fno/ns2d_1e-3_train.hdf5', 'ns2d_fno/ns2d_1e-3_test.hdf5', 1000, 200, 20, None),
    'ns2d_pdb_M1_eta1e-1_zeta1e-1': Spec('pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1/train', 'pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1/test', 9000, 200, 11, None),
    'ns2d_pdb_M1_eta1e-2_zeta1e-2': Spec('pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2/train', 'pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2/test', 9000, 200, 11, None),
    'ns2d_pdb_M1e-1_eta1e-1_zeta1e-1': Spec('pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1/train', 'pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1/test', 9000, 200, 11, None),
    'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2': Spec('pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/train', 'pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/test', 9000, 200, 11, None),
    'swe_pdb': Spec('pdebench/swe_pdb/train', 'pdebench/swe_pdb/test', 900, 60, 91, None),
    'dr_pdb': Spec('pdebench/dr_pdb/train', 'pdebench/dr_pdb/test', 900, 60, 91, None),
    'ns2d_pda': Spec('pdearena/ns2d_pda/train', 'pdearena/ns2d_pda/test', 6500, 650, 4, None),
    'ns2d_cond_pda': Spec('pdearena/ns2d_cond_pda/train', 'pdearena/ns2d_cond_pda/test', 3100, 200, 46, None),
    'cfdbench': Spec('cfdbench/ns2d_cdb_train.hdf5', 'cfdbench/ns2d_cdb_test.hdf5', 9000, 1000, 10, 2),
}


class PDEDataset(Dataset):
    """Trajectories of one or more datasets, resized to res x res and padded to n_channels with ones.

    train=True : a random window of t_in frames and the next t_ar frames. Dataset d is repeated weights[d] times.
    train=False: the first t_in frames and the whole rollout target, with the mask of evaluated points and channels.
    Items are (x [X, Y, t_in, C], y [X, Y, T_out, C], mask [X, Y, 1, C]).
    """

    def __init__(self, names, data_root, train=True, n_list=None, weights=None, res=128, t_in=10, t_ar=1, n_channels=4):
        self.names = list(names)
        self.specs = [DATASETS[n] for n in self.names]
        self.paths = [os.path.join(data_root, s.train if train else s.test) for s in self.specs]
        self.train, self.res, self.t_in, self.t_ar, self.n_channels = train, res, t_in, t_ar, n_channels
        self.sizes = n_list or [s.n_train if train else s.n_test for s in self.specs]
        self.weights = weights or [1] * len(self.names)
        self.cumulative = np.cumsum([n * w for n, w in zip(self.sizes, self.weights)])
        self._files = {}                                     # opened lazily, once per worker process

    def __len__(self):
        return int(self.cumulative[-1])

    def _read(self, d, i):
        path = self.paths[d]
        if not path.endswith('.hdf5'):                       # one file per trajectory
            with h5py.File(os.path.join(path, f'data_{i}.hdf5'), 'r') as f:
                return f['data'][:]
        if d not in self._files:
            self._files[d] = h5py.File(path, 'r')
        return self._files[d]['data'][i][:]

    def _pad(self, x):
        """Bilinear resize to res x res, then pad channels with ones: [H, W, T, C] -> [res, res, T, n_channels]."""
        H, W, T, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1)
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res), mode='bilinear').squeeze(0).permute(1, 2, 0)
        x = x.view(*x.shape[:2], T, C)
        out = torch.ones([*x.shape[:-1], self.n_channels])
        out[..., :C] = x
        return out

    @staticmethod
    def _eval_mask(x, orig_size):
        """Score only the original grid points (every k-th point after upsampling) and the evaluated channels."""
        mask = torch.zeros(*x.shape[:2], 1, x.shape[-1])
        kx, ky = max(x.shape[0] // orig_size[0], 1), max(x.shape[1] // orig_size[1], 1)
        mask[::kx, ::ky, :, :orig_size[-1]] = 1
        return mask

    def __getitem__(self, idx):
        d = int(np.searchsorted(self.cumulative, idx + 1))
        i = (idx - (self.cumulative[d - 1] if d > 0 else 0)) // self.weights[d]
        sample = torch.from_numpy(self._read(d, i)).float()
        if sample.ndim == 3:
            sample = sample.unsqueeze(-1)
        orig_size = list(sample.shape)
        orig_size[-1] = self.specs[d].eval_channels or orig_size[-1]
        sample = self._pad(sample)

        if self.train:
            T = sample.shape[-2]
            s = np.random.randint(max(T - (self.t_in + self.t_ar) + 1, 1))
            x, y = sample[..., s:s + self.t_in, :], sample[..., s + self.t_in:min(s + self.t_in + self.t_ar, T), :]
            mask = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else:
            x, y = sample[..., :self.t_in, :], sample[..., self.t_in:self.t_in + self.specs[d].rollout, :]
            mask = self._eval_mask(sample, orig_size)
        return x, y, mask

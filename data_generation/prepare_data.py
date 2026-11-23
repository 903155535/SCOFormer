"""Convert the FNO, PDEBench, and PDEArena source files to the layout of utils/data.py.

    python data_generation/prepare_data.py fno      --raw <folder with the three FNO files>  --out ./data
    python data_generation/prepare_data.py pdebench --raw <PDEBench 2D folder>               --out ./data
    python data_generation/prepare_data.py pdearena --raw <PDEArena folder>                  --out ./data

CFDBench has its own script, prepare_cfdbench.py. Each trajectory is stored as `data` with shape [X, Y, T(, C)].
"""
import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm


def fno_names(stem):
    """Accepted names of an FNO file: prefix ns_ or NavierStokes_, extension .mat or .h5."""
    return tuple(prefix + stem + ext for prefix in ('ns_', 'NavierStokes_') for ext in ('.mat', '.h5'))


# FNO vorticity data (Li et al., 2021): accepted file names, trajectories N, frames kept, training trajectories
FNO = {
    '1e-5': (fno_names('V1e-5_N1200_T20'), 1200, 20, 1000),
    '1e-4': (fno_names('V1e-4_N10000_T30'), 10000, 30, 9800),
    '1e-3': (fno_names('V1e-3_N5000_T50'), 5000, 50, 4800),
}
# PDEBench 2D compressible NS (Takamoto et al., 2022): channels Vx, Vy, density, pressure; 9000 train / 1000 test
CNS = {
    'ns2d_pdb_M1_eta1e-1_zeta1e-1': '2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5',
    'ns2d_pdb_M1_eta1e-2_zeta1e-2': '2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
    'ns2d_pdb_M1e-1_eta1e-1_zeta1e-1': '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5',
    'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2': '2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
}
# PDEBench shallow water and diffusion-reaction: 1000 groups '0000'...'0999' with data [T, X, Y, C]; 900 / 100
GROUPED = {'swe_pdb': '2D_rdb_NA_NA.h5', 'dr_pdb': '2D_diff-react_NA_NA.h5'}
# PDEArena (Gupta & Brandstetter, 2023): channels u, vx, vy; train and valid files form the training split
PDEARENA = {'ns2d_pda': 'NavierStokes-2D', 'ns2d_cond_pda': 'NavierStokes-2D-conditoned'}
# This file was incomplete in the download used for the paper. Skipping it keeps the training split identical.
PDEARENA_SKIP = {'NavierStokes2D_train_496019_0.38774_32.h5'}


def find(root, *names):
    for r, _, files in os.walk(root):
        for name in names:
            if name in files:
                return os.path.join(r, name)
    raise FileNotFoundError(f'{names} not found under {root}')


def read_fno(path):
    """Reads the vorticity u from an HDF5 file (including MATLAB v7.3) or a MATLAB v5 file."""
    try:
        u = np.array(h5py.File(path, 'r')['u'])
    except OSError:                                          # MATLAB v5 file
        import scipy.io
        u = scipy.io.loadmat(path)['u']
    return u


def save(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=array)


def prepare_fno(raw, out):
    for nu, (names, n, frames, n_train) in FNO.items():
        u = read_fno(find(raw, *names))
        # u stored with N first is used as is; u stored with N last (MATLAB v7.3 and HDF5 copies) is reversed to N first
        u = (u if u.shape[0] == n else u.transpose(3, 1, 2, 0))[..., :frames]
        save(f'{out}/ns2d_fno/ns2d_{nu}_train.hdf5', u[:n_train])
        save(f'{out}/ns2d_fno/ns2d_{nu}_test.hdf5', u[n_train:])
        print(f'FNO nu={nu}: train {u[:n_train].shape}, test {u[n_train:].shape}')


def prepare_pdebench(raw, out):
    for target, name in CNS.items():
        with h5py.File(find(raw, name), 'r') as f:
            n = f['Vx'].shape[0]
            for i in tqdm(range(n), desc=target):
                traj = np.stack([np.asarray(f[k][i], dtype=np.float32) for k in ('Vx', 'Vy', 'density', 'pressure')], -1)
                split, j = ('train', i) if i < 9 * n // 10 else ('test', i - 9 * n // 10)
                save(f'{out}/pdebench/{target}/{split}/data_{j}.hdf5', traj.transpose(1, 2, 0, 3))
    for target, name in GROUPED.items():
        with h5py.File(find(raw, name), 'r') as f:
            for i in tqdm(range(len(f.keys())), desc=target):
                split, j = ('train', i) if i < 900 else ('test', i - 900)
                save(f'{out}/pdebench/{target}/{split}/data_{j}.hdf5', f[f'{i:04d}/data'][:].transpose(1, 2, 0, 3))


def prepare_pdearena(raw, out):
    for target, folder in PDEARENA.items():
        files = sorted(os.path.join(r, n) for r, _, fs in os.walk(os.path.join(raw, folder)) for n in fs
                       if n.endswith('.h5') and n not in PDEARENA_SKIP)
        count = {'train': 0, 'test': 0}
        for path in tqdm(files, desc=target):
            key = next(k for k in ('test', 'train', 'valid') if k in os.path.basename(path))
            split = 'test' if key == 'test' else 'train'
            with h5py.File(path, 'r') as f:
                trajs = np.stack([f[key]['u'][:], f[key]['vx'][:], f[key]['vy'][:]], -1).transpose(0, 2, 3, 1, 4)
            for traj in trajs:
                save(f'{out}/pdearena/{target}/{split}/data_{count[split]}.hdf5', traj)
                count[split] += 1
        print(f'{target}: {count}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', choices=['fno', 'pdebench', 'pdearena'])
    p.add_argument('--raw', required=True, help='folder with the downloaded source files')
    p.add_argument('--out', default='./data')
    a = p.parse_args()
    {'fno': prepare_fno, 'pdebench': prepare_pdebench, 'pdearena': prepare_pdearena}[a.source](a.raw, a.out)

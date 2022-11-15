import argparse
import logging
import os

import h5py
import numpy as np
import torch
import tqdm

logging.basicConfig(
    format='[%(asctime)s] %(pathname)s:%(lineno)d %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data',
                    help='The directory in which the data are stored.')
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    logger.error(
        f'Data directory "{args.data_dir}" does not exist. Consider running "python download.py --data_dir {args.data_dir}" first.')
    exit(1)

os.makedirs(os.path.join(args.data_dir, '1x'))
os.makedirs(os.path.join(args.data_dir, '1ccx'))

file_path = os.path.join(args.data_dir, 'ani.h5')
logger.info(f'Preprocessing {file_path}...')
with h5py.File(file_path, 'r') as f:
    for name, molecule in tqdm.tqdm(f.items(), leave=False):
        directory_1x = os.path.join(args.data_dir, '1x', name)
        directory_1ccx = os.path.join(args.data_dir, '1ccx', name)

        atomic_numbers = torch.Tensor(np.array(molecule['atomic_numbers']))
        coordinates = torch.Tensor(np.array(molecule['coordinates']))
        dft_energies = torch.Tensor(np.array(molecule['wb97x_dz.energy']))
        ccsd_energies = torch.Tensor(np.array(molecule['ccsd(t)_cbs.energy']))

        os.makedirs(directory_1x)
        if torch.isnan(ccsd_energies).sum() < ccsd_energies.shape[0]:
            os.makedirs(directory_1ccx)

        ccsd_i = 0
        for i in range(coordinates.shape[0]):
            torch.save({
                'atomic_numbers': atomic_numbers,
                'coordinates': coordinates[i],
                'energy': dft_energies[i],
            }, os.path.join(directory_1x, f'{i}.pt'))

            if not torch.isnan(ccsd_energies[i]):
                torch.save({
                    'atomic_numbers': atomic_numbers,
                    'coordinates': coordinates[i],
                    'energy': dft_energies[i],
                }, os.path.join(directory_1ccx, f'{ccsd_i}.pt'))
                ccsd_i += 1
logger.info(f'Successfully preprocessed {file_path}.')

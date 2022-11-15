import argparse
import logging
import os

import h5py
import numpy as np
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

file_path = os.path.join(args.data_dir, 'ani.h5')
file_path_1x = os.path.join(args.data_dir, 'ani1x.h5')
file_path_1ccx = os.path.join(args.data_dir, 'ani1ccx.h5')

logger.info(f'Preprocessing {file_path}...')
with h5py.File(file_path, 'r') as f:
    with h5py.File(file_path_1x, 'w') as f_1x:
        with h5py.File(file_path_1ccx, 'w') as f_1ccx:
            for name, molecule in tqdm.tqdm(f.items(), leave=False):
                atomic_numbers = molecule['atomic_numbers']
                coordinates = molecule['coordinates']
                dft_energies = molecule['wb97x_dz.energy']
                ccsd_energies = molecule['ccsd(t)_cbs.energy']

                group_1x = f_1x.create_group(name)
                group_1x.create_dataset('atomic_numbers', data=atomic_numbers)
                group_1x.create_dataset('coordinates', data=coordinates)
                group_1x.create_dataset('energy', data=dft_energies)

                ccsd_nan = np.isnan(ccsd_energies)
                if ccsd_nan.sum() < ccsd_energies.shape[0]:
                    group_1ccx = f_1ccx.create_group(name)
                    group_1ccx.create_dataset(
                        'atomic_numbers', data=atomic_numbers)
                    group_1ccx.create_dataset(
                        'coordinates', data=coordinates[~ccsd_nan])
                    group_1ccx.create_dataset(
                        'energy', data=ccsd_energies[~ccsd_nan])
logger.info(f'Successfully preprocessed {file_path}.')

import argparse
import gzip
import logging
import os
import shutil

import requests
import tqdm


logging.basicConfig(
    format='[%(asctime)s] %(pathname)s:%(lineno)d %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--clear', action='store_true',
                    help='When present, clear the data directory.')
parser.add_argument('--data_dir', type=str, default='data',
                    help='The directory in which to download the data.')
parser.add_argument('--dataset', type=str, default=['1x'], nargs='+',
                    choices=['1x', '1ccx'], help='The dataset to download.')
args = parser.parse_args()

if os.path.exists(args.data_dir):
    if args.clear:
        logger.info(f'Clearing data directory...')
        shutil.rmtree(args.data_dir)
        os.makedirs(args.data_dir)
else:
    os.makedirs(args.data_dir)

URLS = {
    '1x': 'https://zenodo.org/record/4081694/files/292.hdf5.gz?download=1',
    '1ccx': 'https://zenodo.org/record/4081692/files/293.hdf5.gz?download=1'
}

for dataset in args.dataset:
    file_path = os.path.join(args.data_dir, f'ani-{dataset}.hdf5.gz')
    logger.info(f'Downloading {URLS[dataset]} to {file_path}...')
    r = requests.get(URLS[dataset], stream=True)
    if r.status_code == 200:
        file_size = int(r.headers.get('Content-Length', 0))
        progress_bar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(1024)
        progress_bar.close()
        logger.info(f'Downloaded {URLS[dataset]} to {file_path}.')
        logger.info(f'Unzipping {file_path}...')
        with gzip.open(file_path, 'rb') as f_in:
            with open(file_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f'Successfully unzipped {file_path}.')
        logger.info(f'Removing {file_path}...')
        os.remove(file_path)
    else:
        logger.error(
            f'Error downloading dataset {dataset} to {URLS[dataset]}.')

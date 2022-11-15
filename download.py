import argparse
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
args = parser.parse_args()

if os.path.exists(args.data_dir):
    if args.clear:
        logger.info(f'Clearing data directory...')
        shutil.rmtree(args.data_dir)
        os.makedirs(args.data_dir)
else:
    os.makedirs(args.data_dir)

URL = 'https://springernature.figshare.com/ndownloader/files/18112775'

file_path = os.path.join(args.data_dir, f'ani.h5')
logger.info(f'Downloading {URL} to {file_path}...')
r = requests.get(URL, stream=True)
if r.status_code == 200:
    file_size = int(r.headers.get('Content-Length', 0))
    progress_bar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(1024)
    progress_bar.close()
    logger.info(f'Successfully downloaded {URL} to {file_path}.')
else:
    logger.error(
        f'Error downloading {URL} to {file_path}.')

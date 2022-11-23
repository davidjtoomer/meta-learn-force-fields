import argparse
import logging
import os

import numpy as np
import torch
from torch.utils import tensorboard

from meta_learn_force_fields.data import train_val_test_split
from meta_learn_force_fields.models import FeaturizerConfig, MAML


logging.basicConfig(
    format='[%(asctime)s] %(pathname)s:%(lineno)d %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='1ccx',
                    choices=['1x', '1ccx'], help='The dataset on which to train.')
parser.add_argument('--data_dir', type=str, default='data',
                    help='The directory in which to download the data.')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='The directory in which to store logs.')
parser.add_argument('--log_interval', type=int, default=1,
                    help='The number of epochs between logging to stdout.')
parser.add_argument('--inner_lr', type=float, default=0.1,
                    help='The inner loop learning rate.')
parser.add_argument('--outer_lr', type=float, default=0.001,
                    help='The outer loop learning rate.')
parser.add_argument('--learn_inner_lr', action='store_true',
                    help='When present, learn the inner loop learning rates.')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='The number of epochs to train for.')
parser.add_argument('--num_tasks_per_epoch', type=int,
                    default=160, help='The number of tasks per epoch.')
parser.add_argument('--batch_size', type=int,
                    default=16, help='The batch size.')
parser.add_argument('--num_support', type=int, default=5,
                    help='The number of support examples.')
parser.add_argument('--num_query', type=int, default=1,
                    help='The number of query examples.')
parser.add_argument('--num_inner_steps', type=int, default=0,
                    help='The number of inner loop updates.')
parser.add_argument('--train_frac', type=float, default=0.6,
                    help='The percentage of the dataset to use for training.')
args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
writer = tensorboard.SummaryWriter(args.log_dir)

logger.info('Loading data...')
file_path = os.path.join(args.data_dir, f'ani{args.dataset}.h5')
if not os.path.exists(file_path):
    logger.error(
        'The dataset does not exist. Please check your arguments, or run download_data.py.')
    exit(1)
train_dataloader, val_dataloader, test_dataloader = train_val_test_split(
    file_path,
    train_frac=args.train_frac,
    batch_size=args.batch_size,
    num_support=args.num_support,
    num_query=args.num_query,
    num_tasks_per_epoch=args.num_tasks_per_epoch)
logger.info(f'Successfully loaded data from {file_path}.')

logger.info('Creating model...')
featurizer_config = FeaturizerConfig(
    g1_param_ranges={
        'cutoff_radius': [10.0]
    },
    g2_param_ranges={
        'cutoff_radius': [10.0],
        'center_radius': np.linspace(0.0, 8.0, 16),
        'eta': np.logspace(-4, 0, 5)
    },
    g3_param_ranges={
        'cutoff_radius': [10.0],
        'kappa': np.linspace(1.0, 4.0, 4),
    },
    g4_param_ranges={
        'cutoff_radius': [10.0],
        'eta': np.logspace(-4, 0, 5),
        'zeta': np.linspace(1.0, 4.0, 4),
        'lambda_': [-1.0, 1.0]
    },
    g5_param_ranges={
        'cutoff_radius': [10.0],
        'eta': np.logspace(-4, 0, 5),
        'zeta': np.linspace(1.0, 4.0, 4),
        'lambda_': [-1.0, 1.0]
    })
num_features = featurizer_config.num_features + 1

mlp_layers = [num_features, 512, 128, 64, 1]
model = MAML(
    mlp_layers,
    featurizer_config,
    num_inner_steps=args.num_inner_steps,
    inner_lr=args.inner_lr,
    learn_inner_lr=args.learn_inner_lr,
    outer_lr=args.outer_lr)
logger.info('Successfully created model.')

logger.info('Training model...')


def run_one_epoch(dataloader: torch.utils.data.DataLoader, optimizer, train: bool = True):
    pre_adapt_support_losses = []
    post_adapt_support_losses = []
    post_adapt_query_losses = []
    for batch in dataloader:
        if train:
            optimizer.zero_grad()
        outer_loss, support_losses = model.outer_loop(batch, train=train)
        if train:
            outer_loss.backward()
            optimizer.step()

        pre_adapt_support_losses.append(support_losses[0].item())
        post_adapt_support_losses.append(support_losses[-1].item())
        post_adapt_query_losses.append(outer_loss.item())

    return np.mean(pre_adapt_support_losses), np.mean(post_adapt_support_losses), np.mean(post_adapt_query_losses)


optimizer = torch.optim.Adam(
    list(model.meta_parameters.values()) + list(model.inner_lrs.values()),
    lr=args.outer_lr)

for epoch in range(args.num_epochs):
    pre_adapt_support_loss_train, post_adapt_support_loss_train, post_adapt_query_loss_train = run_one_epoch(
        train_dataloader, optimizer, train=True)
    writer.add_scalar('train/pre_adapt_support_MAE',
                      pre_adapt_support_loss_train, epoch)
    writer.add_scalar('train/post_adapt_support_MAE',
                      post_adapt_support_loss_train, epoch)
    writer.add_scalar('train/post_adapt_query_MAE',
                      post_adapt_query_loss_train, epoch)

    pre_adapt_support_loss_val, post_adapt_support_loss_val, post_adapt_query_loss_val = run_one_epoch(
        val_dataloader, optimizer, train=False)
    writer.add_scalar('val/pre_adapt_support_MAE',
                      pre_adapt_support_loss_val, epoch)
    writer.add_scalar('val/post_adapt_support_MAE',
                      post_adapt_support_loss_val, epoch)
    writer.add_scalar('val/post_adapt_query_MAE',
                      post_adapt_query_loss_val, epoch)

    if epoch % args.log_interval == 0:
        logger.info(
            f'Epoch {epoch} TRAIN.\t'
            f'Pre-adapt support MAE: {pre_adapt_support_loss_train}. '
            f'Post-adapt support MAE: {post_adapt_support_loss_train}. '
            f'Post-adapt query MAE: {post_adapt_query_loss_train}.')
        logger.info(
            f'Epoch {epoch} VALID.\t'
            f'Pre-adapt support MAE: {pre_adapt_support_loss_val}. '
            f'Post-adapt support MAE: {post_adapt_support_loss_val}. '
            f'Post-adapt query MAE: {post_adapt_query_loss_val}.')

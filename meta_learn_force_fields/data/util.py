import torch

from .dataset import ANIDataset
from .sampler import ANISampler


def identity(x):
    return x


def train_val_test_split(
        file_path: str,
        train_frac: float = 0.6,
        batch_size: int = 1,
        num_support: int = 1,
        num_query: int = 1,
        num_tasks_per_epoch: int = 10):
    dataset = ANIDataset(
        file_path, num_support=num_support, num_query=num_query)
    n = len(dataset)
    train_n = int(n * train_frac)
    val_n = int(n * (1 - train_frac) / 2)

    train_indices = range(train_n)
    val_indices = range(train_n, train_n + val_n)
    test_indices = range(train_n + val_n, n)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=ANISampler(train_indices, num_tasks_per_epoch),
        collate_fn=identity,
        drop_last=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=ANISampler(val_indices, num_tasks_per_epoch),
        collate_fn=identity,
        drop_last=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=ANISampler(test_indices, num_tasks_per_epoch),
        collate_fn=identity,
        drop_last=True
    )
    return train_dataloader, val_dataloader, test_dataloader

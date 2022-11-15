import random

import h5py
import numpy as np
import torch


class ANIDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, num_support: int = 1, num_query: int = 1):
        self.file_path = file_path
        self.num_support = num_support
        self.num_query = num_query

        self.data = None
        with h5py.File(file_path, 'r') as f:
            self.keys = list(f.keys())
            random.shuffle(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        if self.data is None:
            self.data = h5py.File(self.file_path, 'r')

        key = self.keys[idx]
        atomic_numbers = torch.Tensor(
            np.array(self.data[key]['atomic_numbers']))
        coordinates = torch.Tensor(np.array(self.data[key]['coordinates']))
        energies = torch.Tensor(np.array(self.data[key]['energy']))

        indices = np.random.default_rng().choice(
            len(energies), size=self.num_support + self.num_query, replace=False)

        support_indices = indices[:self.num_support]
        query_indices = indices[self.num_support:]

        return atomic_numbers, coordinates[support_indices], energies[support_indices], coordinates[query_indices], energies[query_indices]

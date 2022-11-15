import numpy as np
import torch


class ANISampler(torch.utils.data.Sampler):
    def __init__(self, indices: range, num_tasks_per_epoch: int):
        self.indices = indices
        self.num_tasks_per_epoch = num_tasks_per_epoch

    def __iter__(self):
        return (np.random.default_rng().choice(self.indices, replace=False) for _ in range(self.num_tasks_per_epoch))

    def __len__(self):
        return self.num_tasks_per_epoch

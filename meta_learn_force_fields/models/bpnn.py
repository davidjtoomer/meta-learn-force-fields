from typing import List

import torch

from .featurizer import Featurizer, FeaturizerConfig


class BPNN(torch.nn.Module):
    def __init__(
            self,
            num_layers: int,
            atomic_numbers: List[int],
            feature_config: FeaturizerConfig):
        super().__init__()
        self.num_layers = num_layers
        self.atomic_numbers = atomic_numbers
        self.feature_config = feature_config

        self.featurizer = Featurizer(self.feature_config)

    def forward(
            self,
            atomic_numbers: torch.Tensor,
            coordinates: torch.Tensor,
            parameters: dict) -> torch.Tensor:
        x = self.featurizer(coordinates)
        energies = torch.empty(coordinates.shape[0], coordinates.shape[1])
        for atomic_number in self.atomic_numbers:
            indices = torch.where(atomic_numbers == atomic_number)[0]
            atoms = x[:, indices]
            for i in range(self.num_layers):
                atoms = torch.nn.functional.linear(
                    atoms,
                    parameters[f'atom_{atomic_number}_weight_{i}'],
                    parameters[f'atom_{atomic_number}_bias_{i}'])
                if i < self.num_layers - 1:
                    atoms = torch.nn.functional.silu(atoms)
            atoms = atoms.squeeze(-1)
            energies[:, indices] = atoms
        return energies.sum(dim=-1)

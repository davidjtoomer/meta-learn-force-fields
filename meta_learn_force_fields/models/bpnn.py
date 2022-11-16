import torch

from .featurizer import Featurizer, FeaturizerConfig


class BPNN(torch.nn.Module):
    def __init__(self, num_layers: int, feature_config: FeaturizerConfig):
        super().__init__()
        self.num_layers = num_layers
        self.feature_config = feature_config

        self.featurizer = Featurizer(self.feature_config)

    def forward(self, atomic_numbers: torch.Tensor, coordinates: torch.Tensor, parameters: dict) -> torch.Tensor:
        x = self.featurizer(atomic_numbers, coordinates)
        for i in range(self.num_layers):
            x = torch.nn.functional.linear(
                x, parameters[f'weight_{i}'], parameters[f'bias_{i}'])
            x = torch.nn.functional.silu(x)
        x = x.squeeze()
        return x.sum(dim=-1)

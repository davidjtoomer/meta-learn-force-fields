import torch

from .symmetry_functions import G1, G2, G3, G4, G5


class FeaturizerConfig:
    def __init__(
            self,
            cutoff_radius: float = 6.0,
            g1_params: dict = None,
            g2_params: dict = None,
            g3_params: dict = None,
            g4_params: dict = None,
            g5_params: dict = None):
        self.cutoff_radius = cutoff_radius
        self.g1_params = g1_params
        self.g2_params = g2_params
        self.g3_params = g3_params
        self.g4_params = g4_params
        self.g5_params = g5_params

    @property
    def num_features(self):
        num_features = 0
        for params in [
                self.g1_params,
                self.g2_params,
                self.g3_params,
                self.g4_params,
                self.g5_params]:
            if params is not None:
                num_features += len(params)
        return num_features


class Featurizer(torch.nn.Module):
    def __init__(self, config: FeaturizerConfig):
        super().__init__()
        self.config = config

        self.radial_symmetry_functions = []
        if self.config.g1_params is not None:
            for params in self.config.g1_params:
                self.radial_symmetry_functions.append(
                    G1(cutoff_radius=self.config.cutoff_radius, **params))
        if self.config.g2_params is not None:
            for params in self.config.g2_params:
                self.radial_symmetry_functions.append(
                    G2(cutoff_radius=self.config.cutoff_radius, **params))
        if self.config.g3_params is not None:
            for params in self.config.g3_params:
                self.radial_symmetry_functions.append(
                    G3(cutoff_radius=self.config.cutoff_radius, **params))

        self.angular_symmetry_functions = []
        if self.config.g4_params is not None:
            for params in self.config.g4_params:
                self.angular_symmetry_functions.append(
                    G4(cutoff_radius=self.config.cutoff_radius, **params))
        if self.config.g5_params is not None:
            for params in self.config.g5_params:
                self.angular_symmetry_functions.append(
                    G5(cutoff_radius=self.config.cutoff_radius, **params))

    def forward(self, atomic_numbers: torch.Tensor,
                coordinates: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(coordinates, coordinates)
        distance_vectors = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)

        features = []
        features.append(atomic_numbers)
        for function in self.radial_symmetry_functions:
            features.append(function(distances))
        for function in self.angular_symmetry_functions:
            features.append(function(distances, distance_vectors))
        return torch.cat(features, dim=-1)

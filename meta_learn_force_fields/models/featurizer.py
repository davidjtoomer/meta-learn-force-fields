import torch

from .symmetry_functions import G1, G2, G3, G4, G5


class FeaturizerConfig:
    def __init__(
            self,
            g1_param_ranges: dict = None,
            g2_param_ranges: dict = None,
            g3_param_ranges: dict = None,
            g4_param_ranges: dict = None,
            g5_param_ranges: dict = None):
        self.g1_param_ranges = g1_param_ranges
        self.g2_param_ranges = g2_param_ranges
        self.g3_param_ranges = g3_param_ranges
        self.g4_param_ranges = g4_param_ranges
        self.g5_param_ranges = g5_param_ranges

        self.set_params()

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

    def set_params(self):
        self.g1_params = []
        if self.g1_param_ranges is not None:
            for cutoff_radius in self.g1_param_ranges['cutoff_radius']:
                self.g1_params.append({
                    'cutoff_radius': cutoff_radius
                })

        self.g2_params = []
        if self.g2_param_ranges is not None:
            for cutoff_radius in self.g2_param_ranges['cutoff_radius']:
                for center_radius in self.g2_param_ranges['center_radius']:
                    for eta in self.g2_param_ranges['eta']:
                        self.g2_params.append({
                            'cutoff_radius': cutoff_radius,
                            'center_radius': center_radius,
                            'eta': eta
                        })

        self.g3_params = []
        if self.g3_param_ranges is not None:
            for cutoff_radius in self.g3_param_ranges['cutoff_radius']:
                for kappa in self.g3_param_ranges['kappa']:
                    self.g3_params.append({
                        'cutoff_radius': cutoff_radius,
                        'kappa': kappa
                    })

        self.g4_params = []
        if self.g4_param_ranges is not None:
            for cutoff_radius in self.g4_param_ranges['cutoff_radius']:
                for eta in self.g4_param_ranges['eta']:
                    for zeta in self.g4_param_ranges['zeta']:
                        for lambda_ in self.g4_param_ranges['lambda_']:
                            self.g4_params.append({
                                'cutoff_radius': cutoff_radius,
                                'eta': eta,
                                'zeta': zeta,
                                'lambda_': lambda_
                            })

        self.g5_params = []
        if self.g5_param_ranges is not None:
            for cutoff_radius in self.g4_param_ranges['cutoff_radius']:
                for eta in self.g4_param_ranges['eta']:
                    for zeta in self.g4_param_ranges['zeta']:
                        for lambda_ in self.g4_param_ranges['lambda_']:
                            self.g5_params.append({
                                'cutoff_radius': cutoff_radius,
                                'eta': eta,
                                'zeta': zeta,
                                'lambda_': lambda_
                            })


class Featurizer(torch.nn.Module):
    def __init__(self, config: FeaturizerConfig):
        super().__init__()
        self.config = config

        self.radial_symmetry_functions = []
        if self.config.g1_params is not None:
            for params in self.config.g1_params:
                self.radial_symmetry_functions.append(G1(**params))
        if self.config.g2_params is not None:
            for params in self.config.g2_params:
                self.radial_symmetry_functions.append(G2(**params))
        if self.config.g3_params is not None:
            for params in self.config.g3_params:
                self.radial_symmetry_functions.append(G3(**params))

        self.angular_symmetry_functions = []
        if self.config.g4_params is not None:
            for params in self.config.g4_params:
                self.angular_symmetry_functions.append(
                    G4(**params))
        if self.config.g5_params is not None:
            for params in self.config.g5_params:
                self.angular_symmetry_functions.append(
                    G5(**params))

    def forward(self, atomic_numbers: torch.Tensor,
                coordinates: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(coordinates, coordinates)
        distance_vectors = coordinates.unsqueeze(-2) - \
            coordinates.unsqueeze(-3)

        atomic_numbers = atomic_numbers.expand(coordinates.shape[0], -1)

        features = []
        features.append(atomic_numbers.unsqueeze(-1))
        for function in self.radial_symmetry_functions:
            features.append(function(distances).unsqueeze(-1))
        for function in self.angular_symmetry_functions:
            features.append(
                function(distance_vectors, distances).unsqueeze(-1))
        return torch.cat(features, dim=-1)

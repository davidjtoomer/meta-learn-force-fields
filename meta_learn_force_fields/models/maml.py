from typing import List

import torch

from .bpnn import BPNN
from .featurizer import FeaturizerConfig


class MAML:
    def __init__(
            self,
            mlp_layers: List[int],
            feature_config: FeaturizerConfig,
            num_inner_steps: int = 1,
            inner_lr: float = 0.1,
            learn_inner_lr: bool = False,
            outer_lr: float = 0.001):
        self.mlp_layers = mlp_layers
        self.feature_config = feature_config
        self.num_inner_steps = num_inner_steps
        self.inner_lr = inner_lr
        self.learn_inner_lr = learn_inner_lr
        self.outer_lr = outer_lr

        self.num_mlp_layers = len(self.mlp_layers) - 1

        self.model = BPNN(self.num_mlp_layers, self.feature_config)

        self.meta_parameters = {}
        for i in range(self.num_mlp_layers):
            self.meta_parameters[f'weight_{i}'] = torch.nn.xavier_uniform_(
                torch.empty(
                    self.mlp_layers[i], self.mlp_layers[i + 1], requires_grad=True)
            )
            self.meta_parameters[f'bias_{i}'] = torch.nn.xavier_uniform_(
                torch.empty(self.mlp_layers[i + 1], requires_grad=True)
            )

        self.inner_lrs = {
            key: torch.tensor(
                self.inner_lr,
                requires_grad=self.learn_inner_lr) for key in self.meta_parameters.keys()}

        self.loss_fn = torch.nn.L1Loss()

    def inner_loop(
            self,
            atomic_numbers: torch.Tensor,
            coordinates: torch.Tensor,
            energies: torch.Tensor,
            train: bool) -> torch.Tensor:
        losses = []
        parameters = {key: torch.clone(value)
                      for key, value in self.meta_parameters.items()}
        for _ in range(self.num_inner_steps):
            energy_pred = self.model(atomic_numbers, coordinates, parameters)
            loss = self.loss(energy_pred, energies)
            gradients = torch.autograd.grad(
                loss, parameters.values(), create_graph=train)
            for i, (key, value) in enumerate(parameters.items()):
                parameters[key] = value - self.inner_lrs[key] * gradients[i]
            losses.append(loss.item())

        return parameters, losses

    def outer_loop(self, task_batch: list, train: bool):
        outer_loss_batch = []
        support_losses_batch = []
        for task in task_batch:
            if not task:
                continue
            atomic_numbers, support_coordinates, support_energies, query_coordinates, query_energies = task

            parameters, inner_losses = self.inner_loop(
                atomic_numbers, support_coordinates, support_energies, train)
            support_losses_batch.append(inner_losses)

            outer_energy_pred = self.model(
                atomic_numbers, query_coordinates, parameters)
            outer_loss = self.loss_fn(outer_energy_pred, query_energies)
            outer_loss_batch.append(outer_loss)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        return outer_loss, support_losses_batch
from typing import List

import torch

from .bpnn import BPNN
from .featurizer import FeaturizerConfig


class MAML:
    def __init__(
            self,
            mlp_layers: List[int],
            atomic_numbers: List[int],
            feature_config: FeaturizerConfig,
            num_inner_steps: int = 1,
            inner_lr: float = 0.1,
            learn_inner_lr: bool = False,
            outer_lr: float = 0.001,
            loss_fn=torch.nn.L1Loss(),
            device=torch.device('cpu')):
        self.mlp_layers = mlp_layers
        self.atomic_numbers = atomic_numbers
        self.feature_config = feature_config
        self.num_inner_steps = num_inner_steps
        self.inner_lr = inner_lr
        self.learn_inner_lr = learn_inner_lr
        self.outer_lr = outer_lr
        self.loss_fn = loss_fn
        self.device = device

        self.num_mlp_layers = len(self.mlp_layers) - 1

        self.model = BPNN(self.num_mlp_layers,
                          self.atomic_numbers, self.feature_config)
        self.model.to(self.device)

        self.meta_parameters = {}
        for atomic_number in self.atomic_numbers:
            for i in range(self.num_mlp_layers):
                self.meta_parameters[f'atom_{atomic_number}_weight_{i}'] = torch.nn.init.xavier_normal_(
                    torch.empty(self.mlp_layers[i + 1], self.mlp_layers[i], requires_grad=True, device=self.device))
                self.meta_parameters[f'atom_{atomic_number}_bias_{i}'] = torch.nn.init.zeros_(
                    torch.empty(self.mlp_layers[i + 1], requires_grad=True, device=self.device))

        self.inner_lrs = {
            key: torch.tensor(
                self.inner_lr,
                requires_grad=self.learn_inner_lr,
                device=self.device) for key in self.meta_parameters.keys()}

    def inner_loop(
            self,
            atomic_numbers: torch.Tensor,
            coordinates: torch.Tensor,
            energies: torch.Tensor,
            train: bool):
        losses = []
        parameters = {key: torch.clone(value)
                      for key, value in self.meta_parameters.items()}
        for _ in range(self.num_inner_steps):
            energy_pred = self.model(atomic_numbers, coordinates, parameters)
            loss = self.loss_fn(energy_pred, energies)
            gradients = torch.autograd.grad(
                loss, parameters.values(), create_graph=train)
            for i, (key, value) in enumerate(parameters.items()):
                parameters[key] = value - self.inner_lrs[key] * gradients[i]
            losses.append(loss)

        energy_pred = self.model(atomic_numbers, coordinates, parameters)
        loss = self.loss_fn(energy_pred, energies)
        losses.append(loss)

        return parameters, torch.tensor(losses)

    def outer_loop(self, task_batch: list, train: bool = True):
        outer_loss_batch = []
        support_losses_batch = []
        for task in task_batch:
            if not task:
                continue
            atomic_numbers, support_coordinates, support_energies, query_coordinates, query_energies = task
            atomic_numbers = atomic_numbers.to(self.device)
            support_coordinates = support_coordinates.to(self.device)
            support_energies = support_energies.to(self.device)
            query_coordinates = query_coordinates.to(self.device)
            query_energies = query_energies.to(self.device)

            parameters, inner_losses = self.inner_loop(
                atomic_numbers, support_coordinates, support_energies, train)
            support_losses_batch.append(inner_losses)

            outer_energy_pred = self.model(
                atomic_numbers, query_coordinates, parameters)
            outer_loss = self.loss_fn(outer_energy_pred, query_energies)
            outer_loss_batch.append(outer_loss)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        support_losses = torch.mean(torch.stack(support_losses_batch), dim=0)
        return outer_loss, support_losses

    def state_dict(self):
        return {
            'meta_parameters': self.meta_parameters,
            'inner_lrs': self.inner_lrs,
        }

    def load_state_dict(self, state_dict):
        self.meta_parameters = state_dict['meta_parameters']
        self.inner_lrs = state_dict['inner_lrs']

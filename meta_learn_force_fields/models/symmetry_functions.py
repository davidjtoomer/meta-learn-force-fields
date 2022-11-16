import torch


class RadialFunction(torch.nn.Module):
    def __init__(self, cutoff_radius: float = 5.0):
        super().__init__()
        self.cutoff_radius = cutoff_radius

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        radials = torch.tanh(1 - (distances / self.cutoff_radius)) ** 3
        return torch.where(
            distances <= self.cutoff_radius,
            radials,
            torch.zeros_like(distances))


class AngularFunction(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distance_vectors: torch.Tensor,
                distances: torch.Tensor) -> torch.Tensor:
        dot_products = torch.einsum(
            '...ijd,...jkd->...ijk', distance_vectors, distance_vectors)
        magnitudes = torch.einsum(
            '...ij,...jk->...ijk', distances, distances)
        return dot_products / magnitudes


class G1(torch.nn.Module):
    def __init__(self, cutoff_radius: float = 5.0):
        super().__init__()
        self.cutoff_radius = cutoff_radius

        self.radial_function = RadialFunction(self.cutoff_radius)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        radials = self.radial_function(distances)
        return radials.sum(dim=-1)


class G2(torch.nn.Module):
    def __init__(self, cutoff_radius: float = 5.0, center_radius: float = 5.0, eta: float = 0.5):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.center_radius = center_radius
        self.eta = eta

        self.radial_function = RadialFunction(self.cutoff_radius)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        exponential_term = torch.exp(-self.eta *
                                     (distances - self.center_radius) ** 2)
        radial_term = self.radial_function(distances)
        return (exponential_term * radial_term).sum(dim=-1)


class G3(torch.nn.Module):
    def __init__(self, cutoff_radius: float = 5.0, kappa: float = 1.0):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.kappa = kappa

        self.radial_function = RadialFunction(self.cutoff_radius)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        cosine_term = torch.cos(self.kappa * distances)
        radial_term = self.radial_function(distances)
        return (cosine_term * radial_term).sum(dim=-1)


class G4(torch.nn.Module):
    def __init__(
            self,
            cutoff_radius: float = 5.0,
            eta: float = 0.5,
            zeta: float = 1.0,
            lambda_: float = 1.0):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.eta = eta
        self.zeta = zeta
        self.lambda_ = lambda_

        self.radial_function = RadialFunction(self.cutoff_radius)
        self.angular_function = AngularFunction()

    def forward(self, distance_vectors: torch.Tensor,
                distances: torch.Tensor) -> torch.Tensor:
        cosines = self.angular_function(distance_vectors, distances)
        cosine_term = (1 + self.lambda_ * cosines) ** self.zeta
        cosine_term = torch.nan_to_num(
            cosine_term, nan=0.0, posinf=0.0, neginf=0.0)

        exponent = torch.unsqueeze(distances, dim=-1) ** 2 + torch.unsqueeze(
            distances, dim=-2) ** 2 + torch.unsqueeze(distances, dim=-3) ** 2
        exponential_term = torch.exp(-self.eta * exponent)

        radials = self.radial_function(distances)
        radial_term = torch.unsqueeze(radials,
                                      dim=-1) * torch.unsqueeze(radials,
                                                                dim=-2) * torch.unsqueeze(radials,
                                                                                          dim=-3)

        coefficient = 2 ** (1 - self.zeta)

        return coefficient * (cosine_term * exponential_term * radial_term).sum(dim=(-1, -2))


class G5(torch.nn.Module):
    def __init__(
            self,
            cutoff_radius: float = 5.0,
            eta: float = 0.5,
            zeta: float = 1.0,
            lambda_: float = 1.0):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.eta = eta
        self.zeta = zeta
        self.lambda_ = lambda_

        self.radial_function = RadialFunction(self.cutoff_radius)
        self.angular_function = AngularFunction()

    def forward(self, distance_vectors: torch.Tensor,
                distances: torch.Tensor) -> torch.Tensor:
        cosines = self.angular_function(distance_vectors, distances)
        cosine_term = (1 + self.lambda_ * cosines) ** self.zeta
        cosine_term = torch.nan_to_num(
            cosine_term, nan=0.0, posinf=0.0, neginf=0.0)

        exponent = torch.unsqueeze(
            distances, dim=-1) ** 2 + torch.unsqueeze(distances, dim=-2) ** 2
        exponential_term = torch.exp(-self.eta * exponent)

        radials = self.radial_function(distances)
        radial_term = torch.unsqueeze(
            radials, dim=-1) * torch.unsqueeze(radials, dim=-2)

        coefficient = 2 ** (1 - self.zeta)

        return coefficient * (cosine_term * exponential_term * radial_term).sum(dim=(-1, -2))

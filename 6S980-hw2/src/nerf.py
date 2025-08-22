from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .field.field import Field
import torch

def f32(x):
    return torch.tensor(x, dtype=torch.float32)

class NeRF(nn.Module):
    cfg: DictConfig
    field: Field

    def __init__(self, cfg: DictConfig, field: Field) -> None:
        super().__init__()
        self.cfg = cfg
        self.field = field

    def forward(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
    ) -> Float[Tensor, "batch 3"]:
        """Render the rays using volumetric rendering. Use the following steps:

        1. Generate sample locations along the rays using self.generate_samples().
        2. Evaluate the neural field at the sample locations. The neural field's output
           has four channels: three for RGB color and one for volumetric density. Don't
           forget to map these channels to valid output ranges.
        3. Compute the alpha values for the evaluated volumetric densities using
           self.compute_alpha_values().
        4. Composite these alpha values together with the evaluated colors from.
        """
        (query_point, depth_point) = self.generate_samples(origins, directions, near, far, self.cfg["num_samples"])

        sigma_colorRGB = self.field(query_point.reshape(-1, 3))

        sigma = torch.nn.functional.softplus(sigma_colorRGB[:, 0][..., None]) 

        colorRGB = torch.sigmoid(sigma_colorRGB[:,1:])

        colorRGB = colorRGB.reshape(query_point.shape) 
        sigma = sigma.reshape(query_point.shape[0], query_point.shape[1])

        alpha = self.compute_alpha_values(sigma, depth_point)

        out = self.alpha_composite(alpha, colorRGB)

        return out

        #print("here")


    def generate_samples(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
        num_samples: int,
    ) -> tuple[
        Float[Tensor, "batch sample 3"],  # xyz sample locations
        Float[Tensor, "batch sample+1"],  # sample boundaries
    ]:
        """For each ray, equally divide the space between the specified near and far
        planes into num_samples segments. Return the segment boundaries (including the
        endpoints at the near and far planes). Also return sample locations, which fall
        at the midpoints of the segments.
        """
        
        depth_values = torch.linspace(near, far, num_samples,device=origins.device).expand(origins.shape[0], num_samples)
        query_points = origins[..., None, :] + directions[..., None, :] * depth_values[..., :, None]
        mid_point = (depth_values[:, -1] + depth_values[:, 0]) / 2.
        mid_point_expanded = torch.ones((origins.shape[0]),device="cuda:0",dtype=torch.float32)*mid_point
        
        return tuple([query_points, torch.cat([depth_values, mid_point_expanded[:,None]],dim=1)])

    def compute_alpha_values(
        self,
        sigma: Float[Tensor, "batch sample"],
        boundaries: Float[Tensor, "batch sample+1"],
    ) -> Float[Tensor, "batch sample"]:
        """Compute alpha values from volumetric densities (values of sigma) and segment
        boundaries.
        """

        return 1.0 - torch.exp(-sigma * boundaries[:,:-1])

    def alpha_composite(
        self,
        alphas: Float[Tensor, "batch sample"],
        colors: Float[Tensor, "batch sample 3"],
    ) -> Float[Tensor, "batch 3"]:
        """Alpha-composite the supplied alpha values and colors. You may assume that the
        background is black.
        """
        accumulated_transmittance = torch.cumprod(1- alphas + 1e-10, 1)

        weights_ = torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

        weights = weights_.unsqueeze(2)*alphas.unsqueeze(2)

        c = (weights*colors).sum(dim=1)

        weight_sum = weights.sum(-1).sum(-1) # Regularization for white background

        return c + 1 - weight_sum.unsqueeze(-1)


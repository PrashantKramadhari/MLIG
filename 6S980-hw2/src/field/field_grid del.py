from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field
import torch

class FieldGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a grid for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/grid.yaml):

        - side_length: the side length in each dimension

        Your architecture only needs to support 2D and 3D grids.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate in (2, 3)
        if d_coordinate == 2 :
            d = torch.linspace(0, 1,cfg["side_length"]-1, device="cuda:0")
            meshx, meshy = torch.meshgrid((d, d))
            self.grid = torch.stack((meshy, meshx), 2).permute(2,1,0)
            self.grid = self.grid.unsqueeze(0) # add batch dim
        else:
            d = torch.linspace(-1, 1, cfg.side_length, device="cuda:0")
            meshx, meshy, meshz = torch.meshgrid((d, d, d))
            self.grid = torch.stack((meshy, meshx, meshz), 3,).permute(3,2,1,0)
            self.grid = self.grid.unsqueeze(0) # add batch dim


    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.
        """

        return torch.nn.functional.grid_sample(self.grid, coordinates[None,None,:,:],align_corners = True).squeeze(0).squeeze(1).permute(1,0)

from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field
import torch
import torch.nn as nn

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
        self.coord = d_coordinate
        if self.coord  == 2:
            self.grid = nn.Parameter(torch.randn(1, d_out,cfg["side_length"]-1,cfg["side_length"]-1))
        elif self.coord  == 3: 
            self.grid = nn.Parameter(torch.randn(1, d_out,cfg["side_length"]-1,cfg["side_length"]-1,cfg["side_length"]-1))
   

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.
        """
        if self.coord  == 2:
            print(self.grid.shape)
            print(coordinates[None,None,:,:].shape)
            out= torch.nn.functional.grid_sample(self.grid, coordinates[None,None,:,:],align_corners = True).squeeze(0).squeeze(1).permute(1,0)
            print(out.shape)
            return out
        elif self.coord == 3:
            return torch.nn.functional.grid_sample(self.grid, coordinates[None,None,None,:,:],align_corners = True).squeeze(0).squeeze(1).permute(1,0)



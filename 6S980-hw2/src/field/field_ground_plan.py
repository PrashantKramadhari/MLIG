from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field
from .field_mlp import FieldMLP
from .field_grid import FieldGrid
from ..components.positional_encoding import PositionalEncoding
import torch
class FieldGroundPlan(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a neural ground plan. You should reuse the following components:

        - FieldGrid from  src/field/field_grid.py
        - FieldMLP from src/field/field_mlp.py
        - PositionalEncoding from src/components/positional_encoding.py

        Your ground plan only has to handle the 3D case.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate == 3
        self.pos_emd_en = cfg['positional_encoding_octaves'] 
        self.pos_enc = PositionalEncoding(cfg['positional_encoding_octaves'])
        self.grid = FieldGrid(cfg["grid"],d_coordinate-1, d_out)
        self.ground_net = FieldMLP(cfg["mlp"],d_out+self.pos_enc.d_out(1), d_out)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the ground plan at the specified coordinates. You should:

        - Sample the grid using the X and Y coordinates.
        - Positionally encode the Z coordinates.
        - Concatenate the grid's outputs with the corresponding encoded Z values, then
          feed the result through the MLP.
        """

        grid_x_y = self.grid(coordinates[:,:2])
        pos_end_z = self.pos_enc(coordinates[:,2].reshape(-1,1))
        x = torch.cat([grid_x_y, pos_end_z],dim=-1)
        out = self.ground_net(x)

        return out

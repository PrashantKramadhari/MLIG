from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field
from .field_mlp import FieldMLP
from .field_grid import FieldGrid


class FieldHybridGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a hybrid grid-mlp neural field. You should reuse FieldGrid from
        src/field/field_grid.py and FieldMLP from src/field/field_mlp.py in your
        implementation.

        Hint: Since you're reusing existing components, you only need to add one line
        each to __init__ and forward!
        """
        super().__init__(cfg, d_coordinate, d_out)
        self.grid = FieldGrid(cfg["grid"],d_coordinate, cfg["d_grid_feature"])
        self.hybrid_net = FieldMLP(cfg["mlp"],cfg["d_grid_feature"],d_out)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        
        out = self.grid(coordinates)
        return self.hybrid_net(out)

from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
from torch import nn

from .field import Field
from ..components.positional_encoding import PositionalEncoding

class FieldMLP(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up an MLP for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/mlp.yaml):

        - positional_encoding_octaves: The number of octaves in the positional encoding.
          If this parameter is None, do not positionally encode the input.
        - num_hidden_layers: The number of hidden linear layers.
        - d_hidden: The dimensionality of the hidden layers.

        Don't forget to add ReLU between your linear layers!
        """

        super().__init__(cfg, d_coordinate, d_out)

        self.mlp = []
        self.pos_emd_en = cfg['positional_encoding_octaves'] 
        if (self.pos_emd_en != None ):
            self.pos_enc = PositionalEncoding(cfg['positional_encoding_octaves'])
            self.mlp.append(nn.Linear(self.pos_enc.d_out(d_coordinate),cfg["d_hidden"]))
        else :
            self.mlp.append(nn.Linear(d_coordinate, cfg["d_hidden"]))


        def init_weights_normal(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for i in range(cfg["num_hidden_layers"]):
            self.mlp.append(nn.Linear(cfg["d_hidden"], cfg["d_hidden"]))
            self.mlp.append(nn.ReLU(inplace=True))

        self.mlp.append(nn.Linear(cfg["d_hidden"], d_out))

        self.mlp = nn.Sequential(*self.mlp)

        self.mlp.apply(init_weights_normal)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""
        if (self.pos_emd_en != None ):
            out = self.mlp(self.pos_enc(coordinates))
        else:
            out= self.mlp(coordinates)
        return out
    

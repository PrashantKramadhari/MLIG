from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .field import Field
from ..components.sine_layer import SineLayer
import numpy as np
import torch

class FieldSiren(Field):
    network: nn.Sequential

    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a SIREN network using the sine layers at src/components/sine_layer.py.
        Your network should consist of:

        - An input sine layer whose output dimensionality is 256
        - Two hidden sine layers with width 256
        - An output linear layer
        """
        super().__init__(cfg, d_coordinate, d_out)
        self.net = []
        self.hidden_layers = 2
        self.hidden_features = 256
        self.sine_layer_out = 256
        self.outermost_linear = True
        self.hidden_omega_0 = 30
        self.net.append(SineLayer(d_coordinate, self.hidden_features))

        for i in range(self.hidden_layers):
            self.net.append(SineLayer(self.hidden_features, self.hidden_features))

        if self.outermost_linear:
            final_linear = nn.Linear(self.hidden_features, d_out)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / self.hidden_features) / self.hidden_omega_0, 
                                              np.sqrt(6 / self.hidden_features) / self.hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(self.hidden_features,d_out))
        
        self.net = nn.Sequential(*self.net)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        coordinates = coordinates.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coordinates)
        return output        

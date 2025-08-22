import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import torch

def f32(x):
    return torch.tensor(x, dtype=torch.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        self.num_octaves = num_octaves

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """

        if len(samples.shape) ==1 : 
            bz=1
            dim_in=samples.shape[0]
        else :
            bz= samples.shape[0]
            dim_in=samples.shape[1]

        
        out = torch.empty(self.d_out(dim_in))

        #for bz_it in range(bz):
        x = torch.tensor([],device=samples.device)
        for i in range(self.num_octaves):
            sin = torch.sin((2 ** i) * 2*torch.pi * samples[:])
            cos = torch.cos((2 ** i) * 2*torch.pi * samples[:])
            x = torch.cat([x, sin, cos], dim=-1)
        
        return f32(x)
    
    def d_out(self, dimensionality: int):
        return self.num_octaves * dimensionality * 2

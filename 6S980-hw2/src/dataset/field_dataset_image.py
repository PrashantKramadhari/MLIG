from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
from .field_dataset import FieldDataset
import torch
from PIL import Image
import torchvision.transforms as transforms 
import numpy as np
import cv2
import matplotlib.pyplot as plt

class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""

        super().__init__(cfg)

        img =  cv2.cvtColor(cv2.imread(cfg.path),cv2.COLOR_BGR2RGB)

        self.img = torch.tensor(img , dtype=torch.float32,
                                device=torch.device('cuda:0')).permute(2,0,1).unsqueeze(0)
    
        
    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        
        """
        coordinates_renorm =  (2* (coordinates - 0))-1
        output = torch.nn.functional.grid_sample(self.img, coordinates_renorm[None,None,:,:],align_corners = True).squeeze(0).squeeze(1).permute(1,0)
        return output /255

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""

        return (128,) * self.d_coordinate

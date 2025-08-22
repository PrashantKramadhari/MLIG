from jaxtyping import Float
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import torch

def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """
    blank_image =  np.zeros((resolution[0], resolution[1]), np.uint8)
    blank_image.fill(0)
    #fig = plt.figure(figsize=(resolution[0], resolution[1]))
    #plt.scatter(blank_image[:,0], blank_image[0,:])




    pt_cam_cood          = extrinsics.inverse() @ torch.hstack((vertices, torch.ones((vertices.shape[0],1)))).T
    pt_img_coor          = (intrinsics @ pt_cam_cood[:,:3,:]).permute(0,2,1)
    pt_img_coor[:,:,:2] /= pt_img_coor[:,:, 2:]
    pt_pixel_coor        = pt_img_coor[:,:,:2]

    plt.xlim(resolution[0])
    plt.ylim(resolution[1])
    for i in range(pt_pixel_coor.shape[0]):
        plt.scatter(256*pt_pixel_coor[i,:,0], 256*pt_pixel_coor[i,:,1], c='black')
        plt.savefig('./output_images/'+str(i)+'.jpg')

    
    return pt_pixel_coor


    #raise NotImplementedError("This is your homework.")

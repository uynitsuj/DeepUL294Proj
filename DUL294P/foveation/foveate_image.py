import torch
# import cv2
import numpy as np
from matplotlib import pyplot as plt

class FoveateImage:
    def __init__(
        self,
        width: int,
        height: int,
        sigma: float = None,
        focus_cone = None,
        mode = 'tanh',
        pixel_ratio = 0.5,
        pff = 0.45 # Preserve Focus Cone Factor
    ):
        self.width = width
        self.height = height
        # assert width%2 == 0
        # assert height%2 == 0 # Assert even dimensions

        if sigma is None:
            self.sigma = (width) / 1.8
        else:
            assert type(sigma) is float
            self.sigma = sigma * width
        if focus_cone is None:
            self.focus_cone = width * 0.10
        else:
            assert type(focus_cone) is float
            self.focus_cone = width * focus_cone
        assert type(pixel_ratio) is float
        self.pixel_ratio = pixel_ratio

        torch.random.manual_seed(2)

        # Initialize and cache foveated masks
        if mode == 'tanh':
            func = self.tanh_2d(self.focus_cone)
            # plt.imshow(func)
            # plt.show()
        if mode == 'gaussian':
            func = self.gaussian_2d(self.focus_cone)
            # plt.imshow(func)
            # plt.show()
        
        rng = torch.rand((self.height,self.width))
        mask = torch.where(func > rng, func+rng*(1-pff), 0.0)
        self.sample_mask_idx = torch.topk((mask).reshape(-1), int(self.pixel_ratio * self.width * self.height))[1]

        sample_mask = torch.zeros_like(rng.reshape(-1))
        sample_mask[self.sample_mask_idx] = 1

        self.sample_mask = sample_mask.reshape((self.height, self.width))

        x = torch.linspace(0, self.width - 1, self.width, dtype=torch.int)
        y = torch.linspace(0, self.height - 1, self.height, dtype=torch.int)
        x, y = torch.meshgrid(x, y)
        

        xgrid_centered = (x - self.width/2 + 0.5).transpose(1,0).contiguous()
        ygrid_centered = (y - self.height/2 + 0.5).transpose(1,0).contiguous()
        
        centered_xs = xgrid_centered.view(-1)[self.sample_mask_idx]
        centered_ys = ygrid_centered.view(-1)[self.sample_mask_idx]

        self.coord_from_center = torch.stack((centered_xs, centered_ys)).T

        self.rs = torch.sqrt(centered_xs**2 + centered_ys**2)
        self.thetas = torch.atan2(centered_ys, centered_xs) * -180 / np.pi

    # @profile
    def foveate(self, image: np.array) -> np.array:
        """
        Processes captured image and stores results in class object.
        :param image: np.array object [W, H, C]
        :return: 
            Foveated image [# Pixels, C]
            Indices of foveated pixels from original image [# Pixels] 
                -> allows for foveated reconstruction of img via 
                recon = torch.zeros((h,w,3), dtype=foveatedimg.dtype)
                recon.view(-1, 3)[idxs] = foveatedimg[range(len(foveatedimg))]
            Radial distance of foveated pixels from center [# Pixels]
            Angle (deg.) of foveated pixels from center [# Pixels]
        """ 
        self.image = image
        w, h, c = image.shape
        self.width = w
        self.height = h

        # Process RGB
        if c == 3:
            mask = self.sample_mask
            mask = mask.unsqueeze(-1)
            self.result = torch.tensor(image).flatten(end_dim=-2)[self.sample_mask_idx,:]
        else:
            print("Image should contain 3 channels")
            raise AttributeError 
        
        return self.result, self.sample_mask_idx, self.rs, self.thetas
    
    def linear_dist_center(self, focus_cone = 0):
        x = torch.linspace(0, self.height - 1, self.height)
        y = torch.linspace(0, self.width - 1, self.width)
        x, y = torch.meshgrid(x, y)
        x_centered = x - self.height/2
        y_centered = y - self.width/2
        rad = torch.maximum(torch.sqrt(x_centered**2 + y_centered**2) - focus_cone, torch.zeros_like(x_centered))
        return rad
    
    def gaussian_2d(self, focus_cone = 0):
        x = torch.linspace(0, self.height - 1, self.height)
        y = torch.linspace(0, self.width - 1, self.width)
        x, y = torch.meshgrid(x, y)
        x_centered = x - self.height/2
        y_centered = y - self.width/2
        
        exponent = -((x_centered**2) / (2*self.sigma**2) + (y_centered**2) / (2*self.sigma**2))
        rad = torch.maximum((1-torch.exp(exponent)), torch.zeros_like(x_centered))

        return (1-rad)*(focus_cone/self.width+1)

    def tanh_2d(self, focus_cone):
        return 1-torch.tanh(self.linear_dist_center(focus_cone)/self.width)

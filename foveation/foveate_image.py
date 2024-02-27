import torch
# import cv2
import numpy as np
from matplotlib import pyplot as plt

class FoveateImage:
    def __init__(
        self,
        # device: torch.device,
        # width: int,
        # height: int,
        sigma: float = 1000.5,
    ):
        self.sigma = sigma

    # def greyscale(self):
        # plt.imshow(self.image[:,:,0])
        # plt.show()
        # greyscale = 0.22self.image[:,:,0]
    
    def foveate(self, image: np.array) -> np.array:
        """
        Processes captured image and stores results in class object.
        :param image: np.array object
        :return: Foveated image
        """
        self.image = image

        # self.greyscale()

        w, h, c = image.shape
        self.width = w
        self.height = h

        # Process RGB
        if c == 3:
            mask = self.sample_mask(mode='conic')
            mask = mask.unsqueeze(-1)
            channel_mask = torch.concat([mask, mask, mask], dim = -1).int()
            self.result = torch.tensor(image) * channel_mask
        else:
            print("Image should contain 3 channels")
            raise AttributeError 
        
        return self.result
    
    def linear_dist_center(self, focus_cone = 0):
        x = torch.linspace(0, self.width - 1, self.width)
        y = torch.linspace(0, self.height - 1, self.height)
        x, y = torch.meshgrid(x, y)
        x_centered = x - self.width/2
        y_centered = y - self.height/2
        # import pdb; pdb.set_trace()
        rad = torch.maximum(torch.sqrt(x_centered**2 + y_centered**2) - focus_cone, torch.zeros_like(x_centered))
        return rad
    
    def gaussian_2d(self):
        x = torch.linspace(0, self.width - 1, self.width)
        y = torch.linspace(0, self.height - 1, self.height)
        x, y = torch.meshgrid(x, y)
        x_centered = x - self.width/2
        y_centered = y - self.height/2
        
        exponent = -((x_centered**2) / (2*self.sigma**2) + (y_centered**2) / (2*self.sigma**2))
        return (torch.exp(exponent))

    def conic_2d(self, focus_cone):

        return 1-torch.tanh(self.linear_dist_center(focus_cone)/self.width)

    def sample_mask(self, mode = 'conic', pixel_ratio = 0.2, focus_cone = None):
        print("Original Size: " + str(self.width*self.height * 3 * 10e-6) + " MB")

        print("New Size: " + str(pixel_ratio * self.width * self.height * 3 * 10e-6) + " MB")

        print("Reduction: " + str(pixel_ratio*100) + "%")

        if focus_cone is None: 
            focus_cone = self.width * 0.03

        if mode == 'gaussian':
            gaussian = self.gaussian_2d()
            torch.random.manual_seed(1)
            rng = torch.rand((self.width,self.height))

            sample_mask_idx = torch.topk((rng * gaussian).reshape(-1), int(pixel_ratio * self.width * self.height))
            sample_mask = torch.zeros_like(rng.reshape(-1))

            sample_mask[sample_mask_idx[1]] = 1

            sample_mask = sample_mask.reshape((self.width, self.height))


            plt.imshow(sample_mask)
            plt.show()
            
        if mode == 'conic':
            conic = self.conic_2d(focus_cone)

            plt.imshow(conic)
            plt.show()
            torch.random.manual_seed(1)
            rng = torch.rand((self.width,self.height))

            # Full-res for center of visual field
            rad = self.linear_dist_center()
            rad[rad <= focus_cone] = 1
            rad[rad > focus_cone] = 0
            
            rng = torch.maximum(rng+0.3, rad)
            
            sample_mask_idx = torch.topk((rng * conic).reshape(-1), int(pixel_ratio * self.width * self.height))
            sample_mask = torch.zeros_like(rng.reshape(-1))

            sample_mask[sample_mask_idx[1]] = 1

            sample_mask = sample_mask.reshape((self.width, self.height))

        return sample_mask
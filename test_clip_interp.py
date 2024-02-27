from data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from encoders.image_encoder import BaseImageEncoderConfig
from encoders.openclip_encoder import OpenCLIPNetworkConfig
from datasets import load_dataset
from torch.utils import data
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin
from torchvision import transforms

import torch

def main():
    device = 'cuda:6'
    """The device to run on"""
    patch_tile_size_range: Tuple[int, int] = (0.05, 0.5)
    """The range of tile sizes to sample from for patch-based training"""
    patch_tile_size_res: int = 7
    """The number of tile sizes to sample from for patch-based training"""
    patch_stride_scaler: float = 0.5
    """The stride scaler for patch-based training"""
    network: BaseImageEncoderConfig = OpenCLIPNetworkConfig(device=device)
    """specifies the vision-language network config"""
    clip_downscale_factor: int = 1
    """The downscale factor for the clip pyramid"""

    clip_interpolator = PyramidEmbeddingDataloader(
        device='cuda:6',
        cfg={
            "tile_size_range": list(patch_tile_size_range),
            "tile_size_res": patch_tile_size_res,
            "stride_scaler": patch_stride_scaler,
            # "image_shape": [h,w],
        },
        model=network.setup()
    )

    dataset = load_dataset("imagenet-1k")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
            
    for i, batch in enumerate(dataset['train']):
        # print(i)
        # print(batch)
        # import pdb; pdb.set_trace()
        clip_out = clip_interpolator.generate_clip_interp(transform(batch['image']))
        H, W = data["image"].shape[:2]
        # scale = torch.rand(1).to(device)*(patch_tile_size_range[1]-patch_tile_size_range[0])+patch_tile_size_range[0]
        scale = torch.tensor(0.1).to(device)
        # curr_scale = scale
        scaled_height = H//clip_downscale_factor
        scaled_width = W//clip_downscale_factor
        random_pixels = torch.randperm(scaled_height*scaled_width)[:int((scaled_height*scaled_height)*0.5)]

        x = torch.arange(0, scaled_width*clip_downscale_factor, clip_downscale_factor).view(1, scaled_width, 1).expand(scaled_height, scaled_width, 1)
        y = torch.arange(0, scaled_height*clip_downscale_factor, clip_downscale_factor).view(scaled_height, 1, 1).expand(scaled_height, scaled_width, 1)
        image_idx_tensor = torch.ones(scaled_height, scaled_width, 1)
        positions = torch.cat((image_idx_tensor, y, x), dim=-1).view(-1, 3).to(int)
        positions = positions[random_pixels]
        with torch.no_grad():
            data["clip"], data["clip_scale"] = clip_interpolator(positions, scale)[0], clip_interpolator(positions, scale)[1]
        import pdb; pdb.set_trace()
        if i == 10:
            break
    

if __name__ == '__main__':
    main()
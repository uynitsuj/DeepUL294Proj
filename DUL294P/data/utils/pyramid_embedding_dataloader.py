import json
import os
from pathlib import Path

import numpy as np
import torch
from data.utils.feature_dataloader import FeatureDataloader
from data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader
from encoders.image_encoder import BaseImageEncoder
from encoders.openclip_encoder import OpenCLIPNetworkConfig
from tqdm import tqdm
import time

class PyramidEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        assert "tile_size_range" in cfg
        assert "tile_size_res" in cfg
        assert "stride_scaler" in cfg
        # assert "image_shape" in cfg
        # assert "model_name" in cfg

        self.tile_sizes = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
        self.strider_scaler_list = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in self.tile_sizes]

        self.model = model
        self.embed_size = self.model.embedding_dim
        self.data_dict = {}
        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self, img_points, scale=None):
        if scale is None:
            return self._random_scales(img_points)
        else:
            return self._uniform_scales(img_points, scale)

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])


    def _random_scales(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        # return: (B, 512), some random scale (between 0, 1)
        img_points = img_points.to(self.device)
        random_scale_bin = torch.randint(self.tile_sizes.shape[0] - 1, size=(img_points.shape[0],), device=self.device)
        random_scale_weight = torch.rand(img_points.shape[0], dtype=torch.float16, device=self.device)

        stepsize = (self.tile_sizes[1] - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0])

        bottom_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)
        top_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)

        for i in range(len(self.tile_sizes) - 1):
            ids = img_points[random_scale_bin == i]
            bottom_interp[random_scale_bin == i] = self.data_dict[i](ids)
            top_interp[random_scale_bin == i] = self.data_dict[i + 1](ids)

        return (
            torch.lerp(bottom_interp, top_interp, random_scale_weight[..., None]),
            (random_scale_bin * stepsize + random_scale_weight * stepsize)[..., None],
        )

    def _uniform_scales(self, img_points, scale):
        # img_points: (B, 3) # (img_ind, x, y)
        scale_bin = torch.floor(
            (scale - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0]) * (self.tile_sizes.shape[0] - 1)
        ).to(torch.int64)
        scale_weight = (scale - self.tile_sizes[scale_bin]) / (
            self.tile_sizes[scale_bin + 1] - self.tile_sizes[scale_bin]
        )
        interp_lst = torch.stack([interp(img_points) for interp in self.data_dict.values()])
        point_inds = torch.arange(img_points.shape[0])
        interp = torch.lerp(
            interp_lst[scale_bin, point_inds],
            interp_lst[scale_bin + 1, point_inds],
            torch.Tensor([scale_weight]).half().to(self.device)[..., None],
        )
        return interp / interp.norm(dim=-1, keepdim=True), scale
    
    def generate_clip_interp(self, image):
        # import pdb; pdb.set_trace()
        C, H, W = image.shape
        for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
            stride_scaler = self.strider_scaler_list[i]
            self.data_dict[i] = PatchEmbeddingDataloader(
                cfg={
                    "tile_ratio": tr.item(),
                    "stride_ratio": stride_scaler,
                    "image_shape": [H,W],
                    # "model_name": self.cfg["model_name"],
                },
                device=self.device,
                model=self.model,
                # image_list=image_list,
                # cache_path=Path(f"{self.cache_path}/level_{i}.npy"),
            )
            self.data_dict[i].create(None)
        img_batch = image.unsqueeze(0)
        start = time.time()

        clip_interp = []
        for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
            clip_interpolations = self.data_dict[i].add_images(img_batch)
            # self.data_dict[i].data[0,...] = clip_interpolations
            clip_interp.append(clip_interpolations)
        
        assert len(self.data_dict) != 0

        
        # for _ in img_batch:
            
        # for i, tr in enumerate(self.tile_sizes):
        #     clip_interp.append(self.data_dict[i].data[0,...])

            # self.out_queue.put(updates)
        #     j+=1
        
        print(f"PyramidEmbeddingProcess took {time.time()-start} seconds")
        return clip_interp
from typing import Callable, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor
from torch_cluster import knn
from torch_scatter import scatter_softmax
from torch_geometric.utils import coalesce
# from torch_geometric.nn import DynamicEdgeConv
from timm.models.layers import DropPath, trunc_normal_
import torch.nn as nn
import math
# from kpconv.kernels import KPConvLayer
# from kpconv.base_modules import FastBatchNorm1d
from lib.pointops2.functions import pointops
import time

class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            # nn.Conv2d(
            #     embed_dim // 2,
            #     embed_dim,
            #     kernel_size=(3, 3),
            #     stride=(2, 2),
            #     padding=(1, 1),
            # ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """ Multi-head self attention (MSA) module

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats, index_0, index_1, index_0_offsets, n_max):
        """ Forward function.

        Args:
            feats: N, C
            xyz: N, 3
            index_0: M,
            index_1: M,
        """
        # print(self.dim)

        N, C = feats.shape
        # import pdb; pdb.set_trace()
        M = index_0.shape[0]
 
        assert index_0.shape[0] == index_1.shape[0]
        
        # Query, Key, Value
        qkv = self.qkv(feats).reshape(N, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2] #[N, num_heads, C//num_heads]
        query = query * self.scale
        # import pdb; pdb.set_trace()
        attn_flat = pointops.attention_step1_v2(query.float(), key.float(), index_1.int(), index_0_offsets.int(), n_max).to(feats.device) #[M, num_heads]

        softmax_attn_flat = scatter_softmax(src=attn_flat, index=index_0, dim=0) #[M, num_heads]

        x = pointops.attention_step2(softmax_attn_flat.float(), value.float(), index_0.int(), index_1.int()).to(feats.device)
        
        x = x.view(N, C)

        x = self.proj(x)
        x = self.proj_drop(x) #[N, C]

        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, k, out_channels = None, dropout=0.0,
            ratio=4.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.k = k
        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)

        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        self.linear_o = nn.Linear(dim, out_channels) if out_channels else None

    def forward(self, feats, polar_pos, edge_index = None):
        skip = feats

        feats = self.norm1(feats)

        # if edge_index is None:
        #     start = time.time()
        #     edge_index = knn(feats, feats, self.k) #.flip([0])
        #     # edge_index = coalesce(torch.cat([edge_index, edge_index.flip([0])]))
        #     elapsed = (time.time() - start)
        #     print(f"Feat KNN Elapsed time: {elapsed}(s)")
        
        index_0, index_1 = edge_index.chunk(2, dim=0)
        index_0 = index_0.view(-1) #[M,]
        index_1 = index_1.view(-1) #[M,]
        index_0, indices = torch.sort(index_0) #[M,]
        index_1 = index_1[indices] #[M,]
        index_0_counts = index_0.bincount()
        n_max = index_0_counts.max()
        index_0_offsets = index_0_counts.cumsum(dim=-1) #[N]
        index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=feats.device), index_0_offsets], 0)

        feats = self.attn(feats, index_0, index_1, index_0_offsets, n_max)
        feats = skip + self.drop_path(feats)
        feats = feats + self.drop_path(self.mlp(self.norm2(feats)))
        if self.linear_o is not None:
            feats = self.linear_o(feats)
        return feats

class MLPPointEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)
    
class FoveatedTransformer(torch.nn.Module):
    def __init__(self, k, out_channels, dropout=0.1, num_layers = 2, channels=[256, 512], num_heads=[2, 3], ratio=4.0):
        super().__init__()
        assert len(channels) == len(num_heads) == num_layers
        self.channels = channels
        self.k = k
        self.patch_embed = ConvTokenizer(
            in_chans=3, embed_dim=channels[0], norm_layer=nn.LayerNorm
        )

        self.point_embedding = MLPPointEmbedding(in_dim=3, out_dim = channels[0])
        self.positional_encoding = MLPPointEmbedding(in_dim=2, out_dim = channels[0])

        self.layers = nn.ModuleList(
            [TransformerBlock(channels[i], num_heads[i], k, out_channels=channels[i+1] if i < num_layers-1 else channels[-1], dropout=dropout,
            ratio=ratio) for i in range(num_layers)
            ]
        )

        self.out_dim = sum(channels[1:]) + channels[-1]
        
        self.mlp = MLP(
            [channels[-1], 256, 512],
            dropout=dropout, norm=None
        )

    def forward(self, img, pix, idxs, rs, thetas, edge_index):
        # pix [N, 3]
        # rs [N]
        # thetas [N]
        # import pdb; pdb.set_trace()

        # start = time.time()
        patch_embed = self.patch_embed(img.permute(0, 3, 1, 2)) # patch_embed: (N, c0)
        # elapsed = (time.time() - start)
        # print(f"Patch Embed Elapsed time: {elapsed}(s)")
        # import pdb; pdb.set_trace()


        polar_pos = torch.stack([rs,thetas]).T
        polar_pos_enc = self.positional_encoding(polar_pos)

        # feats = self.point_embedding(pix) + polar_pos_enc # feats: (N, c0)
        feats = patch_embed.squeeze(0).view(-1, self.channels[0])[idxs] + polar_pos_enc
        
        # start = time.time()
        # edge_index = knn(polar_pos, polar_pos, 9) #.flip([0])
        # # edge_index = coalesce(torch.cat([edge_index, edge_index.flip([0])]))
        # elapsed = (time.time() - start)
        # print(f"KNN Elapsed time: {elapsed}(s)")

        for i, layer in enumerate(self.layers):
            feats = layer(feats, polar_pos, edge_index)

        pooled = feats.mean(dim = 0)

        return self.mlp(pooled)
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

    def forward(self, feats, xyz, index_0, index_1, index_0_offsets, n_max):
        """ Forward function.

        Args:
            feats: N, C
            xyz: N, 3
            index_0: M,
            index_1: M,
        """

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
        # import pdb; pdb.set_trace()
        # # Position embedding
        # relative_position = xyz[index_0] - xyz[index_1]
        # relative_position = torch.round(relative_position * 100000) / 100000
        # relative_position_index = (relative_position + 2 * self.window_size - 0.0001) // self.quant_size
        # assert (relative_position_index >= 0).all()
        # assert (relative_position_index <= 2*self.quant_grid_length - 1).all()

        # assert self.rel_query and self.rel_key
        # if self.rel_query and self.rel_key:
        #     relative_position_bias = pointops.dot_prod_with_idx_v3(query.float(), index_0_offsets.int(), n_max, key.float(), index_1.int(), self.relative_pos_query_table.float(), self.relative_pos_key_table.float(), relative_position_index.int())
        # elif self.rel_query:
        #     relative_position_bias = pointops.dot_prod_with_idx(query.float(), index_0.int(), self.relative_pos_query_table.float(), relative_position_index.int()) #[M, num_heads]
        # elif self.rel_key:
        #     relative_position_bias = pointops.dot_prod_with_idx(key.float(), index_1.int(), self.relative_pos_key_table.float(), relative_position_index.int()) #[M, num_heads]
        # else:
        #     relative_position_bias = 0.0


        # attn_flat = attn_flat #[M, num_heads]
        # import pdb; pdb.set_trace()
        
        softmax_attn_flat = scatter_softmax(src=attn_flat, index=index_0, dim=0) #[M, num_heads]

        # if self.rel_value:
        #     x = pointops.attention_step2_with_rel_pos_value_v2(softmax_attn_flat.float(), value.float(), index_0_offsets.int(), n_max, index_1.int(), self.relative_pos_value_table.float(), relative_position_index.int())
        # else:
        x = pointops.attention_step2(softmax_attn_flat.float(), value.float(), index_0.int(), index_1.int()).to(feats.device)
        
        # import pdb; pdb.set_trace()
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

    def forward(self, feats, xyz, temporal_edge_index, spatial_edge_index, batch):
        skip = feats

        feats = self.norm1(feats)

        if spatial_edge_index is None:
            spatial_edge_index = knn(xyz, xyz, self.k, batch, batch).flip([0])

        edge_index = coalesce(torch.cat([spatial_edge_index, spatial_edge_index.flip([0]), temporal_edge_index], dim=-1)) #[2, M]
        # edge_index = coalesce(torch.cat([spatial_edge_index, spatial_edge_index.flip([0])]))
        
        index_0, index_1 = edge_index.chunk(2, dim=0)
        index_0 = index_0.view(-1) #[M,]
        index_1 = index_1.view(-1) #[M,]
        index_0, indices = torch.sort(index_0) #[M,]
        index_1 = index_1[indices] #[M,]
        index_0_counts = index_0.bincount()
        n_max = index_0_counts.max()
        index_0_offsets = index_0_counts.cumsum(dim=-1) #[N]
        index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=feats.device), index_0_offsets], 0)

        feats = self.attn(feats, xyz, index_0, index_1, index_0_offsets, n_max)
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=False)  # Make sure gradients are not computed
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return self.encoding[:len(x), :]
    
class FoveatedTransformer(torch.nn.Module):
    def __init__(self, k, out_channels, dropout=0.5, num_layers = 2, channels=[48, 64], num_heads=[3, 4], ratio=4.0):
        super().__init__()
        assert len(channels) == len(num_heads) == num_layers

        self.point_embedding = MLPPointEmbedding(in_dim=3, out_dim = channels[0])
        self.positional_encoding = PositionalEncoding(channels[0])

        self.layers = nn.ModuleList(
            [TransformerBlock(channels[i], num_heads[i], k, out_channels=channels[i+1] if i < num_layers-1 else channels[-1], dropout=dropout,
            ratio=ratio) for i in range(num_layers)
            ]
        )

        self.out_dim = sum(channels[1:]) + channels[-1]
        
        self.mlp = MLP(
            [self.out_dim, 512, 256, out_channels],
            dropout=dropout, norm=None
        )

    def forward(self, clr, pos):
        # pos, batch = data.pos, data.batch # pos: (N, 3), batch: (N), i.e. [ 0,  0,  0,  ..., 49, 49, 49]
        # import pdb; pdb.set_trace()
        pe_batch = self.positional_encoding(torch.unique(batch))
        p_enc = torch.zeros((len(batch), pe_batch.shape[1]), device=pe_batch.device)
        # import pdb; pdb.set_trace()
        p_enc[torch.arange(0, len(batch)), :] = pe_batch[batch, :]

        feats = self.point_embedding(pos) + p_enc # feats: (N, 48)

        temporal_edge_index, spatial_edge_index = data.temporal_edge_index, data.edge_index # temporal_edge_index: (2, TE), spatial_edge_index: (2, SE)
        # spatial_edge_index = data.edge_index
        # temporal_edge_index = None
        xyz = pos
        # feats = xyz
        # import pdb; pdb.set_trace()
        
        feats_stack = []
        xyz_stack = []

        for i, layer in enumerate(self.layers):
            feats = layer(feats, xyz, temporal_edge_index, spatial_edge_index if i == 0 else None, batch)

            feats_stack.append(feats)
            xyz_stack.append(xyz)
        
        out = self.mlp(torch.cat(feats_stack, dim=1))

        return F.log_softmax(out, dim=1)
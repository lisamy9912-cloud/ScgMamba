## Our model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt
import os
import sys

current_directory = os.path.dirname(__file__) + '/../' + '../'
sys.path.append(current_directory)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

from models.model.mambablocks import BiSTSSMBlock
from models.model.gcn import ResGCNBlock, ModulatedGCNBlock

class BiDirectionalGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.structure_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.motion_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.out_project = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )
        nn.init.xavier_uniform_(self.structure_gate[1].weight, gain=0.1)
        nn.init.constant_(self.structure_gate[1].bias, 0)
        nn.init.xavier_uniform_(self.motion_gate[1].weight, gain=0.1)
        nn.init.constant_(self.motion_gate[1].bias, 0)
        nn.init.constant_(self.out_project[0].weight, 0)
        nn.init.constant_(self.out_project[0].bias, 0)

    def forward(self, x_spatial, x_temporal):
        gate_s2t = self.structure_gate(x_spatial)
        feat_temporal_refined = x_temporal * gate_s2t
        gate_t2s = self.motion_gate(x_temporal)
        feat_spatial_refined = x_spatial * gate_t2s
        x_fused = feat_spatial_refined + feat_temporal_refined
        return self.out_project(x_fused)


class ScgMamba(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=6, mlp_ratio=2., drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3

        neighbor_link = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
                         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
        self.adj = torch.zeros((num_joints, num_joints))
        for i, j in neighbor_link:
            self.adj[i, j] = 1
            self.adj[j, i] = 1
        self.adj = self.adj.cuda()

        # --- 2. Embeddings ---
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.Spatial_GCNs = nn.ModuleList([
            ModulatedGCNBlock(
                hidden_dim=embed_dim_ratio,
                adj=self.adj,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.TTEblocks = nn.ModuleList([
            BiSTSSMBlock(
                hidden_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                forward_type='v2'
            )
            for i in range(depth)
        ])

        self.FusionLayers = nn.ModuleList([
            BiDirectionalGatedFusion(embed_dim)
            for i in range(depth)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x):
        # x: (B, F, N, C)
        b, f, n, c = x.shape

        # 1. Embedding
        x = rearrange(x, 'b f n c -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, '(b f) n c -> b f n c', f=f)

        if self.adj.device != x.device:
            self.adj = self.adj.to(x.device)
            for blk in self.Spatial_GCNs:
                blk.gcn1.adj = self.adj
                blk.gcn2.adj = self.adj

        x = rearrange(x, 'b f n c -> (b n) f c')
        x += self.Temporal_pos_embed[:, :f, :]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f c -> b f n c', n=n)

        for i in range(self.block_depth):
            x_s_in = rearrange(x, 'b f n c -> (b f) n c')
            x_s = self.Spatial_GCNs[i](x_s_in)
            x_s = rearrange(x_s, '(b f) n c -> b f n c', f=f)
            x_t_in = rearrange(x_s, 'b f n c -> (b n) f 1 c')
            x_t = self.TTEblocks[i](x_t_in)
            x_t = rearrange(x_t, '(b n) f 1 c -> b f n c', n=n)
            delta = self.FusionLayers[i](x_spatial=x_s, x_temporal=x_t)
            x = x_t + delta
        # Head
        x = self.head(x)
        x = x.view(b, f, n, -1)
        return x
        
if __name__ == "__main__":
    torch.cuda.set_device(0)
    model = ScgMamba(num_frame=243, embed_dim_ratio=128,mlp_ratio = 2, depth = 10).cuda()
    from thop import profile, clever_format
    input_shape = (1, 243, 17, 2)
    x = torch.randn(input_shape).cuda()
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))
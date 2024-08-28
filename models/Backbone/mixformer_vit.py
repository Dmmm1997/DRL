from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from timm.models.layers import DropPath, Mlp
import numpy as np

from models.pos_utils import get_2d_sincos_pos_embed
from .utils import to_2tuple




class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w, s_h * s_w], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path1(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x



class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(
            self, img_size_s=256, img_size_t=128, patch_size=16, in_chans=3, num_classes=1000,
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
            attn_drop_rate=0., drop_path_rate=0., weight_init='', embed_layer=PatchEmbed,
            norm_layer=None, act_layer=None):
        super(
            VisionTransformer, self).__init__(
            img_size=224, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, weight_init=weight_init, norm_layer=norm_layer,
            act_layer=act_layer)

        self.patch_embed = embed_layer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)])

        self.grid_size_s = img_size_s // patch_size
        self.grid_size_t = img_size_t // patch_size
        self.num_patches_s = self.grid_size_s ** 2
        self.num_patches_t = self.grid_size_t ** 2
        # self.pos_embed_s = nn.Parameter(
        #     torch.zeros(1, self.num_patches_s, embed_dim),
        #     requires_grad=False)
        # self.pos_embed_t = nn.Parameter(
        #     torch.zeros(1, self.num_patches_t, embed_dim),
        #     requires_grad=False)

        # self.init_pos_embed()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(
            self.pos_embed_t.shape[-1],
            int(self.num_patches_t ** .5),
            cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(
            self.pos_embed_s.shape[-1],
            int(self.num_patches_s ** .5),
            cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def _get_pos_embed(self, pos_embed, H, W):
        ori_pos_H = ori_pos_W = int(pos_embed.shape[1]**0.5)
        return F.interpolate(
            pos_embed[:,1:].reshape(1, ori_pos_H, ori_pos_W, -1).permute(0, 3, 1, 2),
            size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)
        

    def forward(self, x_t, x_s):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_s = self.patch_embed(x_s)
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.grid_size_s
        H_t = W_t = self.grid_size_t

        # pos_embed_s = self._get_pos_embed(self.pos_embed, H_s, W_s)
        # pos_embed_t = self._get_pos_embed(self.pos_embed, H_t, W_t)

        # x_s = x_s + pos_embed_s
        # x_t = x_t + pos_embed_t
        x = torch.cat([x_t, x_s], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, H_t, W_t, H_s, W_s)

        x_t, x_s = torch.split(x, [H_t*W_t, H_s*W_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return [x_t_2d], [x_s_2d]


def get_mixformer_vit(
        vit_type, satellite_size=384, uav_size=128, **kwargs):
    img_size_s = satellite_size
    img_size_t = uav_size
    embed_dim = 768
    if vit_type == 'large_patch16':
        embed_dim = 1024
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=embed_dim, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1)
    elif vit_type == 'base_patch16':
        embed_dim = 768
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1)
    elif vit_type == 'small_patch16':
        embed_dim = 384
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=embed_dim, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1)
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'small_patch16' or 'large_patch16' or 'base_patch16'")

    # if pretrain:
    #     ckpt_path = pretrain_path
    #     if vit_type == 'small_patch16':
    #         ckpt = torch.load(ckpt_path, map_location='cpu')
    #     else:
    #         ckpt = torch.load(ckpt_path, map_location='cpu')['model']
    #     new_dict = {}
    #     for k, v in ckpt.items():
    #         if 'pos_embed' not in k and 'mask_token' not in k:    # use fixed pos embed
    #             new_dict[k] = v
    #     missing_keys, unexpected_keys = vit.load_state_dict(new_dict, strict=False)
    #     print("Load pretrained backbone checkpoint from:", ckpt_path)
    #     print("missing keys:", missing_keys)
    #     print("unexpected keys:", unexpected_keys)
    #     print("Loading pretrained ViT done.")

    return vit, [embed_dim]
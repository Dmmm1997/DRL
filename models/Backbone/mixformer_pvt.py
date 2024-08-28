import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
            sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, t_h, t_w, s_h, s_w):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # split template and search
            template, search = torch.split(x, [t_h * t_w, s_h * s_w], dim=1)

            x_t = template.permute(0, 2, 1).reshape(B, C, t_h, t_w)
            # Compress the scale to reduce the amount of calculations
            x_t = self.sr(x_t)
            # update the new h and w
            new_t_h = int(x_t.shape[2])
            new_t_w = int(x_t.shape[3])
            x_t = x_t.reshape(B, C, -1).permute(0, 2, 1)
            x_t = self.norm(x_t)

            x_s = search.permute(0, 2, 1).reshape(B, C, s_h, s_w)
            # Compress the scale to reduce the amount of calculations
            x_s = self.sr(x_s)
            # update the new h and w
            new_s_h = int(x_s.shape[2])
            new_s_w = int(x_s.shape[3])
            x_s = x_s.reshape(B, C, -1).permute(0, 2, 1)
            x_s = self.norm(x_s)

            x = torch.concat((x_t, x_s), dim=1)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C //
                                    self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C //
                                    self.num_heads).permute(2, 0, 3, 1, 4)
            new_t_h = t_h
            new_t_w = t_w
            new_s_h = s_h
            new_s_w = s_w

        k, v = kv[0], kv[1]
        q_mt, q_s = torch.split(q, [t_h * t_w, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [new_t_h * new_t_w, new_s_h * new_s_w], dim=2)
        v_mt, v_s = torch.split(v, [new_t_h * new_t_w, new_s_h * new_s_w], dim=2)

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
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
            attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
            embed_dims=[64, 128, 256, 512],
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            F4=False, num_stages=4):
        super().__init__()
        self.depths = depths
        self.F4 = F4
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size[i],
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                    norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std=.02)

        # init weights
        self.apply(self._init_weights)

    def load_param(self, checkpoint):
        pretran_model = torch.load(checkpoint)
        model2_dict = self.state_dict()
        state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys()}
        model2_dict.update(state_dict)
        self.load_state_dict(model2_dict)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, template, search):
        out_template = []
        out_search = []

        B = search.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            search, (s_H, s_W) = patch_embed(search)
            template, (t_H, t_W) = patch_embed(template)
            if i == self.num_stages - 1:
                s_pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, s_H, s_W)
                t_pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, t_H, t_W)
            else:
                s_pos_embed = self._get_pos_embed(pos_embed, patch_embed, s_H, s_W)
                t_pos_embed = self._get_pos_embed(pos_embed, patch_embed, t_H, t_W)
            x = torch.cat([template+t_pos_embed, search+s_pos_embed], dim=1)
            x = pos_drop(x)
            C = x.shape[2]
            for blk in block:
                x = blk(x, t_H, t_W, s_H, s_W)

            template, search = torch.split(x, [t_H*t_W, s_H*s_W], dim=1)

            template = template.transpose(1, 2).reshape(B, C, t_H, t_W)
            search = search.transpose(1, 2).reshape(B, C, s_H, s_W)

            out_template.append(template)
            out_search.append(search)

        return out_template, out_search

    def forward(self, template, search):
        template, search = self.forward_features(template, search)

        return template, search


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


class pvt_tiny(PyramidVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(
            pvt_tiny, self).__init__(
            patch_size=[4, 2, 2, 2], embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        # if pretrained:
        #     self.load_param("/home/dmmm/VscodeProject/FPI/pretrain_model/pvt_tiny.pth")


class pvt_small(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(
            pvt_small, self).__init__(
            patch_size=[4, 2, 2, 2], embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

        # if pretrained:
        #     self.load_param("/home/dmmm/VscodeProject/FPI/pretrain_model/pvt_small.pth")


def get_mixformer_pvt(
        vit_type, satellite_size=384, uav_size=128, **kwargs):
    img_size_s = satellite_size
    img_size_t = uav_size
    embed_dim = [64, 128, 320, 512]
    if vit_type == 'PvT-S':
        pvt = pvt_small()
    elif vit_type == 'PvT-T':
        pvt = pvt_tiny()
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'small_patch16' or 'large_patch16' or 'base_patch16'")

    return pvt, embed_dim

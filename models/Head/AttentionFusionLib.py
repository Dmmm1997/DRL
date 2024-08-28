# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
import torch
from timm.models.layers import trunc_normal_
from models.pos_utils import get_2d_sincos_pos_embed
from .utils import ChannelPool, xcorr_fast, xcorr_depthwise, xcorr_slow


class AttentionFusionLib(nn.Module):

    def __init__(
            self, input_ndim, fusion_type, nhead=8, attention_layer_num=4,
            enable_position_embedding=False, pos_length=[128, 384], enable_gc=False,
            mid_ndim=2048, dropout_rate=0.1, activation="relu", head_pool="avg", patch_size=16, **
            kwargs):
        super().__init__()
        featurefusion_layer = FeatureFusionLayer(
            input_ndim, nhead, mid_ndim, dropout_rate, activation, fusion_type=fusion_type)
        self.encoder = Encoder(featurefusion_layer, attention_layer_num)

        decoderCFA_layer = DecoderCFALayer(input_ndim, nhead, mid_ndim, dropout_rate, activation)
        # decoderCFA_norm = nn.LayerNorm(input_ndim)
        decoderCFA_norm = None
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self.d_model = input_ndim
        self.nhead = nhead
        self.fusion_type = fusion_type
        self.pool = ChannelPool(head_pool, input_ndim)
        self.enable_position_embedding = enable_position_embedding
        self.enable_gc = enable_gc

        self._reset_parameters()

        if enable_position_embedding:
            self.grid_size_uav = pos_length[0] // patch_size
            self.grid_size_satellite = pos_length[1] // patch_size
            self.num_patches_uav = self.grid_size_uav ** 2
            self.num_patches_satellite = self.grid_size_satellite ** 2
            self.pos_embed_uav = nn.Parameter(
                torch.zeros(1, self.num_patches_uav, input_ndim),
                requires_grad=False)
            self.pos_embed_satellite = nn.Parameter(
                torch.zeros(1, self.num_patches_satellite, input_ndim),
                requires_grad=False)
            self.init_pos_embed()

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_uav = get_2d_sincos_pos_embed(
            self.pos_embed_uav.shape[-1],
            int(self.num_patches_uav ** .5),
            cls_token=False)
        self.pos_embed_uav.data.copy_(torch.from_numpy(pos_embed_uav).float().unsqueeze(0))

        pos_embed_satellite = get_2d_sincos_pos_embed(
            self.pos_embed_satellite.shape[-1],
            int(self.num_patches_satellite ** .5),
            cls_token=False)
        self.pos_embed_satellite.data.copy_(
            torch.from_numpy(pos_embed_satellite).float().unsqueeze(0))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z, x):
        return self.forward_single(z[0], x)

    def forward_single(
        self,
        src_temp,
        src_search,
        mask_temp=None,
        mask_search=None,
        pos_temp=None,
        pos_search=None,
    ):
        temp_size = src_temp.shape[2]
        search_size = src_search.shape[2]
        search_out = src_search.shape[2]*src_search.shape[3]
        temp_out = src_temp.shape[2]*src_temp.shape[3]
        bs = src_temp.shape[0]
        if mask_temp != None:
            pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
            pos_search = pos_search.flatten(2).permute(2, 0, 1)
            mask_temp = mask_temp.flatten(1)
            mask_search = mask_search.flatten(1)

        if self.fusion_type == "CAT_SA":
            src_search = src_search.flatten(2).transpose(1, 2).contiguous()
            src_temp = src_temp.flatten(2).transpose(1, 2).contiguous()
            src_search = torch.cat((src_temp, src_search), dim=1).permute(1, 0, 2)  # 16*(49+36)*384
            if self.enable_position_embedding:
                # pos_search = self.pos_embed.repeat((bs, 1, 1))
                # pos_search = pos_search.flatten(2).permute(1, 0, 2)
                pos_search = torch.cat(
                    (self.pos_embed_uav, self.pos_embed_satellite),
                    dim=1).transpose(
                    0, 1)

        elif self.fusion_type == "CA":
            src_temp = src_temp.flatten(2).permute(2, 0, 1)
            src_search = src_search.flatten(2).permute(2, 0, 1)

            if self.enable_position_embedding:
                pos_temp = self.pos_embed_uav.transpose(0,1)
                pos_search = self.pos_embed_satellite.transpose(0,1)


        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search,
                                                  src1_key_padding_mask=mask_temp,
                                                  src2_key_padding_mask=mask_search,
                                                  pos_src1=pos_temp,
                                                  pos_src2=pos_search
                                                  )
        if self.fusion_type == "CAT_SA":
            memory_temp, memory_search = torch.split(memory_search, [temp_out, search_out], dim=0)

        # memory_search = self.decoder(memory_search, memory_temp,
        #                   tgt_key_padding_mask=mask_search,
        #                   memory_key_padding_mask=mask_temp,
        #                   pos_enc=pos_temp, pos_dec=pos_search
        #                   )

        if self.fusion_type == "CAT_SA":
            memory_search = memory_search.permute(1, 2, 0).contiguous()
            # memory_temp, memory_search = torch.split(memory_search, [temp_out, search_out], dim=2)
        else:
            memory_temp = memory_temp.permute(1,2,0).contiguous()
            memory_search = memory_search.permute(1,2,0).contiguous()
        
        memory_search = memory_search.transpose(1,2)
        memory_search = self.pool(memory_search).transpose(1,2)

        memory_search = memory_search.reshape(
            memory_search.shape[0],
            memory_search.shape[1],
            search_size, search_size).contiguous()


        
        # if self.enable_gc:
        #     heatmap = xcorr_fast(memory_temp, memory_search)*0.001
        # else:
        #     heatmap = self.pool(memory_search)

        # if self.fusion_type == "CAT_SA":
        #     hs_1 = memory_search.transpose(0, 1).contiguous()
        #     src_temp, src_search = torch.split(hs_1, [temp_out, search_out], dim=1)
        #     hs_1 = src_search
        # else:
        #     hs_1 = memory_search.transpose(0, 1).contiguous()
        # hs = self.pool(hs_1)
        # hs = hs.permute(0, 2, 1).contiguous()
        # b, c, s = hs.shape
        # h = w = np.sqrt(s)
        # h = int(h)
        # w = int(w)
        # hs = hs.view(b, 1, h, w)

        return memory_search, None


class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None,
                ):
        output1 = src1
        output2 = src2

        for layer in self.layers:
            output1, output2 = layer(output1, output2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=pos_src1, pos_src2=pos_src2)

        return output1, output2


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)


class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", fusion_type="CA"):
        super().__init__()

        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)

        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

        self.fusion_type = fusion_type

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None,
                     ):
        if self.fusion_type == "CA":  # src2 is search
            key = self.with_pos_embed(src1, pos_src1)
            query=self.with_pos_embed(src2, pos_src2)
            query2 = self.multihead_attn2(query = query, key =key, value = src2, 
                                         attn_mask=src1_mask,
                                         key_padding_mask=src1_key_padding_mask)[0]

            query2 = src2 + self.dropout22(query2)
            query2 = self.dropout22(query2)
            query2 = self.norm22(query2)
            src2 = self.linear22(self.dropout2(self.activation2(self.linear21(query2))))
            # src2 = src2 + self.dropout23(query2)
            # src2 = self.norm23(src2)

        elif self.fusion_type == "CAT_SA":
            q2 = k2 = self.with_pos_embed(src2, pos_src2)
            query2 = self.self_attn1(q2, k2, value=src2, attn_mask=src1_mask,
                                    key_padding_mask=src1_key_padding_mask)[0]
            
            query2 = src2 + self.dropout11(query2)
            query2 = self.norm11(query2)
            src2 = self.linear22(self.dropout2(self.activation2(self.linear21(query2))))
            # src2 = src2 + self.dropout23(src22)
            # src2 = self.norm23(src2)

        return src1, src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None,
                ):
        return self.forward_post(
            src1, src2, src1_mask, src2_mask, src1_key_padding_mask, src2_key_padding_mask,
            pos_src1, pos_src2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

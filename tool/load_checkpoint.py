import torch
from collections import OrderedDict

ckpt = torch.load("pretrain_model/mae_pretrain_vit_base.pth", map_location='cpu')
res_ckpt = ckpt["model"]
# res_ckpt = OrderedDict()
# for key, value in ckpt.items():
#     if "backbone.backbone" in key:
#         new_key = key.split("backbone.backbone.")[-1]
#         res_ckpt[new_key] = value

torch.save(res_ckpt, "pretrain_model/mae_pretrain_vit_base.pth")

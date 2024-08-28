import sys
sys.path.insert(0, "/home/dmmm/VscodeProject/FPI/")

from losses.clsloss import CenterBalanceLoss
import torch
from einops import rearrange
import cv2

cls_input = torch.zeros((1,1,384,384))
center_rate = torch.tensor([[0.99],[0.5]])
cls_loss = CenterBalanceLoss(33,15,1)
mask = cls_loss.create_mask(cls_input.size(), center_rate)

new_mask = rearrange(mask.squeeze(0), "c h w -> h w c")
# normalize
new_mask = new_mask/torch.max(new_mask)
new_mask = new_mask.cpu().numpy()*255

cv2.imwrite("tool/visual/loss_visual/cr33_ori384_x0.99_y0.5.jpg",new_mask)

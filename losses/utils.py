import numpy as np
import torch


def create_labels_2(size, rate, center_R):
    ratex, ratey = rate
    labels = np.zeros(size)
    b, c, h, w = size
    X = ratex * (h - 1)
    Y = ratey * (w - 1)
    intX = np.floor(X).reshape(-1, 1)
    intY = np.floor(Y).reshape(-1, 1)
    CenterXY = np.concatenate((intX, intY), axis=-1)
    for i in range(b):
        CenterX, CenterY = CenterXY[i]
        x1, x2, y1, y2 = CenterX, CenterX+center_R//2+1, CenterY, CenterY+center_R//2+1
        labels[i, 0, int(x1):int(x2), int(y1):int(y2)] = 1
    labels_torch = torch.from_numpy(labels).cuda().float()
    return labels_torch


def create_labels(size, rate, center_R):
    ratex, ratey = rate
    b, c, h, w = size
    labels = np.zeros((b,1,h,w))
    X = ratex * (h - 1)
    Y = ratey * (w - 1)
    intX = np.round(X).reshape(-1, 1)
    intY = np.round(Y).reshape(-1, 1)
    CenterXY = np.concatenate((intX, intY), axis=-1)
    for i in range(b):
        CenterX, CenterY = CenterXY[i]
        pad_right = pad_left = pad_top = pad_bottom = 0
        if CenterX+center_R//2 > h-1:
            pad_bottom = int(CenterX+center_R//2-(h-1))
        if CenterX-center_R//2 < 0:
            pad_top = int(-1*(CenterX-center_R//2))
        if CenterY+center_R//2 > h-1:
            pad_right = int(CenterY+center_R//2-(w-1))
        if CenterY-center_R//2 < 0:
            pad_left = int(-1*(CenterY-center_R//2))
        new_label = np.pad(
            labels[i, 0],
            ((pad_top, pad_bottom),
                (pad_left, pad_right)),
            'constant', constant_values=(-1, -1))
        new_center = [CenterX + pad_top, CenterY + pad_left]
        x1, x2, y1, y2 = new_center[0]-center_R//2,\
            new_center[0]+center_R//2+1,\
            new_center[1]-center_R//2,\
            new_center[1]+center_R//2+1
        label = new_label.copy()
        label[int(x1):int(x2), int(y1):int(y2)] = 1
        label_mask = new_label != -1
        new_label_out = label[label_mask].reshape(h, w)
        labels[i, :] = new_label_out
    labels_torch = torch.from_numpy(labels).cuda().float()
    return labels_torch

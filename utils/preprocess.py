import numpy as np
import torch

def random_crop(image_short, image_long, size):
    H = image_short.shape[1]
    W = image_short.shape[2]
    ps = size
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    image_short = image_short[:,yy:yy + ps, xx:xx + ps]
    image_long = image_long[:,yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2]
    return image_short, image_long

def pack_sony_raw(batch, ):
    H = batch.shape[1]
    W = batch.shape[2]

    out = torch.cat((batch[:, 0:H:2, 0:W:2], 
                     batch[:, 0:H:2, 1:W:2],
                     batch[:, 1:H:2, 1:W:2],
                     batch[:, 1:H:2, 0:W:2]), dim=0)
    return out

def adjust_black_level(batch, black_level=512, device=None):
    if not device:
        device = torch.device('cpu')
    batch = torch.maximum(batch - black_level, torch.Tensor([0]).to(device=device)) / (16383 - black_level)
    return batch
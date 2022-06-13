from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from einops import rearrange

class selfsupervise_loss(nn.Module):

    def __init__(self, heads=6):
        super(selfsupervise_loss, self).__init__()
        self.heads = heads
        self.smoothl1 = torch.nn.SmoothL1Loss()

    def forward(self, attns, smooth=1e-40):
        layer = len(attns)
        for i in range(layer):
            attni = attns[i]  # b h n n
            b, h, n, d = attni.shape
            #attentionmap_visual(attni)
            # entropy loss
            log_attni = torch.log2(attni + smooth)
            entropy = -1 * torch.sum(attni * log_attni, dim=-1) / torch.log2(torch.tensor(n*1.0)) # b h n
            entropy_min = torch.min(entropy, dim=-1)[0]  # b h
            p_loss = (entropy_min-0.9).clamp_min(0)*(1/0.1)

            # symmetry loss
            attni_t = attni.permute(0, 1, 3, 2)
            #distance = torch.abs(attni_t*n - attni*n)  # b h n n
            #distance = torch.sum(distance, dim=-1)/n  # b h n
            #s_loss = torch.sum(distance, dim=-1)/n  # b h
            s_loss = self.smoothl1(attni*n, attni_t*n).clamp_min(0.1)

            if i == 0:
                loss = 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)
            else:
                loss += 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)

        return loss / layer

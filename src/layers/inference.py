# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveQuant(nn.Module):
    def __init__(self, mode="ste", half_bin=0.5, quant_fn=torch.round):
        super().__init__()
        assert mode in ["ste", "noise"], "Unsupported mode"
        self.mode = mode
        self.half_bin = half_bin
        self.quant_fn = quant_fn

    def forward(self, x):
        if self.mode == "ste":
            if self.training:
                return (self.quant_fn(x) - x).detach() + x  # straight-through estimator
            else:
                return self.quant_fn(x)  # hard rounding
        elif self.mode == "noise":
            if self.training:
                noise = torch.empty_like(x).uniform_(-self.half_bin, self.half_bin)
                return x + noise
            else:
                return self.quant_fn(x)

def clamp_reciprocal_with_quant(q_dec, y, min_val):
    q_dec = torch.clamp_min(q_dec, min_val)
    q_enc = torch.reciprocal(q_dec)
    y = y * q_enc
    return q_dec, y

def add_and_multiply(y_hat_0, y_hat_1, q_dec):
    y_hat = y_hat_0 + y_hat_1
    y_hat = y_hat * q_dec
    return y_hat

def replicate_pad(x, pad_b, pad_r):
    if pad_b == 0 and pad_r == 0:
        return x
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")

############for decompress##############
def combine_for_reading_2x(x, mask, inplace=False):
    x = x * mask
    x0, x1 = x.chunk(2, 1)
    return x0 + x1


def restore_y_2x(y, means, mask):
    return (torch.cat((y, y), dim=1) + means) * mask


def restore_y_2x_with_cat_after(y, means, mask, to_cat):
    out = (torch.cat((y, y), dim=1) + means) * mask
    return out, torch.cat((out, to_cat), dim=1)


def restore_y_4x(y, means, mask):
    return (torch.cat((y, y, y, y), dim=1) + means) * mask


def build_index_dec(scales, scale_min, scale_max, log_scale_min, log_step_recip, skip_thres=None):

    skip_cond = None
    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    if skip_thres is not None:
        skip_cond = scales > skip_thres
    return indexes, skip_cond


def build_index_enc(symbols, scales, scale_min, scale_max, log_scale_min,
                    log_step_recip, skip_thres=None):

    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    out = (symbols << 8) + indexes
    if skip_thres is not None:
        skip_cond = scales > skip_thres
        out = out[skip_cond]
    return out


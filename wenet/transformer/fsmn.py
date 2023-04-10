#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Ao Zhang (Northwestern Polytechnical University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FsmnLayer(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, left_frame=1, right_frame=1, left_dilation=1, right_dilation=1):
        super(FsmnLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.left_dilation = left_dilation
        self.right_dilation = right_dilation
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        if left_frame > 0:
            self.pad_left = nn.ConstantPad1d([left_dilation*left_frame, 0], 0)
            self.conv_left = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=left_frame+1,
                                       dilation=left_dilation, bias=False, groups=hidden_dim)
        if right_frame > 0:
            self.pad_right = nn.ConstantPad1d(
                [-right_dilation, right_dilation*right_frame], 0)
            self.conv_right = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=right_frame,
                                        dilation=right_dilation, bias=False, groups=hidden_dim)
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)
        self.cache_size = left_frame*left_dilation+right_frame*right_dilation+1
        self.cache = torch.zeros([1, self.hidden_dim, self.cache_size])

    def forward(self, x, hidden=None):
        x_data = x.transpose(1, 2)
        p_in = self.conv_in(x_data)
        if self.left_frame > 0:
            p_left = self.pad_left(p_in)
            p_left = self.conv_left(p_left)
        else:
            p_left = 0
        if self.right_frame > 0:
            p_right = self.pad_right(p_in)
            p_right = self.conv_right(p_right)
        else:
            p_right = 0
        p_out = p_in + p_right+p_left
        if hidden is not None:
            p_out = hidden + p_out
        out = F.relu(self.conv_out(p_out))
        out = out.transpose(1,2)
        return out, p_out

    def forwardstreaming(self, x, cache):
        x_data = x.transpose(1, 2)
        p_in = self.conv_in(x_data)
        if self.left_frame > 0:
            p_left = self.pad_left(p_in)
            p_left = self.conv_left(p_left)
        else:
            p_left = 0
        if self.right_frame > 0:
            p_right = self.pad_right(p_in)
            p_right = self.conv_right(p_right)
        else:
            p_right = 0
        p_out = p_in + p_right + p_left
        out = F.relu(self.conv_out(p_out))
        out = out.transpose(1,2)
        return out, new_cache
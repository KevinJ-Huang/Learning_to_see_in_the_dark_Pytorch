#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by xuchongbo at 20171130 in Meitu.
"""

import torch.nn as nn


class AccScore(nn.Module):
    def __init__(self):
        super(AccScore, self).__init__()

    def forward(self, output, target):
        if len(output.data.size()) == 3:
            total_size = output.data.size(0) * output.data.size(1) * output.data.size(2)
        elif len(output.data.size()) == 4:
            total_size = output.data.size(0) * output.data.size(1) * output.data.size(2) * output.data.size(3)

        score = (output.data == target.data).sum() / float(total_size)

        return score


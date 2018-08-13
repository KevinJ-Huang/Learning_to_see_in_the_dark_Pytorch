#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xuchongbo at 20171130 in Meitu.
"""

import torch.nn as nn


class IouScore(nn.Module):
    def __init__(self, nclass, is_decouple=False):
        super(IouScore, self).__init__()
        self.nclass = nclass
        self.is_decouple = is_decouple

    def _iou(self, a, b, label):
        s = (a == label) + (b == label)
        ins = (s == 2).sum()
        union = (s >= 1).sum()
        return ins*1.0 / union if union > 0 else 0

    def forward(self, output, target):
        """
        output: B*1*H*W
        target: B*H*W
        """
        score_dict = {}
        for idx in range(self.nclass):
            if self.is_decouple:
                score_dict['class_%s' % idx] = self._iou(output[:, idx, :, :].data, target[:, idx, :, :].data, 1.0)
            else:
                score_dict['class_%s' % idx] = self._iou(output.data, target.data, idx)

        return score_dict 

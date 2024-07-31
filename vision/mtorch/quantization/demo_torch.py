#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   demo_torch.py
@Time    :   2024/07/18 09:10:27
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module


import torch

import torch.nn as nn

from torchvision.models import regnet

from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver






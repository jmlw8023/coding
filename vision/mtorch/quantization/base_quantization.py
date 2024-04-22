#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   base_quantization.py
@Time    :   2024/04/16 16:55:58
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''
# link: https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html
# import module

import torch
from torch import nn
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer import (xnnpack_quantizer)



class M(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.linear = nn.Linear()
        
    
    def forward(self, x):
        
        return self.linear(x)



def test_demo():
    
    example_in = (torch.randn(1, 5), )
    print(example_in)
    
    m = M().eval() 
    
    m = capture_pre_autograd_graph(m, *example_in)
    
    # 量化
    






if __name__ == '__main__':
    
    
    
    
    pass








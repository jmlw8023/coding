

import os, sys

import torch
import torch.nn as nn
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt

from ultralytics.nn.modules.conv import CBAM, Conv, GhostConv


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def test_module():

    img = torch.randn(1, 3, 28, 28)

    conv = Conv(3, 64, 3, 1)
    print(conv)
    c_out = conv(img)   # torch.Size([1, 64, 28, 28])
    print(f'conv out shape = {c_out.shape}')

    ghost = GhostConv(64, 128)
    print(ghost)
    g_out = ghost(c_out)
    print(f'ghost out shape = {g_out.shape}')
    
    cbam = CBAM(128, kernel_size=3)
    print(cbam)
    cb_out = cbam(g_out)
    print(f'cbam out shape = {cb_out.shape}')
    



class MNetwork(nn.Module):
    def __init__(self, parms=None):
        super().__init__()
        
        self.flatten = nn.Flatten()
        
        self.leners = nn.Sequential(
            nn.Linear(24 * 24, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        num = self.leners(x)
        return num


def test_class_moudle():
    
    model = MNetwork().to(device)
    print(model)
    # 用模型的 parameters() 或 named_parameters() 方法访问所有参数
    for name, param in model.named_parameters():
        # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        print(f"Layer: {name} | Size: {param.size()}\n")

    img = torch.rand(1, 24, 24, device=device)
    y_res = model(img)
    
    y_pred_lis = nn.Softmax(dim=1)(y_res)
    
    y_pred = y_pred_lis.argmax(1)
    print(f'prediction class = {y_pred.item()}')



device = ('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')

if __name__ == '__main__':
    
    print(f'now using {device} device')
    
    test_module()
    # 测试网络
    # test_class_moudle()
    
    
    pass









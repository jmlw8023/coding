#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   numpy_bp.py
@Time    :   2024/07/17 09:02:18
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''
"""## 
(1)隐含层层数:三层 BP 神经网络
(2)隐含层神经元个数:15 ，输出层节点为 1
(3)初始权值:尽量使加权求和后的输出值在 0 的附近
(4)激活函数:归一化后的数据落在(-1,1)之间
(5) 学习率:取值范围一般在(0.01, 0.08)之间。本次取学习速率=0.01。
(6)期望误差:采用均方误差函数

    
"""

# import module


import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 输出层
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # 使用tanh作为激活函数
        x = self.fc2(x)
        return x

# 设定超参数
input_size = 4  # 假设输入维度为4
hidden_size = 15
output_size = 1
learning_rate = 0.01

# 实例化网络
net = Net(input_size, hidden_size, output_size)

# 损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # 随机梯度下降优化器

# 假设你已经有了输入数据X和标签Y
# X = torch.tensor([[...]], dtype=torch.float32)  # 输入数据
# Y = torch.tensor([[...]], dtype=torch.float32)  # 输出标签

# 训练网络
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清零梯度
    outputs = net(X)       # 前向传播
    loss = criterion(outputs, Y)  # 计算损失
    loss.backward()       # 反向传播
    optimizer.step()      # 更新权重

    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))





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


import numpy as np

def sigmoid(x):
    return np.tanh(x)  # 使用tanh作为激活函数

def sigmoid_derivative(x):
    return 1 - np.power(x, 2)  # tanh的导数

def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01  # 输入层到隐藏层的权重
    b1 = np.zeros((1, hidden_size))  # 隐藏层的偏置
    W2 = np.random.randn(hidden_size, output_size) * 0.01  # 隐藏层到输出层的权重
    b2 = np.zeros((1, output_size))  # 输出层的偏置
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def backpropagation(X, Y, A1, A2, W1, W2, learning_rate):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2

def train(X, Y, iterations, input_size, hidden_size, output_size, learning_rate):
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    
    for i in range(iterations):
        _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
        cost = np.mean(np.square(A2 - Y))
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")
        
        W1, b1, W2, b2 = backpropagation(X, Y, _, A2, W1, W2, learning_rate)
    
    return W1, b1, W2, b2

# 假设输入数据X和标签Y已经准备好
# X = np.array([[...]])  # 输入数据
# Y = np.array([[...]])  # 输出标签

# 使用以下参数调用train函数
input_size = 4  # 假设输入维度为4
hidden_size = 15
output_size = 1
learning_rate = 0.01
iterations = 10000

# W1, b1, W2, b2 = train(X, Y, iterations, input_size, hidden_size, output_size, learning_rate)





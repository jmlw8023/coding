#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   LinearRegression.py
@Time    :   2024/05/08 17:06:24
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression



# 生成模拟数据
X, y = make_regression(n_samples=100, n_features=1, noise=0.3)

# 创建并训练模型
model = LinearRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X[:5])


print("x = ", X)
print("y = ", y)
print("预测结果：", y_pred)

# 绘制线性回归线
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Linear Regression Line')
plt.title('Linear Regression Example')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   pca.py
@Time    :   2024/05/08 17:16:50
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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


# 生成模拟数据或使用真实数据，这里使用随机数据
X = np.random.rand(100, 10)

# 数据预处理，标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 执行PCA降维
pca = PCA(n_components=2)  # 降到二维
principal_components = pca.fit_transform(X_scaled)

# 查看降维后的数据
print("降维后的前两主成分：\n", principal_components[:5])


plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Random Data')
plt.grid(True)
plt.show()

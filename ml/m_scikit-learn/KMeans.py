#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   KMeans.py
@Time    :   2024/05/08 17:15:20
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

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成模拟数据
X, _ = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.60)

# 创建并训练模型
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# 预测数据所属的簇
predictions = kmeans.predict(X)

print("簇标签：", predictions)

plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

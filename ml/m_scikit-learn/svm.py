#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   svm.py
@Time    :   2024/05/08 17:16:07
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


from sklearn import svm
from sklearn import datasets

# # 加载鸢尾花数据集
# iris = datasets.load_iris()
# X, y = iris.data, iris.target

# # 创建并训练模型
# clf = svm.SVC(kernel='linear')  # 线性核函数
# clf.fit(X, y)

# # 预测新数据点的类别
# new_data = [[5.1, 3.5, 1.4, 0.2]]  # 一个样本特征
# prediction = clf.predict(new_data)
# print("预测类别：", prediction)


# 生成样本数据
X_svm, y_svm = datasets.make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# 创建SVM模型并拟合数据
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_svm, y_svm)

# 绘制决策边界
def plot_svc_decision_boundary(model, ax=None, plot_support=True):
    """Plot the decision boundaries for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格以评估模型
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # 绘制决策边界和边际
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# 绘制SVM决策边界
plt.scatter(X_svm[:, 0], X_svm[:, 1], c=y_svm, cmap='viridis', edgecolors='k')
plot_svc_decision_boundary(clf)
plt.title('SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


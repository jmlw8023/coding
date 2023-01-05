# -*- encoding: utf-8 -*-
'''
@File    :   demo01.py
@Time    :   2023/01/05 10:27:51
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :
'''

# import packets
import numpy as np
import matplotlib.pyplot as plt


# 从0 ~ 2 中，抽取50个点 
x = np.linspace(0, 2, 50)

# 
plt.plot(x, x**2)

# x 轴 和y轴
plt.xlabel('x label')
plt.ylabel('y label')
# 图例
plt.legend()

plt.show()


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
import matplotlib
import matplotlib.pyplot as plt

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class Demo01():
    def __init__(self, random=False):
        
        if random:

            self.data = np.random.randint(-10, 20, 20)
            print(self.data.tolist())
        else:
            # # a = [9, -8, 8, -5, 6, 5, 5, -7, -4, -1, 7, 5, 18, 5, 13, 19, 9, -6, 7, -5]
            self.data = [13, -8, 1, 5, 9, 8, 16, 19, 18, -6, 19, 15, 1, 7, -3, -8, 2, 5, -8, 1]
        

        # # 从0 ~ 2 中，抽取50个点 
        # x = np.linspace(0, 2, 50)
        # # 
        # plt.plot(x, x**2)
    # @classmethod
    def test(self):
   
        x = np.arange(len(self.data))
        y = np.bincount(np.abs(self.data))

        # print(type(x))
        # print(x.tolist())
        # print(y.tolist())
        # print(len(x))
        # print(len(y))

        # plt.bar(x, y)
        plt.bar(x, y, width=0.8, label='个数')
        plt.plot(x, y, 'ro')

        # plt.xlim(left=1)
        # plt.ylim()

        # x 轴 和y轴
        plt.xlabel('x label')
        plt.ylabel('y label')
        # 图标题
        plt.title('This is first pyplot demo')
        # 图例
        plt.legend()

        for x1, y1, num in zip(x, y, y):
            # x1 为位置， y1 是高度位置， num是对应字符串， ha为 horizontalalignment
            plt.text(x1, y1 + 0.1, num, ha='center', fontsize=12)

        # 限制坐标阈值  x, y
        plt.axis([-1, 20, 0, 6])     # [xmin, xmax, ymin, ymax]

        # 保存结果图片
        plt.savefig('./res.png', dpi=300)
        plt.show()



if __name__ == '__main__':
    
    d = Demo01()
    d.test()

# -*- encoding: utf-8 -*-
'''
@File    :   scatter_plot_and.py
@Time    :   2023/02/13 15:10:47
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
# import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
# 设置中文
mpl.rcParams['font.sans-serif'] = ['STSong']
# 显示负号
mpl.rcParams['axes.unicode_minus'] = False


# 曲线图例子
def plot_demo():
    # 创建数据
    x = np.linspace(0.01, 12, 1000)
    y1 = np.cos(x)
    y2 = np.sin(x)
    y3 = np.random.rand(1000)
    # 1行 4列
    fig, ax = plt.subplots(1, 4, figsize=(12, 8))

    ax[0].plot(x, y1, ls='-', label='y1 plot figure')
    ax[0].plot(x, y2, ls=':', label='y2 plot figure')
    ax[1].scatter(x, y3, label='y3 scatter figure')   

    # 添加图形内容标签
    ax[0].text(3.10, 1, 'y2=sin(x)', weight='bold', color='b')
    ax[0].text(7.10, 1, 'y1=cos(x)', weight='bold', color='r')

    # 设置X轴范围
    ax[1].set_xlim(0, 10)
    ax[3].set_xlim(0, 8)
    ax[2].set_xlim(0, 10)
    ax[2].set_ylim(-2, 3)

    # 设置图标题
    ax[0].set_title('plot 曲线图')

    ax[0].set_xlabel('数值')

    # 绘制水平参考线
    ax[3].axhline(y=1.1, c='b', ls='-', lw=2)
    ax[3].axvline(x=3.3, c='g', ls='-', lw=2)

    # 设置注释
    ax[2].plot(x, y1)
    ax[2].annotate('maximum最大值', xy=(np.pi/5, 1.0),xytext=((np.pi/2)+0.15,1.5),weight="bold",color="b", arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color="b"))

    # 绘制垂直于x轴的参考区域
    ax[3].axvspan(xmin=1.2, xmax=2.5, facecolor='y', alpha=0.3)

    # 设置网线格
    ax[0].grid(linestyle=':', color='gray')
    ax[3].grid(linestyle='--', color='r')

    # 去掉坐标轴
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_color(None)

    # ax[0].legend(loc=1)          # 左上
    ax[0].legend(loc=3)          # 左下
    ax[1].legend(loc=4)               # 右下
    # ax[1].legend(loc=10) # 居中

    plt.show()

# 柱状图例子
def bar_demo():

    # some simple data
    x = [1,2,3,4,5,6,7,8]
    y = [3,1,4,5,8,9,7,2]

    box_weight = np.random.randint(0, 12, 100)
    bins = range(1, 13)


    kinds = "简易箱","保温箱","行李箱","密封箱"
    colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3"]
    soldNums = [0.05,0.45,0.15,0.35]

    barSlices = 12
    theta = np.linspace(0.0, 2*np.pi, barSlices, endpoint=False)
    r = 30*np.random.rand(barSlices)

    x1 = np.linspace(0.1, 0.8, 6)
    y1 = np.exp(x1)

    xx = [1,2,3,4,5]
    yy = [6,10,4,5,1]
    yy1 = [2,6,3,8,5]

    fig, ax = plt.subplots(2,3, figsize=(10, 8))

    # create bar    
    # 第1行 第2列
    ax[0, 1].bar(x,y,align="center",color="c",tick_label=["q","a","c","e","r","j","b","p"],hatch="/")
    
    # 第2行 第1列
    # ax[1, 0].barh(x,y,align="center",color="g",tick_label=["q","a","c","e","r","j","b","p"],hatch="/")
    ax[1][0].barh(x, y, align="center",color="g",tick_label=["q","a","c","e","r","j","b","p"],hatch="/")

    # 柱状图
    # ax[0, 2].hist(box_weight, bins=bins, color='r', histtype='bar', rwidth=0.8, alpha=0.7)
    ax[0][2].hist(box_weight, bins=bins, label=bins, color='r', histtype='bar', rwidth=0.8, alpha=0.7)
    
    
    #  饼状图
    ax[1][1].pie(soldNums, labels=kinds, autopct='%3.1f%%', startangle=60, colors=colors)


    # 绘制误差棒图
    ax[1][2].errorbar(x1, y1, fmt='bo:', yerr=0.3, xerr=0.03)

    # 气泡图
    # plt.axes(projection='polar')
    # ax[0][0].setpo.polar(theta, color='chartreuse', linewidth=2, marker='*', mfc='b', ms=10)
    # # ax0 = plt.subplot(111, polar=True)

    # 堆积柱状图
    ax[0][0].bar(xx, yy, align="center", color="#66c2a5", tick_label=["A","B","C","D","E"], label="班级A")
    ax[0, 0].bar(xx, yy1, align="center", bottom=yy, color="#8da0cb", label="班级B")
    # set x,y_axis label
    ax[0][0].set_xlabel("测试难度")
    ax[0, 0].set_ylabel("试卷份数")
    ax[0, 0].legend()



    # 设置刻度范围
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    xmajor = MultipleLocator(1)
    ax[0, 2].xaxis.set_major_locator(xmajor)

    # ax[0][2].set_xlabel(bins)

    # set x,y_axis label
    ax[0, 1].set_xlabel("箱子编号")
    ax[0, 1].set_ylabel("箱子重量(kg)")

    ax[1, 0].set_ylabel("箱子编号")
    ax[1, 0].set_xlabel("箱子重量(kg)")

    ax[0, 2].set_ylabel("销售个数(个)")
    ax[0, 2].set_xlabel("箱子重量(kg)")

    ax[1, 1].set_title("不同类型箱子的销售数量占比")

    plt.tight_layout()
    # plt.legend()

    plt.show()



def main():

    # plot_demo()
    bar_demo()
    



if __name__ == '__main__':
    main()
    
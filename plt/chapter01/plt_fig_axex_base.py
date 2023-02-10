# -*- encoding: utf-8 -*-
'''
@File    :   plt_fig_axex_base.py
@Time    :   2023/02/10 09:43:06
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.image as mpimg 
from matplotlib.cbook import get_sample_data
from matplotlib.patches import Circle
# TextArea=用于创建文字元素，OffsetImage用于创建图片元素，导入之后创建画布并调整了坐标轴范围
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)

# MultipleLocator和FormatStrFormatter  一个负责位置，一个负责样式
from matplotlib.ticker import MultipleLocator, FormatStrFormatter




# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STSong']
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# 可以显示符号
mpl.rcParams['axes.unicode_minus'] = False

# 中文字体设定
# 
def set_chinese():

    # 方法一：
    # # 字体不是中文的是不能解决！！， 用于后文font指定字体
    # font = mpl.font_manager.FontProperties(fname='../datas/font/simkai.ttf')   
    # plt.title('中文字体', fontproperties=font)
    # plt.show()

    # 方法二：
    # # 打印matplotlib中内置的字体
    # font_list = mpl.font_manager.FontManager().ttflist
    # print(font_list)
    """  可用中文字体
        # 'STSong'
        # 'STFangsong'
        # 'FZShuTi'
        # 'Microsoft YaHei'
        # -------------------'Times New Roman'
    """
    # # 指定字体即可
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 6), dpi=100)
    plt.title('测试中文字体', fontsize=12)
    plt.show()



# 画布figure 与 axes的关系
def fig_axes_demo():
    
    # 创建3行，3列的图到figure中
    fig, axes = plt.subplots(3, 3, figsize=(18, 9))

    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    # 绘制数据到 第1行 第3个figure中
    ax1[2].barh(range(5), range(5, 10))
    # 绘制数据到 第三行 第1个figure中
    ax2[0].barh(range(5), range(5, 10))
    # 绘制数据到 第三行 第2个figure中
    ax3[1].barh(range(5), range(5, 10))

    # 紧凑型
    fig.tight_layout()
    # plt.tight_layout()

    plt.show()


# 样式设定
def style_set():
    
    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.linspace(0.1, 20, 100)
    print(x)
    y = np.sin(x)
    # y = [i ** 2 for i in x]
    
    ax.plot(x, y, label='random label')
    # 增加样式
    """
    # 样式名称， 使用方法 plt.stype.use('样式名称')
    # Solarize_Light2
    # _classic_test_patch
    # bmh
    # classic
    # dark_background
    # fast
    # fivethirtyeight
    # ggplot
    # grayscale
    # seaborn
    # seaborn-bright
    # seaborn-colorblind
    # seaborn-dark
    # seaborn-dark-palette
    # seaborn-darkgrid
    # seaborn-deep
    # seaborn-muted
    # seaborn-notebook
    # seaborn-paper
    # seaborn-pastel
    # seaborn-poster
    # seaborn-talk
    # seaborn-ticks
    # seaborn-white
    # seaborn-whitegrid
    # tableau-colorblind10
    """
    fig.legend()
    # fig.set_tight_layout(True)
    plt.style.use('seaborn-white')
    plt.show()



# plot图  图例设置
def plot_demo():

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # 准备数据
    # 0.5 -- 5 均匀取100个数
    x1 = np.linspace(0.5, 5, 100)
    y1 = np.sin(x1)
    # 正态分布中随机取 100
    x2 =  np.linspace(2, 6, 100)
    y2 = np.cos(x2)
    # y2 = [i for i in range(len(x2))]

    ax.plot(x1, y1, ls='-', lw=2, label='x1 plot figure')
    ax.plot(x2, y2, ls='-', lw=2, label='x2 plot figure')

    # 去掉边框线
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)

    # ax.set_title('图例的表达方式')
    # fontstyle修改字体样式，italic是斜体，oblique是倾斜
    # , heavy可以加粗 'light', 'normal', 'medium', 'semibold','bold', 'heavy', 'black'，
    #     # fontsize参数调整字体大小，数字越大字体越大
    #     修改字体可以使用family参数来实现, 标题位置可以通过loc参数调整，比如loc居左
    ax.set_title('图例的表达方式',fontsize=22,color = 'blue',fontstyle='italic',fontweight = "heavy",family = "STSong", loc ='right')
    
    # 增加图例  loc 坐标轴象限的位置

    # plt.legend(fontsize=12, loc=10)
    #  frameon=False 是否保留图例边框, facecolor='red' 图例填充颜色
    plt.legend(fontsize=12, loc=3, bbox_to_anchor=(0.1, 0.5), edgecolor='blue', frameon=True, facecolor='red')

    plt.show()



def show_img():

    fig, ax = plt.subplots(figsize=(8,6), dpi=100)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # # 打开图片
    img_path = r'../datas/font/heart.jpg'
    # img_arr = mpimg.imread('../datas/font/heart.jpg')
    # print(img_arr.shape)    # (640, 640, 3)
    # with get_sample_data(img_path) as f_img:

    img_arr = plt.imread(img_path, format='jpg')

    img_box = OffsetImage(img_arr, zoom=0.2)
    # 画布中的坐标系转化为一个坐标系对象
    img_box.image.axes = ax

    annbb = AnnotationBbox(
        img_box,
        [0.4, 0.5], # X轴
        xybox=(60., -10.),
        xycoords='data',
        boxcoords='offset points',
        pad=1.2 # 图片框中的图片填充边距
    )
    ax.add_artist(annbb)

    txt_box = TextArea(r'这是第一个画的', textprops= dict(fontsize = 12))
    anb = AnnotationBbox(
        txt_box,
        [0.72, 0.5],
        xybox=(0.8, 0.85),
        xycoords='data',
        boxcoords=('axes fraction', 'data'),
        box_alignment=(0.1, 0.6),
        arrowprops=dict(arrowstyle='->', color='deeppink')
    )
    ax.add_artist(anb)

    ax.set_title('绘制图到plt中')
    # fig.set_tight_layout(True)

    plt.style.use('fivethirtyeight')   # default样式 会与开始设定的字体中文相冲突！！！！！！

    plt.show()

# 背景图
def background_img():

    img_path = r'../datas/font/heart.jpg'
    # # 打开图片
    img = plt.imread(img_path)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)     # figsize=(8, 6)  (w, h)

    ax.imshow(img, extent=[0, 5.5, 0, 7.5])   # x, y ---> extent参数控制xy轴坐标范围

    #添加文字
    ax.text(1.3, 6.5,"2023又是一个开始",fontsize = 30,color = '#1FEFFF',
        verticalalignment = 'center',horizontalalignment = 'center')

    # 去掉边框线
    # ax.spines['right'].set_color('None')    # 使用 ax.splines['right'] 获取最右边的 Spine对象， 之后就可以通过设置颜色去掉边界框 
    ax.spines['right'].set_color('red')     # 设置为红色边框线
    ax.spines['top'].set_color('None')
    ax.spines['left'].set_color('None')
    ax.spines['bottom'].set_color('None')



    # # 去掉坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def data_show_demo():

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    # # 设置左边边框线居中
    # ax.set_yticklabels(['', 'J', 'o', 'l', 'l', 'Y'])
    # ax.spines['left'].set_position(('data', 0.6))
    # 去掉右边、顶部 刻度线
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)

    # 把刻度读取出来
    ax = plt.gca()
    # 设置刻度范围
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    # from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    # MultipleLocator和FormatStrFormatter  一个负责位置，一个负责样式
    # 主刻度值
    xmajorLocator = MultipleLocator(1)  # 主刻度值之间间隔 为 1
    ymajorLocator = MultipleLocator(1)  # 主刻度值之间间隔 为 1
    xminorLocator = MultipleLocator(0.25)  # 副刻度值之间间隔 为 0.25
    yminorLocator = MultipleLocator(0.25)  # 副刻度值之间间隔 为 0.25

    # 设置刻度位置
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    # 设置刻度值  # 副刻度值
    xminorFormatter = FormatStrFormatter('%0.1f')
    ax.xaxis.set_minor_formatter(xminorFormatter)
    
    yminorFormatter = FormatStrFormatter('%0.2f')
    ax.yaxis.set_minor_formatter(yminorFormatter)

    # 突出主刻度值
    ax.tick_params(which='major', length=8, labelsize=10)
    ax.tick_params(which='minor', length=4)
    

    # 设置坐标轴标签
    ax.set_xlabel('x轴', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='blue',lw=1 ,alpha=0.7))
    ax.set_ylabel('y轴', fontsize=16)

    # 缩紧图，空白去掉
    fig.set_tight_layout(True)

    plt.show()


def main():

    # fig_axes_demo()
    # set_chinese()
    # style_set()
    # show_img()
    # background_img()
    # data_show_demo()
    plot_demo()



    pass




if __name__ == '__main__':
    main()
    






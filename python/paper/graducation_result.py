
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def txt_read(txt_path, txt_shuffix='.txt'):

    map50 = []
    map595 = []
    if os.path.isfile(txt_path):
        # if txt_path.endwith(txt_path):
            with open(txt_path) as f:
                # lines = f.readline()      # 0.10362 0.046976
                lines = f.readlines()
                # print(lines)
            for line in lines:
                m1, m2= line.strip().split()
                # print(m1, ', ',  m2)
                map50.append(float(m1))
                map595.append(float(m2))
             
    return map50, map595


def all_map():
    v3_lis = ['YOLOv3s_mAP0.5', 'YOLOv3s_mAP.5:.95', 'YOLOv3x_mAP0.5', 'YOLOv3x_mAP.5:.95']
    v4_lis = ['YOLOv4s_mAP0.5', 'YOLOv4s_mAP.5:.95', 'YOLOv4x_mAP0.5', 'YOLOv4x_mAP.5:.95']
    v5_lis = ['YOLOv5s_mAP0.5', 'YOLOv5s_mAP.5:.95', 'YOLOv5x_mAP0.5', 'YOLOv5x_mAP.5:.95']
    v6_lis = ['YOLOv6x_mAP0.5', 'YOLOv6x_mAP.5:.95','YOLOv6s_mAP0.5', 'YOLOv6s_mAP.5:.95']
    v7_lis = ['YOLOv7s_mAP0.5', 'YOLOv7s_mAP.5:.95', 'YOLOv7x_mAP0.5', 'YOLOv7x_mAP.5:.95']

    name_label = v3_lis + v4_lis + v5_lis + v6_lis + v7_lis
    # txt_read(txt_path)

    # map_50, map_595 = txt_read(txt_path)
    v3s_map_50, v3s_map_595 = txt_read(r'data/v3x.txt')
    v3x_map_50, v3x_map_595 = txt_read(r'data/v3x.txt')
    v4s_map_50, v4s_map_595 = txt_read(r'data/v4s.txt')
    v4x_map_50, v4x_map_595 = txt_read(r'data/v4x.txt')
    v5s_map_50, v5s_map_595 = txt_read(r'data/v5x.txt')
    v5x_map_50, v5x_map_595 = txt_read(r'data/v5x.txt')
    v6s_map_50, v6s_map_595 = txt_read(r'data/v6s.txt')
    v6x_map_50, v6x_map_595 = txt_read(r'data/v6x.txt')
    v7s_map_50, v7s_map_595 = txt_read(r'data/v7s.txt')
    v7x_map_50, v7x_map_595 = txt_read(r'data/v7x.txt')

    # print(map_50)
    # x = [i for i  in range(len(v5s_map_50))]
    v3s_x = [i for i  in range(len(v3s_map_50))]
    v4s_x = [i for i  in range(len(v4s_map_50))]
    v5s_x = [i for i  in range(len(v5s_map_50))]
    v6s_x = [i for i  in range(len(v6s_map_50))]
    v7s_x = [i for i  in range(len(v7s_map_50))]
    v3x_x = [i for i  in range(len(v3x_map_50))]
    v4x_x = [i for i  in range(len(v4x_map_50))]
    v5x_x = [i for i  in range(len(v5x_map_50))]
    v6x_x = [i for i  in range(len(v6x_map_50))]
    v7x_x = [i for i  in range(len(v7x_map_50))]
    # x = [i for i  in range(len(map_50)) if i % 2 == 0]
    print('v3s -> ',len(v3s_x))
    print('v3x -> ', len(v3x_x))
    print('v4s -> ',len(v4s_x))
    print('v4x -> ', len(v4x_x))
    print('v5s -> ',len(v5s_x))
    print('v5x -> ', len(v5x_x))
    print('v6x -> ', len(v6x_x))    
    print('v6s -> ',len(v6s_x))
    print('v7x -> ', len(v7x_x))
    print('v7s -> ',len(v7s_x))

    v5s5 = v5s_map_50
    v5s9 = v5s_map_595
    # print(v5s5)
    fig, ax = plt.subplots()

 
    # plt.plot(x, v5s5)
    # plt.plot(x, v5s9)

    # ax.plot(v3s_x, v3s_map_50, lw=1.5)
    # ax.plot(v3s_x, v3s_map_595, ls=':', lw=1.5)
    # ax.plot(v4s_x, v4s_map_50, lw=1.5)
    # ax.plot(v4s_x, v4s_map_595, ls=':', lw=1.5)
    # ax.plot(v5s_x, v5s_map_50, lw=1.5)
    # ax.plot(v5s_x, v5s_map_50, ls=':', lw=1.5)
    # ax.plot(v6s_x, v6s_map_50, lw=1.5)
    # ax.plot(v6s_x, v6s_map_595, ls=':', lw=1.5)
    # ax.plot(v7s_x, v7s_map_50, lw=1.5)
    # ax.plot(v7s_x, v7s_map_595, ls=':', lw=1.5)

    # ax.plot(v3x_x, v3x_map_50, lw=1.5)
    # ax.plot(v3x_x, v3x_map_595, ls=':', lw=1.5)
    # ax.plot(v4x_x, v4x_map_50, lw=1.5)
    # ax.plot(v4x_x, v4x_map_595, ls=':', lw=1.5)
    # ax.plot(v5x_x, v5x_map_50, lw=1.5)
    # ax.plot(v5x_x, v5x_map_595, ls=':', lw=1.5)
    # ax.plot(v6s_x, v6x_map_50, lw=1.5)
    # ax.plot(v6s_x, v6x_map_595, ls=':', lw=1.5)
    # ax.plot(v7x_x, v7x_map_50, lw=1.5)
    # ax.plot(v7x_x, v7x_map_595, ls=':', lw=1.5)

    # ############################
    ax.plot(v3s_x, v3s_map_50, lw=1.5)
    ax.plot(v3s_x, v3s_map_595, ls=':', lw=1.5)
    ax.plot(v3x_x, v3x_map_50, lw=1.5)
    ax.plot(v3x_x, v3x_map_595, ls=':', lw=1.5)

    ax.plot(v4s_x, v4s_map_50, lw=1.5)
    ax.plot(v4s_x, v4s_map_595, ls=':', lw=1.5)
    ax.plot(v4x_x, v4x_map_50, lw=1.5)
    ax.plot(v4x_x, v4x_map_595, ls=':', lw=1.5)
  
    ax.plot(v5s_x, v5s_map_50, lw=1.5)
    ax.plot(v5s_x, v5s_map_50, ls=':', lw=1.5)
    ax.plot(v5x_x, v5x_map_50, lw=1.5)
    ax.plot(v5x_x, v5x_map_595, ls=':', lw=1.5)

    ax.plot(v6s_x, v6s_map_50, lw=1.5)
    ax.plot(v6s_x, v6s_map_595, ls=':', lw=1.5)
    ax.plot(v6s_x, v6x_map_50, lw=1.5)
    ax.plot(v6s_x, v6x_map_595, ls=':', lw=1.5)
  
    ax.plot(v7s_x, v7s_map_50, lw=1.5)
    ax.plot(v7s_x, v7s_map_595, ls=':', lw=1.5)
    ax.plot(v7x_x, v7x_map_50, lw=1.5)
    ax.plot(v7x_x, v7x_map_595, ls=':', lw=1.5)

    # 去除边
    # ax.spines['top'].set_visible(False)

    # ax=plt.gca()  #gca:get current axis得到当前轴
    # #设置图片的右边框和上边框为不显示
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')


    # plt.xlabel('epoch')
    # plt.xlabel('map')
    # plt.title("不同模型的map曲线图")
    fig.subplots_adjust(wspace=1,hspace=1)


    ax.legend(name_label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('map')
    ax.set_title('不同模型的map曲线图')
    fig.tight_layout()
    ax.set_xlim(0, 400)

    plt.savefig('./data/yolomap.png', dpiconda=300)


    plt.show()
    

# txt_path = r'data/v5s.txt'
    # name_label = ['YOLOv5s_mAP0.5', 'YOLOv5s_mAP.5:.95', 'YOLOv5x_mAP0.5', 'YOLOv5x_mAP.5:.95', 'YOLOv6x_mAP0.5', 'YOLOv6x_mAP.5:.95','YOLOv6s_mAP0.5', 'YOLOv6s_mAP.5:.95']
    v3s_lis = ['YOLOv3s_mAP0.5', 'YOLOv3s_mAP.5:.95']
    v4s_lis = ['YOLOv4s_mAP0.5', 'YOLOv4s_mAP.5:.95']
    v5s_lis = ['YOLOv5s_mAP0.5', 'YOLOv5s_mAP.5:.95']
    v6s_lis = ['YOLOv6s_mAP0.5', 'YOLOv6s_mAP.5:.95']
    v7s_lis = ['YOLOv7s_mAP0.5', 'YOLOv7s_mAP.5:.95']

    v3x_lis = ['YOLOv3x_mAP0.5', 'YOLOv3x_mAP.5:.95']
    v4x_lis = ['YOLOv4x_mAP0.5', 'YOLOv4x_mAP.5:.95']
    v5x_lis = ['YOLOv5x_mAP0.5', 'YOLOv5x_mAP.5:.95']
    v6x_lis = ['YOLOv6x_mAP0.5', 'YOLOv6x_mAP.5:.95']
    v7x_lis = ['YOLOv7x_mAP0.5', 'YOLOv7x_mAP.5:.95']
    # txt_read(txt_path)

    # map_50, map_595 = txt_read(txt_path)
    v3s_map_50, v3s_map_595 = txt_read(r'data/v3x.txt')
    v3x_map_50, v3x_map_595 = txt_read(r'data/v3x.txt')
    v4s_map_50, v4s_map_595 = txt_read(r'data/v4s.txt')
    v4x_map_50, v4x_map_595 = txt_read(r'data/v4x.txt')
    v5s_map_50, v5s_map_595 = txt_read(r'data/v5x.txt')
    v5x_map_50, v5x_map_595 = txt_read(r'data/v5x.txt')
    v6s_map_50, v6s_map_595 = txt_read(r'data/v6x.txt')
    v6x_map_50, v6x_map_595 = txt_read(r'data/v6s.txt')
    v7s_map_50, v7s_map_595 = txt_read(r'data/v7s.txt')
    v7x_map_50, v7x_map_595 = txt_read(r'data/v7x.txt')

    # print(map_50)
    # x = [i for i  in range(len(v5s_map_50))]
    v3s_x = [i for i  in range(len(v3s_map_50))]
    v4s_x = [i for i  in range(len(v4s_map_50))]
    v5s_x = [i for i  in range(len(v5s_map_50))]
    v6s_x = [i for i  in range(len(v6s_map_50))]
    v7s_x = [i for i  in range(len(v7s_map_50))]
    v3x_x = [i for i  in range(len(v3x_map_50))]
    v4x_x = [i for i  in range(len(v4x_map_50))]
    v5x_x = [i for i  in range(len(v5x_map_50))]
    v6x_x = [i for i  in range(len(v6x_map_50))]
    v7x_x = [i for i  in range(len(v7x_map_50))]
    # x = [i for i  in range(len(map_50)) if i % 2 == 0]
    print('v3s -> ',len(v3s_x))
    print('v3x -> ', len(v3x_x))
    print('v4s -> ',len(v4s_x))
    print('v4x -> ', len(v4x_x))
    print('v5s -> ',len(v5s_x))
    print('v5x -> ', len(v5x_x))
    print('v6x -> ', len(v6x_x))    
    print('v6s -> ',len(v6s_x))
    print('v7x -> ', len(v7x_x))
    print('v7s -> ',len(v7s_x))

    v5s5 = v5s_map_50
    v5s9 = v5s_map_595
    # print(v5s5)
    fig, ax = plt.subplots()

 
    # plt.plot(x, v5s5)
    # plt.plot(x, v5s9)

    # name_label = v3s_lis + v4s_lis + v5s_lis + v6s_lis + v7s_lis
    # ax.plot(v3s_x, v3s_map_50, lw=1.5)
    # ax.plot(v3s_x, v3s_map_595, ls=':', lw=1.5)
    # ax.plot(v4s_x, v4s_map_50, lw=1.5)
    # ax.plot(v4s_x, v4s_map_595, ls=':', lw=1.5)
    # ax.plot(v5s_x, v5s_map_50, lw=1.5)
    # ax.plot(v5s_x, v5s_map_50, ls=':', lw=1.5)
    # ax.plot(v6s_x, v6s_map_50, lw=1.5)
    # ax.plot(v6s_x, v6s_map_595, ls=':', lw=1.5)
    # ax.plot(v7s_x, v7s_map_50, lw=1.5)
    # ax.plot(v7s_x, v7s_map_595, ls=':', lw=1.5)

    name_label = v3x_lis + v4x_lis + v5x_lis + v6x_lis + v7x_lis
    ax.plot(v3x_x, v3x_map_50, lw=1.5)
    ax.plot(v3x_x, v3x_map_595, ls=':', lw=1.5)
    ax.plot(v4x_x, v4x_map_50, lw=1.5)
    ax.plot(v4x_x, v4x_map_595, ls=':', lw=1.5)
    ax.plot(v5x_x, v5x_map_50, lw=1.5)
    ax.plot(v5x_x, v5x_map_595, ls=':', lw=1.5)
    ax.plot(v6x_x, v6x_map_50, lw=1.5)
    ax.plot(v6x_x, v6x_map_595, ls=':', lw=1.5)
    ax.plot(v7x_x, v7x_map_50, lw=1.5)
    ax.plot(v7x_x, v7x_map_595, ls=':', lw=1.5)

    # ############################
    # ax.plot(v3s_x, v3s_map_50, lw=1.5)
    # ax.plot(v3s_x, v3s_map_595, ls=':', lw=1.5)
    # ax.plot(v3x_x, v3x_map_50, lw=1.5)
    # ax.plot(v3x_x, v3x_map_595, ls=':', lw=1.5)

    # ax.plot(v4s_x, v4s_map_50, lw=1.5)
    # ax.plot(v4s_x, v4s_map_595, ls=':', lw=1.5)
    # ax.plot(v4x_x, v4x_map_50, lw=1.5)
    # ax.plot(v4x_x, v4x_map_595, ls=':', lw=1.5)
  
    # ax.plot(v5s_x, v5s_map_50, lw=1.5)
    # ax.plot(v5s_x, v5s_map_50, ls=':', lw=1.5)
    # ax.plot(v5x_x, v5x_map_50, lw=1.5)
    # ax.plot(v5x_x, v5x_map_595, ls=':', lw=1.5)

    # ax.plot(v6s_x, v6s_map_50, lw=1.5)
    # ax.plot(v6s_x, v6s_map_595, ls=':', lw=1.5)
    # ax.plot(v6s_x, v6x_map_50, lw=1.5)
    # ax.plot(v6s_x, v6x_map_595, ls=':', lw=1.5)
  
    # ax.plot(v7s_x, v7s_map_50, lw=1.5)
    # ax.plot(v7s_x, v7s_map_595, ls=':', lw=1.5)
    # ax.plot(v7x_x, v7x_map_50, lw=1.5)
    # ax.plot(v7x_x, v7x_map_595, ls=':', lw=1.5)

    # 去除边
    # ax.spines['top'].set_visible(False)

    # ax=plt.gca()  #gca:get current axis得到当前轴
    # #设置图片的右边框和上边框为不显示
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')


    # plt.xlabel('epoch')
    # plt.xlabel('map')
    # plt.title("不同模型的map曲线图")
    # fig.subplots_adjust(wspace=1,hspace=1)


    # 设置轴的刻度间隔
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))

    # ax.grid()
    ax.legend(name_label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('map')
    ax.set_title('YOLO_x模型的map曲线图')
    ax.set_xlim(0, 300)
    fig.tight_layout()
    plt.savefig('./data/yolomap_x.png', dpi=300, bbox_inches='tight')


    plt.show()
    

def v5_map():
# txt_path = r'data/v5s.txt'
    # name_label = ['YOLOv5s_mAP0.5', 'YOLOv5s_mAP.5:.95', 'YOLOv5x_mAP0.5', 'YOLOv5x_mAP.5:.95', 'YOLOv6x_mAP0.5', 'YOLOv6x_mAP.5:.95','YOLOv6s_mAP0.5', 'YOLOv6s_mAP.5:.95']
    v3s_lis = ['YOLOv3s_mAP0.5', 'YOLOv3s_mAP.5:.95']
    v4s_lis = ['YOLOv4s_mAP0.5', 'YOLOv4s_mAP.5:.95']
    v5s_lis = ['YOLOv5s_mAP0.5', 'YOLOv5s_mAP.5:.95']
    v6s_lis = ['YOLOv6s_mAP0.5', 'YOLOv6s_mAP.5:.95']
    v7s_lis = ['YOLOv7s_mAP0.5', 'YOLOv7s_mAP.5:.95']

    v3x_lis = ['YOLOv3x_mAP0.5', 'YOLOv3x_mAP.5:.95']
    v4x_lis = ['YOLOv4x_mAP0.5', 'YOLOv4x_mAP.5:.95']
    v5x_lis = ['YOLOv5x_mAP0.5', 'YOLOv5x_mAP.5:.95']
    v6x_lis = ['YOLOv6x_mAP0.5', 'YOLOv6x_mAP.5:.95']
    v7x_lis = ['YOLOv7x_mAP0.5', 'YOLOv7x_mAP.5:.95']
    # txt_read(txt_path)

    # map_50, map_595 = txt_read(txt_path)
    v3s_map_50, v3s_map_595 = txt_read(r'data/v3x.txt')
    v3x_map_50, v3x_map_595 = txt_read(r'data/v3x.txt')
    v4s_map_50, v4s_map_595 = txt_read(r'data/v4s.txt')
    v4x_map_50, v4x_map_595 = txt_read(r'data/v4x.txt')
    v5s_map_50, v5s_map_595 = txt_read(r'data/v5s.txt')
    v5x_map_50, v5x_map_595 = txt_read(r'data/v5x.txt')
    v6s_map_50, v6s_map_595 = txt_read(r'data/v6x.txt')
    v6x_map_50, v6x_map_595 = txt_read(r'data/v6s.txt')
    v7s_map_50, v7s_map_595 = txt_read(r'data/v7s.txt')
    v7x_map_50, v7x_map_595 = txt_read(r'data/v7x.txt')

    # print(map_50)
    # x = [i for i  in range(len(v5s_map_50))]
    v3s_x = [i for i  in range(len(v3s_map_50))]
    v4s_x = [i for i  in range(len(v4s_map_50))]
    v5s_x = [i for i  in range(len(v5s_map_50))]
    v6s_x = [i for i  in range(len(v6s_map_50))]
    v7s_x = [i for i  in range(len(v7s_map_50))]
    v3x_x = [i for i  in range(len(v3x_map_50))]
    v4x_x = [i for i  in range(len(v4x_map_50))]
    v5x_x = [i for i  in range(len(v5x_map_50))]
    v6x_x = [i for i  in range(len(v6x_map_50))]
    v7x_x = [i for i  in range(len(v7x_map_50))]
    # x = [i for i  in range(len(map_50)) if i % 2 == 0]
    print('v3s -> ',len(v3s_x))
    print('v3x -> ', len(v3x_x))
    print('v4s -> ',len(v4s_x))
    print('v4x -> ', len(v4x_x))
    print('v5s -> ',len(v5s_x))
    print('v5x -> ', len(v5x_x))
    print('v6x -> ', len(v6x_x))    
    print('v6s -> ',len(v6s_x))
    print('v7x -> ', len(v7x_x))
    print('v7s -> ',len(v7s_x))

    v5s5 = v5s_map_50
    v5s9 = v5s_map_595
    # print(v5s5)
    fig, ax = plt.subplots()

    name_label = v5s_lis + v5x_lis 

    # ############################
    # ax.plot(v3s_x, v3s_map_50, lw=1.5)
    # ax.plot(v3s_x, v3s_map_595, ls=':', lw=1.5)
    # ax.plot(v3x_x, v3x_map_50, lw=1.5)
    # ax.plot(v3x_x, v3x_map_595, ls=':', lw=1.5)

    # ax.plot(v4s_x, v4s_map_50, lw=1.5)
    # ax.plot(v4s_x, v4s_map_595, ls=':', lw=1.5)
    # ax.plot(v4x_x, v4x_map_50, lw=1.5)
    # ax.plot(v4x_x, v4x_map_595, ls=':', lw=1.5)
  
    ax.plot(v5s_x, v5s_map_50, lw=1.5)
    ax.plot(v5s_x, v5s_map_595, ls=':', lw=1.5)
    ax.plot(v5x_x, v5x_map_50, lw=1.5)
    ax.plot(v5x_x, v5x_map_595, ls=':', lw=1.5)

    # ax.plot(v6s_x, v6s_map_50, lw=1.5)
    # ax.plot(v6s_x, v6s_map_595, ls=':', lw=1.5)
    # ax.plot(v6s_x, v6x_map_50, lw=1.5)
    # ax.plot(v6s_x, v6x_map_595, ls=':', lw=1.5)
  
    # ax.plot(v7s_x, v7s_map_50, lw=1.5)
    # ax.plot(v7s_x, v7s_map_595, ls=':', lw=1.5)
    # ax.plot(v7x_x, v7x_map_50, lw=1.5)
    # ax.plot(v7x_x, v7x_map_595, ls=':', lw=1.5)

    # 去除边
    # ax.spines['top'].set_visible(False)

    # 设置轴的刻度间隔
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))

    # ax.grid()
    ax.legend(name_label, loc="lower right")
    ax.set_xlabel('epoch')
    ax.set_ylabel('map')
    ax.set_title('YOLO v5模型的map曲线图')
    ax.set_xlim(0, 300)
    fig.tight_layout()
    plt.savefig('./data/yolomap_v5.png', dpi=300, bbox_inches='tight')


    plt.show()
    

def su_map():

    # txt_path = r'data/v5s.txt'
    # name_label = ['YOLOv5s_mAP0.5', 'YOLOv5s_mAP.5:.95', 'YOLOv5x_mAP0.5', 'YOLOv5x_mAP.5:.95', 'YOLOv6x_mAP0.5', 'YOLOv6x_mAP.5:.95','YOLOv6s_mAP0.5', 'YOLOv6s_mAP.5:.95']
    v3s_lis = ['YOLOv3s_mAP0.5', 'YOLOv3s_mAP.5:.95']
    v4s_lis = ['YOLOv4s_mAP0.5', 'YOLOv4s_mAP.5:.95']
    v5s_lis = ['YOLOv5s_mAP0.5', 'YOLOv5s_mAP.5:.95']
    v6s_lis = ['YOLOv6s_mAP0.5', 'YOLOv6s_mAP.5:.95']
    v7s_lis = ['YOLOv7s_mAP0.5', 'YOLOv7s_mAP.5:.95']

    v3x_lis = ['YOLOv3x_mAP0.5', 'YOLOv3x_mAP.5:.95']
    v4x_lis = ['YOLOv4x_mAP0.5', 'YOLOv4x_mAP.5:.95']
    v5x_lis = ['YOLOv5x_mAP0.5', 'YOLOv5x_mAP.5:.95']
    v6x_lis = ['YOLOv6x_mAP0.5', 'YOLOv6x_mAP.5:.95']
    v7x_lis = ['YOLOv7x_mAP0.5', 'YOLOv7x_mAP.5:.95']
    # txt_read(txt_path)

    # map_50, map_595 = txt_read(txt_path)
    v3s_map_50, v3s_map_595 = txt_read(r'data/v3x.txt')
    v3x_map_50, v3x_map_595 = txt_read(r'data/v3x.txt')
    v4s_map_50, v4s_map_595 = txt_read(r'data/v4s.txt')
    v4x_map_50, v4x_map_595 = txt_read(r'data/v4x.txt')
    v5s_map_50, v5s_map_595 = txt_read(r'data/v5s.txt')
    v5x_map_50, v5x_map_595 = txt_read(r'data/v5x.txt')
    v6s_map_50, v6s_map_595 = txt_read(r'data/v6x.txt')
    v6x_map_50, v6x_map_595 = txt_read(r'data/v6s.txt')
    v7s_map_50, v7s_map_595 = txt_read(r'data/v7s.txt')
    v7x_map_50, v7x_map_595 = txt_read(r'data/v7x.txt')

    # print(map_50)
    # x = [i for i  in range(len(v5s_map_50))]
    v3s_x = [i for i  in range(len(v3s_map_50))]
    v4s_x = [i for i  in range(len(v4s_map_50))]
    v5s_x = [i for i  in range(len(v5s_map_50))]
    v6s_x = [i for i  in range(len(v6s_map_50))]
    v7s_x = [i for i  in range(len(v7s_map_50))]
    v3x_x = [i for i  in range(len(v3x_map_50))]
    v4x_x = [i for i  in range(len(v4x_map_50))]
    v5x_x = [i for i  in range(len(v5x_map_50))]
    v6x_x = [i for i  in range(len(v6x_map_50))]
    v7x_x = [i for i  in range(len(v7x_map_50))]
    # x = [i for i  in range(len(map_50)) if i % 2 == 0]
    print('v3s -> ',len(v3s_x))
    print('v3x -> ', len(v3x_x))
    print('v4s -> ',len(v4s_x))
    print('v4x -> ', len(v4x_x))
    print('v5s -> ',len(v5s_x))
    print('v5x -> ', len(v5x_x))
    print('v6x -> ', len(v6x_x))    
    print('v6s -> ',len(v6s_x))
    print('v7x -> ', len(v7x_x))
    print('v7s -> ',len(v7s_x))

    v5s5 = v5s_map_50
    v5s9 = v5s_map_595
    # print(v5s5)
    fig, ax = plt.subplots()

 
    # plt.plot(x, v5s5)
    # plt.plot(x, v5s9)

    # name_label = v3s_lis + v4s_lis + v5s_lis + v6s_lis + v7s_lis
    # ax.plot(v3s_x, v3s_map_50, lw=1.5)
    # ax.plot(v3s_x, v3s_map_595, ls=':', lw=1.5)
    # ax.plot(v4s_x, v4s_map_50, lw=1.5)
    # ax.plot(v4s_x, v4s_map_595, ls=':', lw=1.5)
    # ax.plot(v5s_x, v5s_map_50, lw=1.5)
    # ax.plot(v5s_x, v5s_map_50, ls=':', lw=1.5)
    # ax.plot(v6s_x, v6s_map_50, lw=1.5)
    # ax.plot(v6s_x, v6s_map_595, ls=':', lw=1.5)
    # ax.plot(v7s_x, v7s_map_50, lw=1.5)
    # ax.plot(v7s_x, v7s_map_595, ls=':', lw=1.5)

    name_label = v3x_lis + v4x_lis + v5x_lis + v6x_lis + v7x_lis
    ax.plot(v3x_x, v3x_map_50, lw=1.5)
    ax.plot(v3x_x, v3x_map_595, ls=':', lw=1.5)
    ax.plot(v4x_x, v4x_map_50, lw=1.5)
    ax.plot(v4x_x, v4x_map_595, ls=':', lw=1.5)
    ax.plot(v5x_x, v5x_map_50, lw=1.5)
    ax.plot(v5x_x, v5x_map_595, ls=':', lw=1.5)
    ax.plot(v6x_x, v6x_map_50, lw=1.5)
    ax.plot(v6x_x, v6x_map_595, ls=':', lw=1.5)
    ax.plot(v7x_x, v7x_map_50, lw=1.5)
    ax.plot(v7x_x, v7x_map_595, ls=':', lw=1.5)

    # ############################
    # ax.plot(v3s_x, v3s_map_50, lw=1.5)
    # ax.plot(v3s_x, v3s_map_595, ls=':', lw=1.5)
    # ax.plot(v3x_x, v3x_map_50, lw=1.5)
    # ax.plot(v3x_x, v3x_map_595, ls=':', lw=1.5)

    # ax.plot(v4s_x, v4s_map_50, lw=1.5)
    # ax.plot(v4s_x, v4s_map_595, ls=':', lw=1.5)
    # ax.plot(v4x_x, v4x_map_50, lw=1.5)
    # ax.plot(v4x_x, v4x_map_595, ls=':', lw=1.5)
  
    # ax.plot(v5s_x, v5s_map_50, lw=1.5)
    # ax.plot(v5s_x, v5s_map_50, ls=':', lw=1.5)
    # ax.plot(v5x_x, v5x_map_50, lw=1.5)
    # ax.plot(v5x_x, v5x_map_595, ls=':', lw=1.5)

    # ax.plot(v6s_x, v6s_map_50, lw=1.5)
    # ax.plot(v6s_x, v6s_map_595, ls=':', lw=1.5)
    # ax.plot(v6s_x, v6x_map_50, lw=1.5)
    # ax.plot(v6s_x, v6x_map_595, ls=':', lw=1.5)
  
    # ax.plot(v7s_x, v7s_map_50, lw=1.5)
    # ax.plot(v7s_x, v7s_map_595, ls=':', lw=1.5)
    # ax.plot(v7x_x, v7x_map_50, lw=1.5)
    # ax.plot(v7x_x, v7x_map_595, ls=':', lw=1.5)

    # 去除边
    # ax.spines['top'].set_visible(False)

    # ax=plt.gca()  #gca:get current axis得到当前轴
    # #设置图片的右边框和上边框为不显示
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')


    # plt.xlabel('epoch')
    # plt.xlabel('map')
    # plt.title("不同模型的map曲线图")
    # fig.subplots_adjust(wspace=1,hspace=1)


    # 设置轴的刻度间隔
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))

    # ax.grid()
    ax.legend(name_label, frameon=False)
    # plt.legend(frameon=False,loc="upper right",fontsize='small') #分别为图例无边框、图例放在右上角、图例大小

    ax.set_xlabel('epoch')
    ax.set_ylabel('map')
    ax.set_title('YOLO_x模型的map曲线图')
    ax.set_xlim(0, 300)
    fig.tight_layout()
    # plt.savefig('./data/yolomap_s.png', dpi=300, bbox_inches='tight')
    plt.savefig('./data/yolomap_x.png', dpi=300)


    plt.show()
    

def main():
    # all_map()

    # su_map()
    v5_map()






if __name__ == '__main__':

    main()



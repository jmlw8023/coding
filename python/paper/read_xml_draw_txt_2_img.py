# -*- encoding: utf-8 -*-
'''
@File    :   read_xml_draw_txt_2_img.py
@Time    :   2023/01/06 14:05:30
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :
'''

# import packets
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
from PIL import Image, ImageDraw, ImageFont


# 没有文件夹就会生成文件夹
def mk_dir(folder_path):
    print(folder_path)
    folder_path = os.path.abspath(folder_path)
    assert os.path.isdir(os.path.dirname(folder_path)), 'Please input folder dir!!'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print('create {} success!!'.format(folder_path))

# 改变xml文件中信息内容（类别的名称），并绘制相关信息到图片中
def change_xml_file_name(root_path, xml_file, color=[0, 255, 200], img=None, target_suffix='.xml'):
    import xml.etree.ElementTree as ET 
    # print('start')

    img_shuffix = '.jpg'
    if xml_file.endswith(target_suffix):
        file_name = xml_file.split('.')[0]
        img_path = os.path.join(root_path, 'images', file_name + img_shuffix)
        save_img_path = os.path.join(root_path, 'results', file_name + img_shuffix)
        txt_file_path = os.path.join(root_path, 'results/txt', file_name + '.txt')
        mk_dir(os.path.join(root_path, 'results'))
        mk_dir(os.path.join(root_path, 'results/txt'))

    
    if img_path is not None:
        img = cv.imread(img_path)
    
    else:
        assert 'please input image path'
    
    h, w, dim = img.shape

    # 生成随机颜色
    color = [random.randint(0, 255) for _ in range(3)]
    # 划线的粗细大小
    line_thickness = round(0.0004 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    # 文件不是 target_suffix （.xml）结尾的就会报错
    assert xml_file.endswith(target_suffix), f'please input {target_suffix} file!!'
    # xml_name = xml_file.split('/')[-1]
    # print(xml_name)
    # xml_name, _ = os.path.splitext(xml_name)
    # jpg_name = xml_name + '.jpg'

    # xml文件路径
    xml_path = os.path.join(root_path, 'Annotations', xml_file)

    # 存储xml文件的路径
    save_xml_path = os.path.join(os.path.dirname(root_path), 'Annotations' , xml_file)
    # 创建根节点
    tree = ET.parse(xml_path)
    # tree = ET.parse(os.path.abspath(xml_path))
    root = tree.getroot()
    # 查找xml中第一层 key为filename的文本value内容
    # filename = root.find('filename').text
    # print(filename)
    
    # 存储的类别
    cls_name = []
    # 绘制显示中文字体的内容
    conext = ''
    # 保留原类别名称（英文）
    txt_context = ''
    fag = False
    for obj in root.iter('object'):
        name = obj.find('name').text
        cls_name.append(name)

        # 修改标签名称
        # if name == 'medium_truck':
        #     print(xml_file)
        #     obj.find('name').text = 'dump_truck'
        #     fag = True

        # 查询根节点 object 下的次一层 'bndbox'
        box = obj.find('bndbox')
        # bndbox 层之下的信息内容
        xmin = box.find('xmin').text
        ymin = box.find('ymin').text
        xmax = box.find('xmax').text
        ymax = box.find('ymax').text
        # temp_list.append(name, xmin, ymin, xmax, ymax)
        # print(name, xmin, ymin, xmax, ymax)

        # 从string 类型转为int 类型
        top, left, right, bottom = int(xmin), int(ymin), int(xmax), int(ymax)
        # 采用cv的rectangle 进行绘制矩形框
        cv.rectangle(img, (int(top), int(left)), (int(right), int(bottom)), color, thickness=line_thickness, lineType=cv.LINE_AA)    # filled

        # 随机生成指定范围的分值
        score = round(np.random.uniform(0.6, 0.92), 3)
        # txt = name + '  {}'.format(score)

        txt_context += name + '\t {} \n'.format(score) 


        if name == 'person':
            conext += '人  {}\n'.format(score)
            name = '人  {}'.format(score)
        elif name == 'dump_truck':
            conext += '土方车  {}\n'.format(score)
            name = '土方车  {}'.format(score)
        elif name == 'car':
            conext += '小轿车  {}\n'.format(score)
            name = '小轿车  {}'.format(score)
        elif name == 'mixer_truck':
            conext += '搅拌车  {}\n'.format(score)  
            name = '搅拌车  {}'.format(score)          
        elif name == 'motor_tractor':
            conext += '牵引车  {}\n'.format(score)
            name = '牵引车  {}'.format(score)  
        elif name == 'microtruck':
            conext += '小卡车  {}\n'.format(score)  
            name = '小卡车 {}'.format(score)

        txt = name 


        # print(txt)
        # conext += txt + '\n'

        if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
            img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)    # 创建绘制图像   
            fontstype = ImageFont.truetype(font=os.path.join(root_path, 'myfont.ttf'), size=20, encoding="utf-8")
            draw.text((top, left - 25), txt, (0, 255, 0), font=fontstype)  # 绘制文本
            # draw.text((right - 40, bottom - left - 10), name, (0, 255, 0), font=fontstype)  # 绘制文本
            # draw.text((top+25, left-25), name, tuple(color), font=fontstype)  # 绘制文本

        # print(temp_list)
        # print('-' * 15)
        img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    

    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(txt_context)


    num = len(cls_name)
    cls_num = len(set(cls_name))
    # info = '总 {} 类别 分别 {} 个目标 \n{}'.format(cls_num, num, conext)
    info = '--总 {} 类别 分别 {} 个目标--'.format(cls_num, num)
    conext = '类别名称   |  预测可信度\n' + conext
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)    # 创建绘制图像   
        fontstype = ImageFont.truetype(font=os.path.join(root_path, 'myfont.ttf'), size=20, encoding="utf-8")
        # draw.text((w - 260, 25), info, (255, 255, 0), font=fontstype)  # 绘制文本到右上角
        # draw.text((w - 260, 50), conext, (255, 10, 0), font=fontstype)  # 绘制文本到右上角

        draw.text((10, 25), info, (255, 255, 0), font=fontstype)  # 绘制文本到左上角
        draw.text((10, 50), conext, (255, 10, 0), font=fontstype)  # 绘制文本到左上角
        img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

    # plt.savefig(save_img_path, dip=300)
    cv.imwrite(save_img_path, img)
    # cv.imshow('res.png', img)

    
    # 全部xml文件文件进行写入
    # tree.write(save_xml_path)
    # 符合条件的xml 标签进行写入
    # if fag:
    #     tree.write(save_xml_path)
    #     print('write {} success!'.format(xml_file))




def draw_txt_2_img(img, txt_data, box_data):
    top, left, right, bottom = box_data
    if (isinstance(image, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)    # 创建绘制图像
        fontstype = ImageFont.truetype("myfont.ttf", 20, encoding="utf-8")
        # draw.text((top, left-25), txt_data, (0, 255, 0), font=fontstype)  # 绘制文本
        draw.text((top, left-25), txt_data, (0, 255, 0), font=fontstype)  # 绘制文本
        image = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


#绘制文字到单张图片中
def draw_box_2_img(image, box_data, cls, score):  
    top, left, right, bottom = box_data
    if (isinstance(image, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)    # 创建绘制图像
        txt = '{0} {1:.2f}'.format(cls, score)
        fontstype = ImageFont.truetype("myfont.ttf", 20, encoding="utf-8")
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print(f' cl ---------- {cl}')
        # draw.text((top, left-25), txt, (125, 255, 0), font=fontstype)  # 绘制文本在右上角
        draw.text((top, right-25), txt, (125, 255, 0), font=fontstype)  # 绘制文本在右上角
        image = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)




# # 数据文件路径
root_path = r'./data'
# # 测试demo
img = r'./data/images/bus.jpg'
# xml = r'./data/Annotations/bus.xml'

# 通过单张图片进行测试
file_xml = 'zidane.xml'

change_xml_file_name(root_path, file_xml)

# 随机获取RGB（0~255）范围内的元素值
# color = [random.randint(0, 255) for _ in range(3)]
# print(color)

# 整个目录中的xml文件进行
# for xml_file in os.listdir(root_folder):
#     # print(xml_file)
#     change_xml_file_name(root_path, xml_file)



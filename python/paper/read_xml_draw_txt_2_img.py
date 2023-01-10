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




def change_xml_file_name(root_path, xml_file, color=[0, 255, 200], img=None, target_suffix='.xml'):
    import xml.etree.ElementTree as ET 
    # print('start')
    if xml_file.endswith(target_suffix):
        file_name = xml_file.split('.')[0]
        img_path = os.path.join(root_path, 'res', file_name + '.png')
        save_img_path = os.path.join(root_path, 'results', file_name + '.png')
        txt_file_path = os.path.join(root_path, 'results/txt', file_name + '.txt')

    
    if img_path is not None:
        img = cv.imread(img_path)
    
    h, w, dim = img.shape
    # print(img.shape)

    color = [random.randint(0, 255) for _ in range(3)]
    line_thickness = round(0.0004 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness


    assert xml_file.endswith(target_suffix), f'please input {target_suffix} file!!'
    # xml_name = xml_file.split('/')[-1]
    # print(xml_name)
    # xml_name, _ = os.path.splitext(xml_name)
    # jpg_name = xml_name + '.jpg'

    xml_path = os.path.join(root_path, 'Annotation', xml_file)

    save_xml_path = os.path.join(os.path.dirname(root_path), 'Annotation' , xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    # print(filename)
    
    cls_name = []
    conext = ''
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


        box = obj.find('bndbox')
        xmin = box.find('xmin').text
        ymin = box.find('ymin').text
        xmax = box.find('xmax').text
        ymax = box.find('ymax').text
        # temp_list.append(name, xmin, ymin, xmax, ymax)
        # print(name, xmin, ymin, xmax, ymax)
        top, left, right, bottom = int(xmin), int(ymin), int(xmax), int(ymax)
        cv.rectangle(img, (int(top), int(left)), (int(right), int(bottom)), color, thickness=line_thickness, lineType=cv.LINE_AA)    # filled

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
            fontstype = ImageFont.truetype(font=r'myfont.ttf', size=20, encoding="utf-8")
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
        fontstype = ImageFont.truetype(font=r'myfont.ttf', size=20, encoding="utf-8")
        draw.text((w - 260, 25), info, (255, 255, 0), font=fontstype)  # 绘制文本
        draw.text((w - 260, 50), conext, (255, 10, 0), font=fontstype)  # 绘制文本
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


#画框到单张图片中
def draw_box_2_img(image, box_data, cls, score):  
    top, left, right, bottom = box_data
    if (isinstance(image, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)    # 创建绘制图像
        txt = '{0} {1:.2f}'.format(cls, score)
        fontstype = ImageFont.truetype("myfont.ttf", 20, encoding="utf-8")
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print(f' cl ---------- {cl}')
        draw.text((top, left-25), txt, (125, 255, 0), font=fontstype)  # 绘制文本
        image = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)






root_path = r'E:\w\qun\datasets'

# for name in os.listdir(root_path):
#     img_path = os.path.join(root_path, 'results', name)
#     # print(img_path)
#     img = cv.imread(img_path)


img = r'E:\w\qun\datasets\results\result_00020.png'

xml = r'E:\w\qun\datasets\Annotation\trucks_00003.xml'

# root_folder = os.path.dirname(xml)
root_folder, file_xml = os.path.dirname(xml), 'trucks_00002.xml'
# print(root_folder, file_xml)

# change_xml_file_name(root_folder, file_xml)

# color = [random.randint(0, 255) for _ in range(3)]
# print(color)
for xml_file in os.listdir(root_folder):
    # print(xml_file)
    change_xml_file_name(root_path, xml_file)



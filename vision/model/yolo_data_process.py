#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   yolo_data_process.py
@Time    :   2024/04/08 11:04:25
@Author  :   hgh 
@Version :   1.0
@Desc    :    通过Image和Annatation标签，进行随机分割数据集并xml生成txt文件，以及对xml文件夹进行统计标签
'''

# import module
import os
import shutil
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xml.etree.ElementTree as ET
from collections import defaultdict

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False



def check_xml_files_in_dir(directory, shuffix_end='.xml', threshold=0.5):  
    """  
    检查目录中的文件是否大部分以 .xml 结尾。  
    :param directory: 要检查的目录路径  
    :param threshold: 阈值，表示以 .xml 结尾的文件应占文件总数的比例  
    # :return: 如果满足条件，返回 True；否则返回 False  
    :return: 返回需要统计结尾的个数
    """  
    # 初始化计数器  
    xml_count = 0  
    total_count = 0  
      
    # 遍历目录中的文件  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            # 如果文件以 .xml 结尾，则增加计数器  
            if str(file).lower().endswith(shuffix_end):  
                xml_count += 1  
            # 增加总文件数  
            total_count += 1  
      
    # 检查是否满足阈值条件  
    if total_count == 0:  
        # 如果没有文件，则默认不满足条件  
        return False  
    else:  
        # 计算比例并检查是否大于或等于阈值  
        # ratio = xml_count / total_count  
        # return ratio >= threshold  
        return xml_count



def statistical_categories(annatation_folder, is_draw=False, is_show=False):
    
    annatation_folder = os.path.abspath(annatation_folder)
    assert os.path.isdir(annatation_folder), 'annatation is not folder!'
    save_path = os.path.dirname(annatation_folder)
    categories_lis = []
    xml_counter_dict = defaultdict(int) 
    img_counter_dict = defaultdict(int) 
    annatation_lis = os.listdir(annatation_folder)
    for xml_file in annatation_lis:
        xml_path = os.path.join(annatation_folder, xml_file)
        if xml_path.lower().endswith('.xml'):
            
            doc = ET.parse(xml_path)
            root = doc.getroot()
            
            objects = root.findall('object')
            
            temp_img_count = set()
            # 统计xml每个类别个数
            for object in objects:
                name = object.find('name').text
                temp_img_count.add(name)
                if name not in categories_lis:
                    categories_lis.append(name)
                    xml_counter_dict[name]  = 0
                    img_counter_dict[name]  = 0
                
                if name in xml_counter_dict:
                    xml_counter_dict[name] += 1
                
            
            # 统计xml是否包含某个类
            for name in temp_img_count:
                img_counter_dict[name] += 1
                
    print(categories_lis)

    if is_draw:
        # 提取键和值到两个列表中  
        xml_keys = list(xml_counter_dict.keys())  
        xml_values = list(xml_counter_dict.values())  
        img_keys = list(img_counter_dict.keys())  
        img_values = list(img_counter_dict.values())  
                    
        width = 0.2  # 柱子宽度以及两组之间的间距
        fig, ax = plt.subplots()
        
        ax.set_xlabel('类别')
        ax.set_ylabel('数量')
        ax.set_title('数据集图片和标签数量统计(共{}张图像)'.format(len(annatation_lis)))
        
        ax.bar(np.arange(len(xml_keys)) - width, xml_values, width, color='r', label='xml个数', alpha=0.7)
        ax.bar(np.arange(len(img_keys)), img_values, width, color='b', label='图像个数', alpha=0.7)
        
        # 在柱状图上显示数据标签
        for i, v in enumerate(xml_values):
            ax.text(i - width, v + 1, str(v), ha='center', va='bottom')
        for i, v in enumerate(img_values):
            ax.text(i, v + 1, str(v), ha='center', va='bottom')
            
        ax.legend() # 增加图例
        ax.set_xticks(np.arange(len(categories_lis)))
        ax.set_xticklabels(categories_lis)
        
        fig.tight_layout()
        plt.savefig('{}/statistical_annatation.png'.format(save_path), dpi=300)
        if is_show:
            plt.show()
        
    return categories_lis, xml_counter_dict, img_counter_dict
        

def convert(size, box):
    if size[0] != 0 and size[1] != 0:
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)
    
# 使用xml转为txt文件格式
def convert_annotation(xml_path, save_label_path, classes):
    
    assert os.path.isfile(xml_path), 'xml file is error!'
    in_file = open(xml_path)
    
    if os.path.isdir(save_label_path):
        xml_path, shuffix = os.path.splitext(os.path.basename(xml_path))
        save_label_path = os.path.join(save_label_path, xml_path + '.txt')
        
    out_file = open(save_label_path, 'w', encoding='utf-8')

    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:     # 困难的也需要
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()

def split_data_with_images(img_folder, annatation_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, create_txt=False, is_shuffle=False, img_shuffix=('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
    
    # 确保比例之和为1  
    assert train_ratio + val_ratio + test_ratio == 1  
    
    assert os.path.isdir(img_folder), 'image_folder is not dir'
    assert os.path.isdir(annatation_folder), 'annatation_folder is not dir'
    
    # 统计标签类别，选择是否绘制类别数量图
    classes, _, _ = statistical_categories(annatation_folder, is_draw=False, is_show=False)
    # print(classes)
    assert len(classes) >= 0 , 'annatation not exist class'
    
    seed = 22
    epsilon = 1e-9  # 判断浮点数阈值
    
    if is_shuffle:
        seed = np.random.randint(1, 100)
    
    random.seed(seed) 
    
    img_folder = os.path.abspath(img_folder)
    save_folder = os.path.dirname(img_folder)
    
    
    with open(os.path.join(save_folder, 'classes.txt'), 'w', encoding='utf-8') as f:
        for name in classes:
            f.write(str(name) + '\n')
    
    annatation_shuffix = '.xml'
    
    anna_nums = check_xml_files_in_dir(annatation_folder, annatation_shuffix)
    
    assert (len(os.listdir(img_folder)) <= anna_nums), '图像未标注完，标签数量有误！'
        

    # 获取所有图像文件  
    image_files = []  
    for root, dirs, files in os.walk(img_folder):  
        for file in files:  
            if file.lower().endswith(img_shuffix):  # 根据需要添加其他格式  
                image_files.append(os.path.join(root, file))  
                
    
     # 打乱文件顺序  
    random.shuffle(image_files)  
  
    # 计算每个集合的文件数量  
    total_files = len(image_files)  
    # train_files_num = int(train_ratio * total_files)  
    # val_files_num = int(val_ratio * total_files)  
    train_files_num = round(train_ratio * total_files)  
    val_files_num = round(val_ratio * total_files)  
    test_files_num = total_files - train_files_num - val_files_num  
  
    # 划分数据集  
    train_files = image_files[:train_files_num]  
    val_files = image_files[train_files_num : train_files_num + val_files_num]  
    test_files = image_files[train_files_num + val_files_num:]  
    
    
    
  
    p_img_train = '{}/train/images/'.format(save_folder)
    p_img_val = '{}/val/images/'.format(save_folder)
    p_an_train = '{}/train/Annotation/'.format(save_folder)
    p_an_val = '{}/val/Annotation/'.format(save_folder)
    
    if create_txt:
        p_txt_train = '{}/train/labels/'.format(save_folder)
        p_txt_val = '{}/val/labels/'.format(save_folder)
        # 创建输出目录（如果不存在）  
        os.makedirs(p_txt_train, exist_ok=True)  
        os.makedirs(p_txt_val, exist_ok=True)  
        
        
    # 创建输出目录（如果不存在）  
    os.makedirs(p_img_train, exist_ok=True)  
    os.makedirs(p_img_val, exist_ok=True)  
    os.makedirs(p_an_train, exist_ok=True)  
    os.makedirs(p_an_val, exist_ok=True)  
    
    # 移动文件到对应的目录  
    for file in train_files:  
        filename, shuffix = os.path.splitext(os.path.basename(file))
        xml_path = os.path.join(annatation_folder, filename + annatation_shuffix)
        assert os.path.isfile(xml_path), '{} not exist xml file'.format(xml_path)
        shutil.copy(file, p_img_train)  
        shutil.copy(xml_path, p_an_train)
        
        if create_txt:
            convert_annotation(xml_path, p_txt_train, classes)
            
        
        
    for file in val_files:  
        filename, shuffix = os.path.splitext(os.path.basename(file))
        xml_path = os.path.join(annatation_folder, filename + annatation_shuffix)
        assert os.path.isfile(xml_path), '{} not exist xml file'.format(xml_path)
        shutil.copy(file, p_img_val)  
        shutil.copy(xml_path, p_an_val)
        
        if create_txt:
            convert_annotation(xml_path, p_txt_val, classes)
            

    # 只有训练和验证集，没有测试集
    if abs(test_ratio) < epsilon:
        p_img_test = '{}/test/images/'.format(save_folder)
        p_an_test = '{}/test/Annotation/'.format(save_folder)
        os.makedirs(p_an_test, exist_ok=True)  
        os.makedirs(p_img_test, exist_ok=True)  
        if create_txt:
            p_txt_test = '{}/test/images/'.format(save_folder)
            os.makedirs(p_txt_test, exist_ok=True)  
            
        for file in test_files:  
            filename, shuffix = os.path.splitext(os.path.basename(file))
            xml_path = os.path.join(annatation_folder, filename + annatation_shuffix)
            assert os.path.isfile(xml_path), '{} not exist xml file'.format(xml_path)
        
            shutil.copy(file, p_img_test)  
            shutil.copy(xml_path, p_an_test)
            
            if create_txt:
                convert_annotation(xml_path, p_txt_val, classes)
                
    print('finish!')





if __name__ == '__main__':
    

    ######################################################################################
    add_img_folder        = r'D:/source/code/datasets/b113/2403/add/images'
    add_annatation_folder = r'D:/source/code/datasets/b113/2403/add/Annotations'
    img_folder        = r'D:/source/code/datasets/b113/2403/yolo/images'
    annatation_folder = r'D:/source/code/datasets/b113/2403/yolo/Annotations'
    # img_folder        = r'D:/source/code/datasets/b113/images'
    # annatation_folder = r'D:/source/code/datasets/b113/Annotations'
    
    train_ratio = 0.9
    val_ratio = 0.1
    test_ratio = 0
    
    split_data_with_images(add_img_folder, add_annatation_folder, train_ratio, val_ratio, test_ratio, create_txt=True)
    # split_data_with_images(img_folder, annatation_folder, train_ratio, val_ratio, test_ratio, create_txt=True)
    
    #####################################################################################
    pass





def test_mul_matplot():

    import matplotlib.pyplot as plt
    import numpy as np

    # 生成示例数据
    N = 5
    men_means = (20, 35, 30, 35, 27)
    women_means = (25, 32, 34, 20, 25)
    children_means = (32, 20, 33, 24, 29)

    ind = np.arange(N)  # 柱状图的x轴位置

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 创建包含三个子图的画布

    # 绘制第一个子图
    axs[0].bar(ind, men_means, color='b', label='Men', alpha=0.7)
    axs[0].set_title('Men')
    for i, v in enumerate(men_means):
        axs[0].text(i, v + 1, str(v), ha='center', va='bottom')

    # 绘制第二个子图
    axs[1].bar(ind, women_means, color='r', label='Women', alpha=0.7)
    axs[1].set_title('Women')
    for i, v in enumerate(women_means):
        axs[1].text(i, v + 1, str(v), ha='center', va='bottom')

    # 绘制第三个子图
    axs[2].bar(ind, children_means, color='g', label='Children', alpha=0.7)
    axs[2].set_title('Children')
    for i, v in enumerate(children_means):
        axs[2].text(i, v + 1, str(v), ha='center', va='bottom')

    # 显示图例
    for ax in axs:
        ax.legend()

    plt.show()



def test_one():
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成示例数据
    N = 5
    men_means = (20, 35, 30, 35, 27)
    women_means = (25, 32, 34, 20, 25)
    children_means = (32, 20, 33, 24, 29)

    ind = np.arange(N)  # 柱状图的x轴位置
    width = 0.2  # 柱状图的宽度

    fig, ax = plt.subplots()

    # 绘制并排的柱状图
    ax.bar(ind - width, men_means, width, color='b', label='Men', alpha=0.7)
    ax.bar(ind, women_means, width, color='r', label='Women', alpha=0.7)
    ax.bar(ind + width, children_means, width, color='g', label='Children', alpha=0.7)

    # 在柱状图上显示数据标签
    for i, v in enumerate(men_means):
        ax.text(i - width, v + 1, str(v), ha='center', va='bottom')
    for i, v in enumerate(women_means):
        ax.text(i, v + 1, str(v), ha='center', va='bottom')
    for i, v in enumerate(children_means):
        ax.text(i + width, v + 1, str(v), ha='center', va='bottom')

    # 设置x轴标签和标题
    ax.set_xticks(ind)
    ax.set_xticklabels(('1', '2', '3', '4', '5'))
    ax.set_xlabel('Groups')
    ax.set_title('Scores by group and gender')

    # 显示图例
    ax.legend()

    plt.show()




















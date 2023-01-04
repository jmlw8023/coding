# -*- encoding: utf-8 -*-
'''
@File    :   dataset_op.py
@Time    :   2022/11/03 09:48:51
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xml.etree.ElementTree as ET
from collections import defaultdict

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False




class OperationDatasets():
    def __init__(self, root_path='./datasets', classes_path='classes.txt') -> None:
        # 指定数据集路径

        self.root_path = root_path
        self.xml_path = os.path.join(self.root_path, 'Annotations')
        # 类别文件路径
        self.classes_path = classes_path
        self.classes, _ = self.get_classes(self.classes_path)

        # self.photo_nums  = np.zeros(len(files_lis))
        self.nums   = np.zeros(len(self.classes))

    #---------------------------------------------------#
    #   获得类别种类
    #---------------------------------------------------#
    def get_classes(self, classes_path):
        with open(self.classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)

    #-------------------------------------------------------#
    #   统计目标数量
    #-------------------------------------------------------#
    def nums_annotation(self, xml_path, file_id, cls_set):
        in_file = open(os.path.join(xml_path, '%s.xml'%(file_id)), encoding='utf-8')
        tree=ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            # difficult = 0 
            # if obj.find('difficult')!=None:
            #     difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes:
                continue
            cls_id = self.classes.index(cls)
            self.cls_dict[cls].append(file_id)
            # if cls_id == 0:
            #     preening.append(file_id)
            cls_set[cls].add(file_id)
            self.nums[self.classes.index(cls)] = self.nums[self.classes.index(cls)] + 1

            xmlbox = obj.find('bndbox')
            # b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
            # list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    
    #-------------------------------------------------------#
    #   进行数据选择后分割数据集
    #-------------------------------------------------------#
    def choose_data(self, data, t_percent=0.80, v_percent=0.90):
        data = list(data)
        total_file = len(data)
        train_data  = round(t_percent * total_file)
        test_data =  round((total_file - train_data) / 2)
        val_data = test_data
        np.random.shuffle(data)
        print()
        train, val, test = data[:train_data], data[train_data:(train_data +val_data)], data[(train_data +val_data):]
        return set(train), set(val), set(test)

    def dataset_statistics(self):
        files_lis = []
        for name in os.listdir(self.xml_path):
            files_lis.append(name[:-4])

        trainval_percent    = 0.9
        train_percent       = 0.75

        num     = len(files_lis)  
        lis    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent) 

        photo_nums  = np.zeros(len(files_lis))
        # self.nums   = np.zeros(len(self.classes))

        cls_set = defaultdict(set)
        cls_dict = defaultdict(list)
        # self.cls_set = cls_set
        self.cls_dict = cls_dict

        for key in self.classes:
            cls_dict[key]
            cls_set[key]

        # [cls_set[key] for key in classes]
        # [cls_dict[key].setdefault(0) for key in classes]

        for xml in files_lis:
            # xml = xml[:-4]
            self.nums_annotation(self.xml_path, xml, cls_set)
        self.cls_set = cls_set
        # print(len(cls_set))
        # print()

        # 打印结果
        with open('./results/dataset_statistics.txt', 'w', encoding='utf-8') as f:
            for name, num in zip(self.classes, self.nums):
                count =  len(cls_set[name])
                f.write(name + ' --> '+  str(count) + '张图片' + '\n')
                f.write(name + ' --> ' + str(num) + '个xml' + '  vs  {} 个 \n'.format(len(cls_dict[name])))
                print(name , ' --> ', count, '张图片')
                print(name , ' --> ', int(num), '个xml', '  {}'.format(len(cls_dict[name])))

    def draw_datasets(self):
        x = self.classes
        y_photo_nums = [ len(self.cls_set[name]) for name in  x]
        y_xml_nums = [ int(num) for num in  self.nums]
    
        ges = np.arange(len(x))
        width = 0.35
        fig, ax = plt.subplots()

        ax.set_xlabel('类别')
        ax.set_ylabel('图片数量')
        ax.set_title('肉鸽行为数据集图片数量和标签数量统计')
        x.insert(0, '')
        ax.set_xticklabels(x)
        xml_ret = ax.bar(ges - width / 2, y_xml_nums, width=width, color='r', hatch=r'\\', label='xml个数')
        photo_ret = ax.bar(ges + width / 2 + 0.01, y_photo_nums, width=width, color='c', hatch='/', label='图片数量')
        ax.legend()

        for num, y_xml_num, y_photo_num in zip(ges, y_xml_nums, y_photo_nums):
            ax.text(num - width / 2, y_xml_num + 15, y_xml_num, ha='center', fontsize=12)
            ax.text(num + width / 2, y_photo_num + 15, y_photo_num, ha='center', fontsize=12)


        # plt.xlabel('类别')
        # plt.ylabel('图片数量')

        # # plt.bar(x, y_photo_nums, color='c', hatch='/')
        # # plt.savefig('images_nums.png', dip=300)

        # plt.bar(x, y_xml_nums, color='r', hatch=r'\\')
        plt.savefig('./results/pigeon_xml_and_photo_nums.png', dip=300)

        fig.tight_layout()
        plt.show()



def create_datasets_file(xml_path, file_id, save_path, ):
    save_path = os.path.join(os.path.dirname(xml_path), 'ImageSets', 'Main')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ftest  = open(os.path.join(save_path,'test.txt'), 'w')  
    ftrain = open(os.path.join(save_path,'train.txt'), 'w')  
    fval   = open(os.path.join(save_path,'val.txt'), 'w') 



test = OperationDatasets(root_path=r'E:/hhh/yan/paper/datasets', classes_path=r'classes/pigeon_classes.txt')
test.dataset_statistics()
test.draw_datasets()

# # 指定数据集路径
# root_path = r'E:\hhh\yan\paper\datasets'
# xml_path = os.path.join(root_path, 'Annotations')
# # 类别文件路径
# classes_path = r'classes.txt'

# classes, _  = get_classes(classes_path)



# # 指定少样本的类别
# kiss_set = cls_set['kiss']
# grooming_set = cls_set['grooming']
# # kiss_set.symmetric_difference(grooming_set)

# # 指定少样本类别，根据比例在所有样本中进行划分数据
# train_res, val_res, test_res = choose_data(kiss_set)
# train_res_g, val_res_g, test_res_g = choose_data(grooming_set - kiss_set)

# train_res.symmetric_difference(train_res_g)
# val_res.symmetric_difference(val_res_g)
# test_res.symmetric_difference(test_res_g)

# # 除去指定类别之外，进行数据按比例分类
# other_set = cls_set[classes[0]].symmetric_difference(cls_set[classes[2]]).symmetric_difference(cls_set[classes[4]]) 
# train, val, test= choose_data(other_set)
# train.symmetric_difference(train_res)
# val.symmetric_difference(val_res)
# test.symmetric_difference(test_res)

# print()















def rename_file(path, shuffix):
    temp = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if file.endswith(shuffix):
                file_name, extension =  os.path.splitext(file)
                temp.append(file_name)
                new_name = 'hf2-' + file[4:]
                print(new_name)
                # os.rename(os.path.join(path, file), os.path.join(path, new_name))
                print(os.path.join(path, file), '-->' , os.path.join(path, new_name))
    print("finish!!")
    return temp

# root_path = r'E:\w\su\hb_B'
# xml_path = os.path.join(root_path, 'annotations')
# img_path = os.path.join(root_path, 'photos')

# imgs = rename_file(img_path, shuffix='jpg')
# xmls = rename_file(xml_path, shuffix='xml')
# res = set(imgs) - set(xmls)
# res2 = set(xmls) - set(imgs)
# print(res)
# print(res2)




# from PIL import Image
# import cv2 as cv

# img_path = os.path.join(root_path, 'photos')

# # img = Image.open(os.path.join(img_path, '27-2.png'))
# # img.convert('RGB')
# # img.save(os.path.join(img_path, '27-2.jpg'))


# img_ = os.path.join(img_path, '27-2.png')
# print(img_)
# img = cv.imread(img_, 1)
# print(img.shape)
# name = os.path.join(img_path, '27-2.jpg')
# print(name)
# cv.imwrite(name, img)



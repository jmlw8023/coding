# -*- encoding: utf-8 -*-
'''
@File    :   transfor_keypoint_2_coco_format.py
@Time    :   2023/09/19 16:02:29
@Author  :   hgh
@Version :   python3.9
@Contact :   jmlw8023@163.com
'''

# import packets
# import os
# 调用一些需要的第三方库
import numpy as np
import pandas as pd
import shutil
import json
import os
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random






# 定义关键点类别信息
categories =   [
        {
            "skeleton": [
                [0 ,1],
                [1 ,2],
                [2 ,3],
                [3 ,4],
                [0 ,5],
                [5 ,6],
                [6 ,7],
                [7 ,8],
                [0 ,9],
                [9 ,10],
                [10 ,11],
                [11 ,12],
                [0 ,13],
                [13 ,14],
                [14 ,15],
                [15 ,16],
                [0 ,17],
                [17 ,18],
                [18 ,19],
                [19, 20]
            ],
            "name": "hand", # 子类（具体类别）
            "supercategory": "hand", # 主类
            "id": 1,
            "keypoints": [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20"
            ]
        }
    ]
path = "./data/handpose_datasets_v2/"
path2 = "./data/handpose_datasets/"
train_json = []
val_json = []
num = 0
image_train = []
annotation_train = []
image_val = []
annotation_val = []
# 遍历标注图片
for img_id , f_ in enumerate(os.listdir(path)):
    if ".jpg" in f_:
        # 获取图片路径
        img_path = path + f_
        # 获取标签路径
        label_path = img_path.replace('.jpg','.json')
        image_id = img_id
        img = Image.open(img_path)
        width = img.width
        height = img.height
        if not os.path.exists(label_path):
            continue
        # 获取标注信息
        with open(label_path, encoding='utf-8') as f:
            hand_dict_ = json.load(f)
            f.close()
            hand_dict_ = hand_dict_["info"]
            # 初始化检测框坐标
            x_max = -65535
            y_max = -65535
            x_min = 65535
            y_min = 65535
            # 初始化关键点信息
            # "keypoints" : [x1,y1,v1,...], 关键点坐标，其中V字段表示关键点属性，0表示未标注，1表示已标注但不可见，2表示已标注且可见
            keypoints = []
            if len(hand_dict_) > 0:
                for msg in hand_dict_:
                    # 标注检测框坐标
                    bbox = msg["bbox"]
                    # 手势关键点坐标
                    pts = msg["pts"]
                    # 左手右手标注
                    handType  = msg["handType"]
                    # 找到外接矩形
                    x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                    # 遍历关键点坐标
                    for i in range(21):
                        x_,y_ = pts[str(i)]["x"],pts[str(i)]["y"]
                        x_ += x1
                        y_ += y1
                        # "keypoints" : [x1,y1,v1,...], 关键点坐标，其中V字段表示关键点属性，0表示未标注，1表示已标注但不可见，2表示已标注且可见
                        # 由于可见与否相当于需要再一次标注，本项目为节省时间，直接设置为已标注且可见：如果读者需要更好的检测效果，可以对标注进行再次矫正
                        keypoints.append([x_ ,y_, 2])
                        x_min = x_ if x_min > x_ else x_min
                        y_min = y_ if y_min > y_ else y_min
                        x_max = x_ if x_max < x_ else x_max
                        y_max = y_ if y_max < y_ else y_max
                    
                    area = float((x_max - x_min) * (y_max - y_min))
                    num_keypoint = 21

            # 准备COCO数据集中image和annotation字典段内容
            image_dict = {
            "id" : image_id,   # 图像id，可从0开始
            "width" : width,   # 图像的宽
            "height" : height,  # 图像的高
            "file_name" : f_   # 文件名
            }
            
            annotation_dict = {
            "id" : num,  # 注释id编号
            "image_id" : image_id,  # 图像id编号
            "segmentation" : [],  # 分割具体数据
            "area" : area,  # 目标检测的区域大小
            "bbox" : [x_min, y_min, int(x_max - x_min), int(y_max - y_min)],  # 目标检测框的坐标详细位置信息
            "iscrowd" : 0,
            "num_keypoints": num_keypoint,
            "keypoints": [k for s in keypoints for k in s],
            "category_id": 1
            }

        # 移动图片
        shutil.copy(img_path, path2 + f_)
        # 划分训练集和测试集
        if num % 5 == 0:
            image_val.append(image_dict)
            annotation_val.append(annotation_dict)
        else:
            image_train.append(image_dict)
            annotation_train.append(annotation_dict)
    num += 1              
# 准备训练集标注数据
item_train = {
"info" : {"version": "1.0", "description":"handpose keypoint dataset"},   # 数据集描述信息
"images" : image_train,   # 图像字典段列表信息
"annotations" : annotation_train,   # 注释类型字典段列表
"categories" : categories,
"licenses" : []  # 许可协议字典段列表
}
train_json.append(item_train)
# 准备验证集标注数据
item_val = {
"info" : {"version": "1.0", "description":"handpose keypoint dataset"},   # 数据集描述信息
"images" : image_val,   # 图像字典段列表信息
"annotations" : annotation_val,   # 注释类型字典段列表
"categories" : categories,
"licenses" : []  # 许可协议字典段列表
}
val_json.append(item_val)
# 创建训练集标签
with open("./train_json.json","w") as f:
    json.dump(train_json[0],f)
# 创建测试集标签
with open("./val_json.json","w") as f:
    json.dump(val_json[0],f)













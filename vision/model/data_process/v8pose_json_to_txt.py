# -*- encoding: utf-8 -*-
'''
@File    :   yolo_pose_data_process.py
@Time    :   2024/05/19 13:03:03
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# 关键点检测json转txt
import os
import json
import shutil
import time
 
import numpy as np
from tqdm import tqdm
 


# 框的类别
bbox_class =["water"]
 
# 关键点的类别，有多少类就写多少
keypoint_class = ['0', '1', '2', '3']
 
 
def process_single_json(labelme_path, save_folder='./data/labels'):

    assert os.path.isfile(labelme_path), 'input path is not file'
    assert os.path.isdir(save_folder), 'input save path is not folder'

    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)
 
    img_width = labelme['imageWidth']  # 图像宽度
    img_height = labelme['imageHeight']  # 图像高度

    os.makedirs(save_folder, exist_ok=True)
 
    # 生成 YOLO 格式的 txt 文件
    # suffix = labelme_path.split('.')[-2]
    # yolo_txt_path = suffix + '.txt'

    name, shffix = os.path.splitext(os.path.basename(labelme_path))
    yolo_txt_path = os.path.join(save_folder, name + '.txt')
 
    with open(yolo_txt_path, 'w', encoding='utf-8') as f:
 
        for each_ann in labelme['shapes']:  # 遍历每个标注
 
            if each_ann['shape_type'] == 'rectangle':  # 每个框，在 txt 里写一行
 
                yolo_str = ''
                ## 框的信息
                # 框的类别 ID
                bbox_class_id = bbox_class.index(each_ann['label'])
                # print(bbox_class_id)
 
                yolo_str += '{} '.format(bbox_class_id)

                # 左上角和右下角的 XY 像素坐标
                bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))
                # 框中心点的 XY 像素坐标
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
                # 框宽度
                bbox_width = bbox_bottom_right_x - bbox_top_left_x
                # 框高度
                bbox_height = bbox_bottom_right_y - bbox_top_left_y
                # 框中心点归一化坐标
                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height
                # 框归一化宽度
                bbox_width_norm = bbox_width / img_width
                # 框归一化高度
                bbox_height_norm = bbox_height / img_height
 
                yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm,
                                                                  bbox_width_norm, bbox_height_norm)
                # print(yolo_str)
                # print("**********************")
                # time.sleep(90000)
                ## 找到该框中所有关键点，存在字典 bbox_keypoints_dict 中
                bbox_keypoints_dict = {}
                for each_ann in labelme['shapes']:  # 遍历所有标注
                    if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                        # 关键点XY坐标、类别
                        x = int(each_ann['points'][0][0])
                        y = int(each_ann['points'][0][1])
                        label = each_ann['label']
                        if (x > bbox_top_left_x) & (x < bbox_bottom_right_x) & (y < bbox_bottom_right_y) & (
                                y > bbox_top_left_y):  # 筛选出在该个体框中的关键点
                            bbox_keypoints_dict[label] = [x, y]
 
                ## 把关键点按顺序排好
                for each_class in keypoint_class:  # 遍历每一类关键点
                    if each_class in bbox_keypoints_dict:
                        keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                        keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                        yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm,
                                                               2)  # 2-可见不遮挡 1-遮挡 0-没有点
                    else:  # 不存在的点，一律为0
                        yolo_str += '0 0 0 '
                # 写入 txt 文件中
                f.write(yolo_str + '\n')
 
    # shutil.move(yolo_txt_path, save_folder)
    print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path))
 


 # 转化的json文件地址
json_path = r'./json_data'    # json文件路径
#  转换后的训练集标注文件至目录
save_folder = r'./labels'   # txt文本文件路径

for file in os.listdir(json_path):
    # try:
    json_file = os.path.join(json_path, file)
    print(json_file)
    process_single_json(json_file, save_folder=save_folder)
    # except:
    #     print('******有误******', labelme_path)
print('YOLO格式的txt标注文件已保存至 ', save_folder)



# cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)

# -*- ecoding: utf-8 -*-
# @FlieName: datasets_split.py
# @Author: hgh
# @Time: 24/02/2022 09:59
# @Function: 进行数据集划分：训练：验证：测试 = 6 ： 2 ： 2 或者 = 8 ： 1 ： 1


import os
import math
import random
import shutil
import json
import argparse
import numpy as np
from pathlib import Path

"""
in param: folder_path 输入文件夹
in param: val_percent 验证集占比
in param: train_percent 训练集在占比
return:  返回训练，验证图像名序列
"""

# 输入文件夹名称，增加list中的数据 返回三个文件夹名称
def one_folder_return_3folder(folder_path, lis=['train', 'val', 'test']):
    res = []
    if os.path.isdir(folder_path):
        for name in lis:
            res.append(os.path.join(folder_path, name))

    return res

# 分割数据集
def split_train_val_test_datasets(folder_path=None,  test_percent=0.10, val_percent=0.10, train_percent=0.80):
    
    if not os.path.isdir(folder_path):
        print('请输入文件夹路径！')
        return -1
    
    img_folder = os.path.join(folder_path, 'images')
    json_folder = os.path.join(folder_path, 'annotations')
    txt_folder = os.path.join(folder_path, 'labels')



    total_img = os.listdir(os.path.join(folder_path, 'images'))
    nums_len = (len(total_img))
    print(nums_len)
    nums_lis = range(nums_len)

    test_num = int((nums_len * test_percent))
    val_num = int(math.ceil(nums_len * val_percent))
    train_num = int(round(nums_len * train_percent))

    print(train_num)
    print(val_num)
    print(test_num)


    train = np.random.choice(nums_lis, size=train_num, replace=False).tolist()
    nums_lis = set(nums_lis).difference(set(train)) # - set(train) # 
    val= np.random.choice(list(nums_lis), size=val_num, replace=False).tolist()
    test = list(set(nums_lis).difference(set(val)) ) # - set(val)

    name_train = open(os.path.join(img_folder, 'name_train.txt'), 'w')
    name_val = open(os.path.join(img_folder, 'name_val.txt'), 'w')
    name_test = open(os.path.join(img_folder, 'name_test.txt'), 'w')
    f_train = open(os.path.join(img_folder, 'train.txt'), 'w')
    f_val = open(os.path.join(img_folder, 'val.txt'), 'w')
    f_test = open(os.path.join(img_folder, 'test.txt'), 'w')


    data_lis = ['train', 'val', 'test']
    for name in data_lis:
        img_path = os.path.join(img_folder, name)
        json_path = os.path.join(json_folder, name)
        txt_path = os.path.join(txt_folder, name)
        mkdir(img_path)
        mkdir(json_path)
        mkdir(txt_path)

    # 存储文件夹    依此返回：'train', 'val', 'test' 路径
    save_img_path = one_folder_return_3folder(img_folder)
    if not save_img_path:
        print('save_img_path error!')
        return -1
    save_json_path = one_folder_return_3folder(json_folder)
    if not save_json_path:
        print('save_img_path error!')
        return -1
    save_txt_path = one_folder_return_3folder(txt_folder)
    if not save_txt_path:
        print('save_img_path error!')
        return -1

    tr_num, tv_num, tt_num = 0, 0, 0
    for i in range(nums_len):
            name_ = total_img[i][:-4]
            file = total_img[i]

            name_img = os.path.join(folder_path, 'images', file)
            name_json = os.path.join(folder_path, 'annotations', name_ + '.json')
            name_txt = os.path.join(folder_path, 'labels', name_ + '.txt')
            name_ += '\n'
            if i in train:
                if os.path.isfile(name_img):
                    name_train.write(name_)
                    f_train.write(name_img)
                    f_train.write('\n')
                    # shutil.move(name, dst_train_dir)
                    shutil.move(name_img, save_img_path[0])
                    shutil.move(name_json, save_json_path[0])
                    shutil.move(name_txt, save_txt_path[0])
                    tr_num += 1
                    print('移动 ', name_img, '------>', save_img_path[0], '成功！')

                # train_lis.append(name)
            elif i in val:
                if os.path.isfile(name_img):
                    name_val.write(name_)
                    f_val.write(name_img)
                    f_val.write('\n')
                    # shutil.move(name, dst_val_dir)
                    shutil.move(name_img, save_img_path[1])
                    shutil.move(name_json, save_json_path[1])
                    shutil.move(name_txt, save_txt_path[1])
                    tv_num += 1
                    print('移动 ', name_img, '------>', save_img_path[1], '成功！')
            else:
                if os.path.isfile(name_img):
                    name_test.write(name_)
                    f_test.write(name_img)
                    f_test.write('\n')
                    # shutil.move(name, dst_val_dir)
                    shutil.move(name_img, save_img_path[2])
                    shutil.move(name_json, save_json_path[2])
                    shutil.move(name_txt, save_txt_path[2])
                    tt_num += 1
                    print('移动 ', name_img, '------>', save_img_path[2], '成功！')
    
    f_train.close()
    f_val.close()
    f_test.close()
    name_train.close()
    name_val.close()
    name_test.close()
    print('移动 train = {} 个, val = {} 个, test = {} 个文件！'.format(tr_num, tv_num, tt_num))

    

# 分割三角板关键点数据集：训练：验证 = 9 ： 1
def split_triangle_datasets(folder_path=None, val_percent=0.10, train_percent=0.90):

    if not os.path.isdir(folder_path):
        print('请输入文件夹路径！')
        return -1
    total_img = os.listdir(os.path.join(folder_path, 'images'))

    dst_train_dir_img = os.path.join(
        folder_path, 'datasets', 'images', 'train')
    dst_train_dir_label = os.path.join(
        folder_path, 'datasets', 'annotations', 'train')
    dst_val_dir_img = os.path.join(folder_path, 'datasets', 'images', 'val')
    dst_val_dir_label = os.path.join(
        folder_path, 'datasets', 'annotations', 'val')

    mkdir(dst_train_dir_img)
    mkdir(dst_train_dir_label)
    mkdir(dst_val_dir_img)
    mkdir(dst_val_dir_label)

    nums_len = len(total_img)
    print(nums_len)
    nums_lis = range(nums_len)

    tv = int(math.ceil(nums_len * val_percent))
    tr = int(round(nums_len * train_percent))

    train = random.sample(nums_lis, tr)
    # val= random.sample(train, tr)


    name_train = open(os.path.join(folder_path, 'datasets',
                      'images', 'name_train.txt'), 'w')
    name_val = open(os.path.join(folder_path, 'datasets',
                    'images', 'name_val.txt'), 'w')
    f_train = open(os.path.join(
        folder_path, 'datasets', 'images', 'train.txt'), 'w')
    f_val = open(os.path.join(
        folder_path, 'datasets', 'images', 'val.txt'), 'w')

    train_lis, val_lis = [], []
    for i in nums_lis:
        name_ = total_img[i][:-4]
        file = total_img[i]
        # name = random.shuffle(names)
        # print(name)
        name_img = os.path.join(folder_path, 'images', file)
        name_json = os.path.join(folder_path, 'labelme_jsons', name_ + '.json')
        name_ += '\n'
        if i in train:
            if os.path.isfile(name_img):
                name_train.write(name_)
                f_train.write(name_img)
                f_train.write('\n')
                # shutil.move(name, dst_train_dir)
                shutil.copy(name_img, dst_train_dir_img)
                shutil.copy(name_json, dst_train_dir_label)
                print('移动 ', name_img, '------>', dst_train_dir_img, '成功！')

            # train_lis.append(name)
        else:
            if os.path.isfile(name_img):
                name_val.write(name_)
                f_val.write(name_img)
                f_val.write('\n')
                # shutil.move(name, dst_val_dir)
                shutil.copy(name_img, dst_val_dir_img)
                shutil.copy(name_json, dst_val_dir_label)
                print('移动 ', name_img, '------>', dst_val_dir_img, '成功！')

    f_train.close()
    f_val.close()
    name_val.close()
    name_val.close()
    # return train_lis, val_lis

# 转换 单个 使用labelme标注的三角板关键点json转为YOLO的txt文件
def process_single_json(opt):
    print(opt)
    print('--------------opt----------')
    labelme_folder = opt.json_path
    save_folder=opt.save_path

    if os.path.isdir(labelme_folder):
        for name in os.listdir(labelme_folder):
            labelme_path = os.path.join(labelme_folder, name)
            if os.path.isfile(labelme_path):
                with open(labelme_path, 'r', encoding='utf-8') as f:
                    labelme = json.load(f)

                img_width = labelme['imageWidth']   # 图像宽度
                img_height = labelme['imageHeight']  # 图像高度

                # 生成 YOLO 格式的 txt 文件
                suffix = labelme_path.split('.')[-2]
                yolo_txt_path = suffix + '.txt'

                with open(yolo_txt_path, 'w', encoding='utf-8') as f:

                    for each_ann in labelme['shapes']:  # 遍历每个标注

                        if each_ann['shape_type'] == 'rectangle':  # 每个框，在 txt 里写一行

                            yolo_str = ''

                            # 框的信息
                            # 框的类别 ID
                            bbox_class_id = opt.bbox_class[each_ann['label']]
                            yolo_str += '{} '.format(bbox_class_id)
                            # 左上角和右下角的 XY 像素坐标
                            bbox_top_left_x = int(
                                min(each_ann['points'][0][0], each_ann['points'][1][0]))
                            bbox_bottom_right_x = int(
                                max(each_ann['points'][0][0], each_ann['points'][1][0]))
                            bbox_top_left_y = int(
                                min(each_ann['points'][0][1], each_ann['points'][1][1]))
                            bbox_bottom_right_y = int(
                                max(each_ann['points'][0][1], each_ann['points'][1][1]))
                            # 框中心点的 XY 像素坐标
                            bbox_center_x = int(
                                (bbox_top_left_x + bbox_bottom_right_x) / 2)
                            bbox_center_y = int(
                                (bbox_top_left_y + bbox_bottom_right_y) / 2)
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

                            yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(
                                bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)

                            # 找到该框中所有关键点，存在字典 bbox_keypoints_dict 中
                            bbox_keypoints_dict = {}
                            for each_ann in labelme['shapes']:  # 遍历所有标注
                                if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                                    # 关键点XY坐标、类别
                                    x = int(each_ann['points'][0][0])
                                    y = int(each_ann['points'][0][1])
                                    label = each_ann['label']
                                    if (x > bbox_top_left_x) & (x < bbox_bottom_right_x) & (y < bbox_bottom_right_y) & (y > bbox_top_left_y):  # 筛选出在该个体框中的关键点
                                        bbox_keypoints_dict[label] = [x, y]

                            # 把关键点按顺序排好
                            for each_class in opt.keypoint_class:  # 遍历每一类关键点
                                if each_class in bbox_keypoints_dict:
                                    keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                                    keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                                    # 2-可见不遮挡 1-遮挡 0-没有点
                                    yolo_str += '{:.5f} {:.5f} {} '.format(
                                        keypoint_x_norm, keypoint_y_norm, 2)
                                else:  # 不存在的点，一律为0
                                    yolo_str += '0 0 0 '
                            # 写入 txt 文件中
                            f.write(yolo_str + '\n')

                # shutil.move(yolo_txt_path, save_folder)
                print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path))

# 批量 转换 使用labelme标注的三角板关键点json转为YOLO的txt文件
def convert_triangle_json_2_txt(root_path=None):
    # root_path = r'/home/hgh/source/datas/keypoints/triangle_215_Keypoint_Labelme/datasets'

    # 检测框 类别名称，和对应标签号
    bbox_class = { 'hand': 0  }
    # 关键点的类别
    # keypoint_class = ['angle_30', 'angle_60', 'angle_90']
    keypoint_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

    # for name in ['train', 'val']:

    #     json_save_path = os.path.join(root_path, 'labels/{}'.format(name))
    #     json_path = os.path.join(root_path, 'annotations/{}'.format(name))
    json_save_path = os.path.join(root_path, 'labels')
    json_path = os.path.join(root_path, 'annotations')


    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_class',
                    default=bbox_class, type=dict,
                    help="input: coco format(dict)")
    parser.add_argument('--keypoint_class',
                    default=keypoint_class, type=list,
                    help="input: keypoint_class format(list)")
    # 这里根据自己的json文件位置，换成自己的就行
    parser.add_argument('--json_path',
                        default=json_path, type=str,
                        help="input: coco format(json)")
    # 这里设置.txt文件保存位置
    parser.add_argument('--save_path', default=json_save_path, type=str,
                        help="specify where to save the output dir of labels")
    arg = parser.parse_args()

    # json_file = arg.json_path  # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径

    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    process_single_json(arg)


def mkdir(path):
    if os.path.isdir(os.path.dirname(path)):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print('创建{}成功！！'.format(path))

    else:
        print('创建{}失败！！'.format(path))



def read_txt_file_repair_context():

    des_folder = '/home/hgh/source/datas/keypoints/palm-3702/labels/test'

    src_folder = "/home/hgh/source/datas/keypoints/palm-3702/labels_old/test"
    lis_files = os.listdir(src_folder)
    for file in lis_files:
        src_txt_path = os.path.join(src_folder, file)
        des_txt_path = os.path.join(des_folder, file)

        # name = Path(txt_path).name
        # print(name)

        # with open(os.path.join(des_folder, name), 'w', encoding='utf-8') as txt_file_f:
        with open(des_txt_path, 'w', encoding='utf-8') as txt_file_f:

            with open(src_txt_path, 'r', encoding='utf-8') as txt_f:
                for line in txt_f.readlines():
                    # print(line)
                    # print(type(line))
                    msg = line.replace('hand', '20')
                    txt_file_f.writelines(msg)
                    print('wite {} successful!'.format(des_txt_path))
        
    
    print('total {} files!'.format(len(lis_files)))


            



def test():

    nums = range(100)
    print(random.sample(nums, 30))


if __name__ == '__main__':

    # # 保证随机可复现
    random.seed(0)

    # 一、json 转换为txt格式数据
    # convert_triangle_json_2_txt()
    # 分割数据集：训练：验证：测试 = 8:1:1
    folder_path = r'/home/hgh/source/datas/keypoints/rgb_palm_keypoints_data_8653/labels'
    # split_train_val_test_datasets(folder_path)

    lis=['train', 'val', 'test']
    # lis=['val']

    count = 0
    for i in lis:
        txt_folder = os.path.join(folder_path, i)

        for name in os.listdir(txt_folder):
            txt_path = os.path.join(txt_folder, name)
            # print(txt_path)


    # txt_path = r'/home/hgh/source/datas/keypoints/rgb_palm_keypoints_data_8653/labels/G010542_l_17.txt'

            new_txt = None
            with open(txt_path, mode='r', encoding='utf-8') as f:
                txt = f.read()
                new_txt = txt[:-2]
                # print(new_txt)
                # print(type(new_txt))
                
            if new_txt is not None:

                if os.path.isfile(txt_path):

                    with open(txt_path, mode='w', encoding='utf-8') as f:
                        txt = f.write(new_txt)
                        count += 1

    print('一共修改 {} 个文件'.format(count))
    # total_img = os.listdir(os.path.join(folder_path, 'images'))
    # nums_len = (len(total_img))
    # print(nums_len)
    # nums_lis = range(nums_len)
    # test_percent=0.10
    # val_percent=0.10
    # train_percent=0.80
    # test_num = int((nums_len * test_percent))
    # val_num = int(math.ceil(nums_len * val_percent))
    # train_num = int(round(nums_len * train_percent))


    # train = np.random.choice(nums_lis, size=train_num, replace=False).tolist()
    # nums_lis = set(nums_lis).difference(set(train)) # - set(train) # 
    # val= np.random.choice(list(nums_lis), size=val_num, replace=False).tolist()
    # test = list(set(nums_lis).difference(set(val)) ) # - set(val)

    # # print(len(nums_lis))

    # print(len(train))
    # print(len(val))
    # print(len(test))
    # print(type(train))
    # print(type(val))
    # print(type(test))

    # root_datasets = r'/home/hgh/source/datas/keypoints/triangle_215_Keypoint_Labelme'
    # # path_name = os.path.join(root_datasets, 'datasets')
    # # mkdir(path_name)
    # split_triangle_datasets(root_datasets)

    # labelme 的关键点json格式转化为YOLO可训练的txt格式
    # convert_triangle_json_2_txt()

    # 修改txt中的hand字段为15
    # read_txt_file_repair_context()






    pass



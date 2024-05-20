# COCO 格式的数据集转化为 YOLO 格式的数据集
# --json_path 输入的json文件路径
# --save_path 保存的文件夹名字，默认为当前目录下的labels。

import os
import json
import argparse


from tqdm import tqdm
from pathlib import Path

# parser = argparse.ArgumentParser()
# # 这里根据自己的json文件位置，换成自己的就行
# parser.add_argument('--json_path',
#                     default=r'E:\val2017\annotations\val.json', type=str,
#                     help="input: coco format(json)")
# # 这里设置.txt文件保存位置
# parser.add_argument('--save_path', default=r'E:\val2017\annotations', type=str,
#                     help="specify where to save the output dir of labels")
# arg = parser.parse_args()


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)


def test(arg):
    json_file = arg.json_path  # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    # id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
    # with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
    #     # 写入classes.txt
    #     for i, category in enumerate(data['categories']):
    #         f.write(f"{category['name']}\n")
    #         id_map[category['id']] = i
    # # print(id_map)

    # 这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    # list_file = open(os.path.join(ana_txt_save_path, 'train2017.txt'), 'w')
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
                counter=0
                for i in range(len(ann["keypoints"])):
                    if ann["keypoints"][i] == 2 or ann["keypoints"][i] == 1 or ann["keypoints"][i] == 0:
                        f_txt.write(" %s " % format(ann["keypoints"][i],'6f'))
                        counter=0
                    else:
                        if counter==0:
                            f_txt.write(" %s " % round((ann["keypoints"][i] / img_width),6))
                        else:
                            f_txt.write(" %s " % round((ann["keypoints"][i] / img_height),6))
                        counter+=1
        f_txt.write("\n")
        f_txt.close()
    #     # 将图片的路径写入train2017或val2017的路径
    #     list_file.write('E:/edgeai-yolov5-yolo-pose/coco_kpts/images/train2017/%s.jpg\n' % (head))
    # list_file.close()




def single_convert_json2txt():
    
    # if os.path.isdir(json_path):
    pass
          





def cocojson_convert_yolotxt():
    root_path = r'/home/hgh/source/datas/keypoints/palm_keypoints_6749/images'
    label_path = r'/home/hgh/source/datas/keypoints/palm_keypoints_6749/labels'
    json_folder = r'/home/hgh/source/datas/keypoints/palm_keypoints_6749/annotations'

    label_path = r'/home/hgh/source/datas/keypoints/palm-3702/labs'
    json_folder = r'/home/hgh/source/datas/keypoints/palm-3702/annotations'
    bbox_class = { 'hand': 0 }
    # 关键点的类别
    keypoint_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    # keypoint_class = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

    # for folder_name in ['train', 'val', 'test']:

    # txt_save_path = os.path.join(label_path, (folder_name))
    # json_path = os.path.join(root_path, (folder_name))
    txt_save_path = os.path.join(label_path)
    json_path = os.path.join(json_folder)

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
    parser.add_argument('--save_path', default=txt_save_path, type=str,
                        help="specify where to save the output dir of labels")
    opt = parser.parse_args()

    # json_file = arg.json_path  # COCO Object Instance 类型的标注
    # ana_txt_save_path = arg.save_path  # 保存的路径

    
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    # img_folder= r'/home/hgh/source/datas/keypoints/palm_keypoints_6749/images'
    # json_folder= r'/home/hgh/source/datas/keypoints/palm_keypoints_6749/3702_Altered_Annotation'
    # json_path = r'/home/hgh/source/datas/keypoints/palm_keypoints_6749/3702_Altered_Annotation/Ir_1_348612532.json'

    # name = Path(json_path).name
    # print(name)

    # data = json.load(open(json_path, 'r'))
    # print(data)

    count = 0
    if os.path.isdir(json_path):
        for name in os.listdir(json_path):
                        name = name[:-5]
                        json_path = os.path.join(json_folder, name + '.json')
                        yolo_txt_path = os.path.join(txt_save_path, name + '.txt')
                        print('process {} file!--------------'.format(json_path))
                        if os.path.isfile(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                json_datas = json.load(f)

                            img_width = json_datas['imageWidth']   # 图像宽度
                            img_height = json_datas['imageHeight']  # 图像高度

                            # # 生成 YOLO 格式的 txt 文件
                            # suffix = json_path.split('.')[-2]
                            # print(suffix)
                            # yolo_txt_path = suffix + '.txt'
                            # print(yolo_txt_path)

                            with open(yolo_txt_path, 'w', encoding='utf-8') as f:

                                for each_ann in json_datas['shapes']:  # 遍历每个标注

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
                                        for each_ann in json_datas['shapes']:  # 遍历所有标注
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
                                                yolo_str += '0.00 0.00 0.00 '
                                        # 写入 txt 文件中
                                        f.write(yolo_str + '\n')
                                        count += 1

                            print('{} --> {} 转换完成'.format(json_path, yolo_txt_path))
    
    print('总共转换了 {} 个文件'.format(count))

cocojson_convert_yolotxt()



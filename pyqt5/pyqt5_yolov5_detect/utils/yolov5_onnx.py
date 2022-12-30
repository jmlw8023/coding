# -*- encoding: utf-8 -*-
'''
@File    :   yolov5.py
@Time    :   2022/12/26 12:35:05
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os
import random
import cv2 as cv
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont





class YOLOV5():    #yolov5 onnx推理
    def __init__(self,onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
    #     self.CLASSES = self.get_classes()



           
    # def get_classes(name_file='coco.names'):
    #     with open(name_file, 'r', encoding='utf-8') as f:
    #         classes = []
    #         for name in f.readlines():
    #             name = name.strip()
    #             classes.append(name)
    #         return classes

    def get_input_name(self):
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
    def get_output_name(self):
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
    def get_input_feed(self,img_tensor):
        input_feed={}
        for name in self.input_name:
            input_feed[name]=img_tensor
        return input_feed

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv.resize(srcimg, (neww, newh), interpolation=cv.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, cv.BORDER_CONSTANT,
                                            value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv.resize(srcimg, (neww, newh), interpolation=cv.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
        else:
            img = cv.resize(srcimg, self.input_shape, interpolation=cv.INTER_AREA)
        return img, newh, neww, top, left


    def inference(self, img):
        # img=cv.imread(img)   #读取图片

        or_img = cv.resize(img,(640,640))
        # or_img = cv.resize(img,(640,640))
        # print(f'org_img type is =  {type(or_img)}')
        img = or_img[:,:,::-1].transpose(2,0,1)  #BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img,axis=0)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None,input_feed)[0]
        return pred, or_img



def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def filter_box(org_box, conf_thres=0.5, iou_thres=0.5): #过滤掉无用的框

    org_box=np.squeeze(org_box)                     #删除为1的维度
    conf = org_box[..., 4] > conf_thres             #删除置信度小于conf_thres的BOX
    # print(conf)
    box = org_box[conf == True]
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))                        #删除重复的类别
    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls                #将第6列元素替换为类别下标
                curr_cls_box.append(box[j][:6])     #当前类别的BOX
        curr_cls_box = np.array(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = nms(curr_cls_box,iou_thres)  #经过非极大抑制后输出的BOX下标
        for k in curr_out_box:
            output.append(curr_cls_box[k])          #利用下标取出非极大抑制后的BOX
    output = np.array(output)
    return output



def nms(dets, thresh): #非极大抑制
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1] #置信度从大到小排序（下标）

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # 计算相交面积
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # 当两个框不想交时x22 - x11或y22 - y11 为负数，
                                           # 两框不相交时把相交面积置0
        h = np.maximum(0, y22 - y11 + 1)  #

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)#计算IOU

        idx = np.where(ious <= thresh)[0]  #IOU小于thresh的框保留下来
        index = index[idx + 1]  # 下标以1开始
        # print(index)

    return keep
















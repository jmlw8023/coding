#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   predict_pose.py
@Time    :   2024/05/17 14:19:16
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# link: https://blog.csdn.net/m0_70694811/article/details/138118137
# import module
import cv2  # 导入 OpenCV 库进行图像处理
import numpy as np  # 导入 numpy 库进行数值操作
from ultralytics import YOLO  # 从 ultralytics 包中导入 YOLO 模型
 
 
import torch
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
 
def preprocess_letterbox(image):
    letterbox = LetterBox(new_shape=640, stride=32, auto=True)
    image = letterbox(image=image)
    image = (image[..., ::-1] / 255.0).astype(np.float32) # BGR to RGB, 0 - 255 to 0.0 - 1.0
    image = image.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
    image = torch.from_numpy(image)
    return image
 
def preprocess_warpAffine(image, dst_width=640, dst_height=640):
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    ox = (dst_width  - scale * image.shape[1]) / 2
    oy = (dst_height - scale * image.shape[0]) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)
    
    img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    IM = cv2.invertAffineTransform(M)
 
    img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre, IM
 
def iou(box1, box2):
    def area_box(box):
        return (box[2] - box[0]) * (box[3] - box[1])
 
    left   = max(box1[0], box2[0])
    top    = max(box1[1], box2[1])
    right  = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    cross  = max((right-left), 0) * max((bottom-top), 0)
    union  = area_box(box1) + area_box(box2) - cross
    if cross == 0 or union == 0:
        return 0
    return cross / union
 
def NMS(boxes, iou_thres):
 
    remove_flags = [False] * len(boxes)
 
    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue
 
        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if remove_flags[j]:
                continue
 
            jbox = boxes[j]
            if iou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes
 
def postprocess(pred, IM=[], conf_thres=0.25, iou_thres=0.45):
 
    # 输入是模型推理的结果，即8400个预测框
    # 1,8400,56 [cx,cy,w,h,conf,17*3]
    boxes = []
    
    # for img_id, box_id in zip(*np.where(pred[...,4] > conf_thres)):
    for img_id, box_id in enumerate(*np.where(pred[...,4] > conf_thres)):
        # print()
        # item = pred[img_id, box_id]
        item = pred[box_id]
        cx, cy, w, h, conf = item[:5]
        left    = cx - w * 0.5
        top     = cy - h * 0.5
        right   = cx + w * 0.5
        bottom  = cy + h * 0.5
        keypoints = item[5:].reshape(-1, 3)
        keypoints[:, 0] = keypoints[:, 0] * IM[0][0] + IM[0][2]
        keypoints[:, 1] = keypoints[:, 1] * IM[1][1] + IM[1][2]
        boxes.append([left, top, right, bottom, conf, *keypoints.reshape(-1).tolist()])
 
    boxes = np.array(boxes)
    lr = boxes[:,[0, 2]]
    tb = boxes[:,[1, 3]]
    boxes[:,[0,2]] = IM[0][0] * lr + IM[0][2]
    boxes[:,[1,3]] = IM[1][1] * tb + IM[1][2]
    boxes = sorted(boxes.tolist(), key=lambda x:x[4], reverse=True)
    
    return NMS(boxes, iou_thres)
 

  
# 将 HSV 颜色转换为 BGR 颜色的函数
def hsv2bgr(h, s, v):
    h_i = int(h * 6)  # 将色调转换为整数值
    f = h * 6 - h_i  # 色调的小数部分
    p = v * (1 - s)  # 计算不同情况下的值
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0  # 将 RGB 值初始化为 0
 
    # 根据色调值确定 RGB 值
    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q
 
    return int(b * 255), int(g * 255), int(r * 255)  # 返回缩放到 255 的 BGR 值
 
# 根据 ID 生成随机颜色的函数
def random_color(id):
    # 根据 ID 生成色调和饱和度值
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)  # 基于 HSV 值返回 BGR 颜色
 
# 定义关键点之间的骨骼连接
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
 
# 定义关键点和肢体的调色板
pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
 
# 基于调色板为关键点和肢体分配颜色
kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

colors = np.random.uniform(0, 255, size=(80, 3))

 
# 主函数
if __name__ == "__main__":
 
    # yolo export model=./yolov8n-pose.pt  imgsz=640 format=onnx opset=12 simplify=True

    # ####################################  自定义后处理进行检测 #######################################################
    img = cv2.imread("./bus.jpg")
    if False:
        # 读取输入图像
        # img = cv2.imread("ultralytics/assets/bus.jpg")
        
        # img = preprocess_letterbox(img)
        img_pre, IM = preprocess_warpAffine(img)
    
        model  = AutoBackend(weights="../ultralytics/yolov8n-pose.onnx")
        names  = model.names
        result = model(img_pre)[0].transpose(-1, -2)  # 1,8400,56
    
        boxes = postprocess(result, IM)
    
        for box in boxes:
            left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            confidence = box[4]
            label = 0
            color = random_color(label)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
            caption = f"{names[label]} {confidence:.2f}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
            cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
            
            keypoints = box[5:]
            keypoints = np.array(keypoints).reshape(-1, 3)
            for i, keypoint in enumerate(keypoints):
                x, y, conf = keypoint
                color_k = [int(x) for x in kpt_color[i]]
                if conf < 0.5:
                    continue
                if x != 0 and y != 0:
                    cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)
            
            for i, sk in enumerate(skeleton):
                pos1 = (int(keypoints[(sk[0] - 1), 0]), int(keypoints[(sk[0] - 1), 1]))
                pos2 = (int(keypoints[(sk[1] - 1), 0]), int(keypoints[(sk[1] - 1), 1]))
    
                conf1 = keypoints[(sk[0] - 1), 2]
                conf2 = keypoints[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
            
        cv2.imwrite("infer-pose.jpg", img)
        print("save done")
    
    
    
    ######################################## 使用官方原始进行检测  #######################################################
    if True:
        # 加载 YOLO 模型
        model = YOLO("../ultralytics/yolov8n-pose.onnx", task='pose')
        # 使用 YOLO 进行目标检测
        # results = model(img)[0]
        results = model(img)
        results = results[0]
        names   = results.names  # 获取类别名称
        boxes   = results.boxes.data.tolist()  # 获取边界框
    
        # 获取模型检测到的关键点
        keypoints = results.keypoints.cpu().numpy()
    
        # 为每个检测到的人绘制关键点和肢体
        for keypoint in keypoints.data:
            for i, (x, y, conf) in enumerate(keypoint):
                color_k = [int(x) for x in kpt_color[i]]  # 获取关键点的颜色
                if conf < 0.5:
                    continue
                if x != 0 and y != 0:
                    cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)  # 绘制关键点
            for i, sk in enumerate(skeleton):
                pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))  # 获取肢体的第一个关键点的位置
                pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))  # 获取肢体的第二个关键点的位置
    
                conf1 = keypoint[(sk[0] - 1), 2]  # 第一个关键点的置信度
                conf2 = keypoint[(sk[1] - 1), 2]  # 第二个关键点的置信度
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)  # 绘制肢体
    
        # 绘制检测到的对象的边界框和标签
        for i, obj in enumerate(boxes):
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])  # 提取边界框坐标
            confidence = obj[4]  # 置信度分数
            label = int(obj[5])  # 类别标签
            # color = random_color(label)  # 为边界框获取随机颜色
            cv2.rectangle(img, (left, top), (right, bottom), color=colors[i], thickness=2, lineType=cv2.LINE_AA)  # 绘制边界框
            caption = f"{names[label]} {confidence:.2f}"  # 生成包含类名和置信度分数的标签
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]  # 获取文本大小
            cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), colors[-i], -1)  # 绘制标签背景的矩形
            cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)  # 放置标签文本
    
        # 保存标注后的图像
        cv2.imwrite("predict-pose.jpg", img)
        print("保存完成")  # 打印保存操作完成的消息
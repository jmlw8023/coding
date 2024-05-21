
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   image_rectification.py
@Time    :   2024/05/15 10:19:00
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module
import os
# import math
import json
import glob
import numpy as np
import cv2 as cv

from collections import defaultdict

from ultralytics import YOLO  # 从 ultralytics 包中导入 YOLO 模型


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 文字识别
from cnocr import CnOcr



class ImageCorrection(object):
    
    
    def __init__(self) -> None:
        self.classes = ["addB", "addS", "pitN", "pitF"]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.ocr = CnOcr()
    
    # 随机创建图像，并进行旋转
    def create_image(self, height=400, width=300):
        
        # 创建一个空白图像（比如 400x300 像素，3通道BGR，全黑）  
        bk_image = np.zeros((height, width, 3), np.uint8)  
        
        # bk_image += 255
        
        # 在图像上绘制矩形框（左上角(50,50)，右下角(200,200)，蓝色，-1表示填充）  
        cv.rectangle(bk_image, (60, 200), (130, 300), (0, 0, 255), 1)  
        
        # 3. 定义旋转中心、旋转角度和缩放因子  
        center = (width // 2, height // 2)  
        angle = 75  # 旋转45度  
        scale = 1.0  # 不缩放  

        # 获取旋转矩阵  
        rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)  
        
        # 5. 应用旋转矩阵并旋转图像  
        rotated_image = cv.warpAffine(bk_image, rotation_matrix, (width, height))  
        d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
        save_folder = os.path.join(d, 'results')
        os.makedirs(save_folder, exist_ok=True)
        # 6. 将旋转后的图像保存到文件  
        cv.imwrite('{}/rotated_image75.jpg'.format(save_folder), rotated_image)  
        print('save image success to {}'.format(save_folder))



    def read_json_info(self, json_file):
        
        assert os.path.isfile(json_file), 'is not file!'
        assert str(json_file).lower().endswith('.json'), 'is not a json file!'
        
        
        with open(json_file, 'r') as f:
            
            data = json.load(f)
            
            # print(data)
            
            
        # 假设 JSON 文件中的 'shapes' 键包含了所有标注的形状  
        shapes = data['shapes']  
    
        rectangles = []  
        all_points = []  
        for shape in shapes:  
            # print(shape)
            # 检查形状类型  
            if shape['shape_type'] == 'rectangle':  
                # 提取矩形的左上角和右下角坐标  
                points = shape['points']  
                # print(points)
                rectangle = {  
                    'label': shape['label'],  
                    'points': [  
                        (points[0][0], points[0][1]),  # 左上角  
                        (points[1][0], points[1][1])  # 右下角（注意：矩形在 labelme 中通常定义为 4 个点，但左上角和右上角是相同的，左下角和右下角也是相同的）  
                    ]  
                }  
                rectangles.append(rectangle)  
                
            elif shape['shape_type'] == 'point':  
                # 提取点的坐标  
                # print(shape)
                point = {  
                    'label': shape['label'],  
                    'points': (shape['points'][0][0], shape['points'][0][1])  
                }  
                # print(point)
                all_points.append(point)  
        
        return rectangles, all_points  

    
    # 使用yolov8 pose关键点信息，之后进行校正检测数字
    def yolo_pose_detect(self, image_folder):
        
        assert os.path.isdir(image_folder), 'input path is not folder!'
            
        img_lis = glob.glob('{}/*.jpg'.format(image_folder))
        # print(json_lis)
        # json_path = json_lis[5]
        # print(json_path)
        # name, _ = os.path.splitext(os.path.basename(json_path))
        # img_path = os.path.join(image_folder, name + '.jpg')
        img_path = img_lis[22]
        print(img_path)
        
        assert os.path.isfile(img_path), 'image is not file!'
        img = cv.imread(img_path)
        
        model = YOLO("./weights/v8npose_water.onnx", task='pose')
        # 使用 YOLO 进行目标检测
        # results = model(img)[0]
        results = model(img)
        results = results[0]
        names   = results.names  # 获取类别名称
        boxes   = results.boxes.data.tolist()  # 获取边界框
    
        # 获取模型检测到的关键点
        keypoints = results.keypoints.cpu().numpy()
        
        # # 定义关键点和肢体的调色板
        # pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
        #                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
        #                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
        #                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
        
        # kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        # 检测到的绘制关键点
        pst_points = []
        for keypoint in keypoints.data:
            for i, (x, y, conf) in enumerate(keypoint):
                # color_k = [int(x) for x in kpt_color[i]]  # 获取关键点的颜色
                if conf < 0.5:
                    continue
                if x != 0 and y != 0:
                    cv.circle(img, (int(x), int(y)), 5, self.colors[i], -1, lineType=cv.LINE_AA)  # 绘制关键点
                    # if i == 0:
                    #     pst_points.append([x-10, y-10])
                    # else:
                    pst_points.append([x, y])
                    
        print(pst_points)
        
        # 绘制检测到的对象的边界框和标签
        for i, obj in enumerate(boxes):
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])  # 提取边界框坐标
            confidence = obj[4]  # 置信度分数
            label = int(obj[5])  # 类别标签
            # color = random_color(label)  # 为边界框获取随机颜色
            cv.rectangle(img, (left, top), (right, bottom), color=self.colors[i], thickness=2, lineType=cv.LINE_AA)  # 绘制边界框
            caption = f"{names[label]} {confidence:.2f}"  # 生成包含类名和置信度分数的标签
            w, h = cv.getTextSize(caption, 0, 1, 2)[0]  # 获取文本大小
            cv.rectangle(img, (left - 3, top - 33), (left + w + 10, top), self.colors[-i], -1)  # 绘制标签背景的矩形
            cv.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)  # 放置标签文本
    
        # # # 显示校正后的图像  
        # cv.imshow('src', img)  
        # cv.imshow('dst', transformed_image)  
        # cv.imshow('bk_image', bk_image)  
        # cv.waitKey(0)
        
        # rectangles, points = self.read_json_info(json_path)
        # # print(rectangles)
        # # print(points)
        
        # pst_points = [list(points[0].get('points')), list(points[1].get('points')), list(points[2].get('points')), list(points[3].get('points'))]
        # # print(pst_points)
        # 看下是否是长边
        a = (abs(pst_points[1][0] - pst_points[0][0]))
        b = (abs(pst_points[1][1] - pst_points[0][1]))
        
        if b > 0:
            w = int(np.sqrt((a ** 2 + b ** 2)))     # 勾股定理
        else:
            w = int(a)
        
        x = (abs(pst_points[-1][0] - pst_points[0][0]))
        y = (abs(pst_points[-1][1] - pst_points[0][1]))
        if x > 0:
            h = int(np.sqrt((y ** 2 + x ** 2)))     # 勾股定理
        else:
            h = int(y)
            
        print(f'w = {w} , h = {h}')
        
        
        
        # 定义这四个点在输出图像中的位置（目标点，这里是一个矩形）  
        # pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])  
        dst = np.float32([[0, 0],[w,0],[w, h],[0,h]])  
        # dst = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])
        
        # 计算透视变换矩阵  
        M = cv.getPerspectiveTransform(np.float32(pst_points), dst)  
        
        height, width = 400, 300
        # 应用透视变换  
        # transformed_image = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))
        # transformed_image = cv.warpPerspective(image, M, (int(height*0.6), int(width*0.6)))
        transformed_image = cv.warpPerspective(img, M, (int(w), int(h)))
        
        # # bk_image = np.zeros((height, width, 3), np.uint8)  
        # # bk_image += 255
        bk_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        # bk_image = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # 计算变换后图像在白色背景上的位置
        x_offset = (width - transformed_image.shape[1]) // 2
        y_offset = (height - transformed_image.shape[0]) // 2
        print(x_offset, y_offset)
        bk_image[y_offset:y_offset+transformed_image.shape[0], x_offset:x_offset+transformed_image.shape[1]] = transformed_image
        
        
        # out = self.ocr.ocr(transformed_image)
        out = self.ocr.ocr_for_single_line(transformed_image)
        print(out)
        
        # # 使用ddddocr检测文本
        # # self.use_ddddocr_cls(bk_image)
        # # self.use_ddddocr_det(bk_image)
        
        # # 显示校正后的图像  
        cv.imshow('src', img)  
        # cv.imshow('dst', transformed_image)  
        cv.imshow('bk_image', bk_image)  
        
        cv.waitKey(0)
        
    
    # 使用labelme标注的关键点信息，之后进行校正检测数字
    def json_2_yolo_pose_txt(self, json_folder, image_folder):
        
        assert os.path.isdir(json_folder), 'input path is not folder!'
            
        json_lis = glob.glob('{}/*.json'.format(json_folder))
        print(json_lis)
        json_path = json_lis[8]
        
        name, _ = os.path.splitext(os.path.basename(json_path))
        img_path = os.path.join(image_folder, name + '.jpg')
        # print(img_path)
        
        rectangles, points = self.read_json_info(json_path)
        # print(rectangles)
        # print(points)
        
        pst_points = [list(points[0].get('points')), list(points[1].get('points')), list(points[2].get('points')), list(points[3].get('points'))]
        # print(pst_points)
        # 看下是否是长边
        a = (abs(pst_points[1][0] - pst_points[0][0]))
        b = (abs(pst_points[1][1] - pst_points[0][1]))
        
        if b > 0:
            w = int(np.sqrt((a ** 2 + b ** 2)))     # 勾股定理
        else:
            w = int(a)
        
        x = (abs(pst_points[-1][0] - pst_points[0][0]))
        y = (abs(pst_points[-1][1] - pst_points[0][1]))
        if x > 0:
            h = int(np.sqrt((y ** 2 + x ** 2)))     # 勾股定理
        else:
            h = int(y)
            
        print(f'w = {w} , h = {h}')
        
        assert os.path.isfile(img_path), 'image is not file!'
        
        image = cv.imread(img_path)
        # 定义这四个点在输出图像中的位置（目标点，这里是一个矩形）  
        # pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])  
        dst = np.float32([[0, 0],[w,0],[w, h],[0,h]])  
        # dst = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])
        
        # 计算透视变换矩阵  
        M = cv.getPerspectiveTransform(np.float32(pst_points), dst)  
        
        height, width = 400, 300
        # 应用透视变换  
        # transformed_image = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))
        # transformed_image = cv.warpPerspective(image, M, (int(height*0.6), int(width*0.6)))
        transformed_image = cv.warpPerspective(image, M, (int(w), int(h)))
        
        # # bk_image = np.zeros((height, width, 3), np.uint8)  
        # # bk_image += 255
        bk_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        # bk_image = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # 计算变换后图像在白色背景上的位置
        x_offset = (width - transformed_image.shape[1]) // 2
        y_offset = (height - transformed_image.shape[0]) // 2
        print(x_offset, y_offset)
        bk_image[y_offset:y_offset+transformed_image.shape[0], x_offset:x_offset+transformed_image.shape[1]] = transformed_image
        
        
        # out = self.ocr.ocr(transformed_image)
        out = self.ocr.ocr_for_single_line(transformed_image)
        print(out)
        
        # 使用ddddocr检测文本
        # self.use_ddddocr_cls(bk_image)
        # self.use_ddddocr_det(bk_image)
        
        # 显示校正后的图像  
        cv.imshow('src', image)  
        cv.imshow('dst', transformed_image)  
        cv.imshow('bk_image', bk_image)  
        
        cv.waitKey(0)
        
        
    # 使用ddddocr进行文字检测
    def use_ddddocr_det(self, cv_img):
        import ddddocr
        
        det = ddddocr.DdddOcr(det=True)
        # # OpenCV的numpy格式图像转为二进制数据格式
        binary_img = None
        if cv_img is not None:
            success, encoded_image = cv.imencode('.jpg', cv_img)  
            if success:  
                    # encoded_image是一个numpy数组，它包含一个元素，该元素是图像的二进制数据  
                    binary_img = encoded_image.tobytes() 

            if binary_img is not None:
                # 二进制格式数据
                # # 文字区域检测
                text_boxes = det.detection(binary_img)
                # # 检测框
                print(text_boxes)

                # 绘制检测框
                for box in text_boxes:
                    x1, y1, x2, y2 = box
                    cv_img = cv.rectangle(cv_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                # 可视化显示
                cv.imshow('det', cv.resize(cv_img, (720, 480)))
                # 一直显示
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                
    # # # 文字识别
    def use_ddddocr_cls(self, cv_img):
        import ddddocr
        
        if isinstance(cv_img, str):
            ocr = ddddocr.DdddOcr()
            result = ocr.classification(cv_img)
            print(result)
        
        elif isinstance(cv_img, np.ndarray):
            ocr = ddddocr.DdddOcr(det=False)
            # # OpenCV的numpy格式图像转为二进制数据格式
            binary_img = None
            if cv_img is not None:
                success, encoded_image = cv.imencode('.jpg', cv_img)  
                if success:  
                        # encoded_image是一个numpy数组，它包含一个元素，该元素是图像的二进制数据  
                        binary_img = encoded_image.tobytes() 

            if binary_img is not None:
                # 二进制格式数据
                # # 文字区域检测
                result = ocr.classification(binary_img)
                print(result)
       
                    
    # def preprocess_letterbox(image):
    #     letterbox = LetterBox(new_shape=640, stride=32, auto=True)
    #     image = letterbox(image=image)
    #     image = (image[..., ::-1] / 255.0).astype(np.float32) # BGR to RGB, 0 - 255 to 0.0 - 1.0
    #     image = image.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
    #     image = torch.from_numpy(image)
    #     return image
    
    # def preprocess_warpAffine(image, dst_width=640, dst_height=640):
    #     scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    #     ox = (dst_width  - scale * image.shape[1]) / 2
    #     oy = (dst_height - scale * image.shape[0]) / 2
    #     M = np.array([
    #         [scale, 0, ox],
    #         [0, scale, oy]
    #     ], dtype=np.float32)
        
    #     img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
    #                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    #     IM = cv2.invertAffineTransform(M)
    
    #     img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    #     img_pre = img_pre.transpose(2, 0, 1)[None]
    #     img_pre = torch.from_numpy(img_pre)
    #     return img_pre, IM
    
    # def iou(self, box1, box2):
    #     def area_box(box):
    #         return (box[2] - box[0]) * (box[3] - box[1])
    
    #     left   = max(box1[0], box2[0])
    #     top    = max(box1[1], box2[1])
    #     right  = min(box1[2], box2[2])
    #     bottom = min(box1[3], box2[3])
    #     cross  = max((right-left), 0) * max((bottom-top), 0)
    #     union  = area_box(box1) + area_box(box2) - cross
    #     if cross == 0 or union == 0:
    #         return 0
    #     return cross / union
    
    # def NMS(self, boxes, iou_thres):
    
    #     remove_flags = [False] * len(boxes)
    
    #     keep_boxes = []
    #     for i, ibox in enumerate(boxes):
    #         if remove_flags[i]:
    #             continue
    
    #         keep_boxes.append(ibox)
    #         for j in range(i + 1, len(boxes)):
    #             if remove_flags[j]:
    #                 continue
    
    #             jbox = boxes[j]
    #             if self.iou(ibox, jbox) > iou_thres:
    #                 remove_flags[j] = True
    #     return keep_boxes
    
    # def postprocess(self, pred, IM=[], conf_thres=0.25, iou_thres=0.45):
    
    #     # 输入是模型推理的结果，即8400个预测框
    #     # 1,8400,56 [cx,cy,w,h,conf,17*3]
    #     boxes = []
        
    #     # for img_id, box_id in zip(*np.where(pred[...,4] > conf_thres)):
    #     for img_id, box_id in enumerate(*np.where(pred[...,4] > conf_thres)):
    #         # print()
    #         # item = pred[img_id, box_id]
    #         item = pred[box_id]
    #         cx, cy, w, h, conf = item[:5]
    #         left    = cx - w * 0.5
    #         top     = cy - h * 0.5
    #         right   = cx + w * 0.5
    #         bottom  = cy + h * 0.5
    #         keypoints = item[5:].reshape(-1, 3)
    #         keypoints[:, 0] = keypoints[:, 0] * IM[0][0] + IM[0][2]
    #         keypoints[:, 1] = keypoints[:, 1] * IM[1][1] + IM[1][2]
    #         boxes.append([left, top, right, bottom, conf, *keypoints.reshape(-1).tolist()])
    
    #     boxes = np.array(boxes)
    #     lr = boxes[:,[0, 2]]
    #     tb = boxes[:,[1, 3]]
    #     boxes[:,[0,2]] = IM[0][0] * lr + IM[0][2]
    #     boxes[:,[1,3]] = IM[1][1] * tb + IM[1][2]
    #     boxes = sorted(boxes.tolist(), key=lambda x:x[4], reverse=True)
        
    #     return self.NMS(boxes, iou_thres)
    



if __name__ == '__main__':
    
    d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    json_folder = os.path.join(d, 'results', 'labels')
    image_folder = os.path.join(d, 'results', 'images')
    
    demo = ImageCorrection()
    
    # demo.create_image()
    # 使用labelme标注的关键点信息，之后进行校正检测数字
    # demo.json_2_yolo_pose_txt(json_folder, image_folder)
    
    # 使用yolov8pose进行推理检测得到关键点信息，之后进行校正得到文本信息
    demo.yolo_pose_detect(image_folder)
    
    
    
    
    
    pass






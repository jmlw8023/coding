
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


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 文字识别
from cnocr import CnOcr



class ImageCorrection(object):
    
    
    def __init__(self) -> None:

        
        
        self.ocr = CnOcr()
        
    
    
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
        a = (abs(pst_points[1][0] - pst_points[0][0]))
        b = (abs(pst_points[1][1] - pst_points[0][1]))
        
        if b > 0:
            w = int(np.sqrt((a ** 2 + b ** 2)))
        else:
            w = int(a)
        
        
        x = (abs(pst_points[-1][0] - pst_points[0][0]))
        y = (abs(pst_points[-1][1] - pst_points[0][1]))
        if x > 0:
            h = int(np.sqrt((y ** 2 + x ** 2)))
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
                
        
        


if __name__ == '__main__':
    
    d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    json_folder = os.path.join(d, 'results', 'labels')
    image_folder = os.path.join(d, 'results', 'images')
    
    demo = ImageCorrection()
    
    # demo.create_image()
    demo.json_2_yolo_pose_txt(json_folder, image_folder)
    
    
    
    
    pass






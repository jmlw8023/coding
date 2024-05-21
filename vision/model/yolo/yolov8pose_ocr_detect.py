
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2024/05/15 10:19:00
@Author  :    
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
        
        self.colors = np.random.uniform(0, 255, size=(5, 3))
        self.ocr = CnOcr()
        
        
        self.init_ocr()

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
                
                # return result
    
    # 初始化OCR
    def init_ocr(self):
        
        import fastdeploy as fd
        cls_bs = 1
        rec_bs = 6
        device = 'cpu'

        rec_model = r'./data/weights/ch_PP-OCRv3_rec_infer'
        rec_label_file = r'./data/weights/ppocr_keys_v1.txt'

        det_model = r'./data/weights/ch_PP-OCRv3_det_infer'

        rec_model_file = os.path.join(rec_model, "inference.pdmodel")
        rec_params_file = os.path.join(rec_model, "inference.pdiparams")

        det_model_file = os.path.join(det_model, "inference.pdmodel")
        det_params_file = os.path.join(det_model, "inference.pdiparams")

        det_option = fd.RuntimeOption()
        # cls_option = fd.RuntimeOption()
        rec_option = fd.RuntimeOption()

        det_option.use_ort_backend()
        # cls_option.use_ort_backend()
        rec_option.use_ort_backend()


        self.rec_model = fd.vision.ocr.Recognizer(
            rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

        self.det_model = fd.vision.ocr.DBDetector(
            det_model_file, det_params_file, runtime_option=det_option)


        # Parameters settings for pre and post processing of Det/Cls/Rec Models.
        # All parameters are set to default values.
        self.det_model.preprocessor.max_side_len = 960
        self.det_model.postprocessor.det_db_thresh = 0.3
        self.det_model.postprocessor.det_db_box_thresh = 0.6
        self.det_model.postprocessor.det_db_unclip_ratio = 1.5
        self.det_model.postprocessor.det_db_score_mode = "slow"
        self.det_model.postprocessor.use_dilation = False
        # det_model.postprocessor.det_db_score_mode = "slow"  # "middle"
        # det_model.postprocessor.use_dilation = False
        # # cls_model.postprocessor.cls_thresh = 0.9

        # Create PP-OCRv3, if cls_model is not needed, just set cls_model=None .
        self.ppocr_v3 = fd.vision.ocr.PPOCRv3(det_model=self.det_model, rec_model=self.rec_model)

        self.ppocr_v3.cls_batch_size = cls_bs
        self.ppocr_v3.rec_batch_size = rec_bs

    # 文件夹使用yolov8 pose关键点信息，之后进行校正检测数字
    def yolo_pose_detect(self, image_folder, save_result_flag=False):
        
        onnx_path = r"./data/weights/v8npose_water.onnx"
        assert os.path.isdir(image_folder), 'input path is not folder!'
        img_lis = glob.glob('{}/*.jpg'.format(image_folder))
            
        if save_result_flag:
            d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
            save_folder = os.path.join(d, 'data', 'results')
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, name)
            
        # img_path = img_lis[22]
        model = YOLO(onnx_path, task='pose')
        for img_path in img_lis:
            print(img_path)
            assert os.path.isfile(img_path), 'image is not file!'
            img = cv.imread(img_path)
            # 使用 YOLO 进行目标检测
            results = model(img)[0]
            names   = results.names  # 获取类别名称
            boxes   = results.boxes.data.tolist()  # 获取边界框
        
            # 获取模型检测到的关键点
            keypoints = results.keypoints.cpu().numpy()
            
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
            # print(pst_points)
            
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
                
            # print(f'w = {w} , h = {h}')
            
            # 定义输出图像四个点位置（目标点，这里是一个矩形）  
            dst = np.float32([[0, 0],[w,0],[w, h],[0,h]])  

            # 计算透视变换矩阵  
            M = cv.getPerspectiveTransform(np.float32(pst_points), dst)  
            
            height, width = 400, 300
            # 透视变换  
            transformed_image = cv.warpPerspective(img, M, (int(w), int(h)))
            
            bk_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # # 计算变换后图像在白色背景上的位置
            x_offset = (width - transformed_image.shape[1]) // 2
            y_offset = (height - transformed_image.shape[0]) // 2
            bk_image[y_offset:y_offset+transformed_image.shape[0], x_offset:x_offset+transformed_image.shape[1]] = transformed_image
            
            out = self.ocr.ocr_for_single_line(transformed_image)
            # print(out)
            number = out.get('text')
            print(number)
            
            if save_result_flag:
                name = os.path.basename(img_path)
                save_path_number = os.path.join(save_folder, name[:-4] + f'_num_{number}.jpg')
                cv.imwrite(save_path, img)
                cv.imwrite(save_path_number, transformed_image)
                print(f'save {name} success!')
        
            # # # 显示校正后的图像  
            # cv.imshow('src', img)  
            # cv.imshow('dst', transformed_image)  
            # cv.imshow('bk_image', bk_image)  
            
            # cv.waitKey(0)
            
  
    # 单张图像，使用yolov8 pose关键点信息，之后进行校正检测数字
    def yolo_pose_detect_oneimage(self, image_folder, save_result_flag=False):
        
        onnx_path = r"./data/weights/v8npose_water.onnx"

        assert os.path.isfile(img_path), 'image is not file!'
        img = cv.imread(img_path)
        
        source_img = img.copy()
        
        model = YOLO(onnx_path, task='pose')
        # 使用 YOLO 进行目标检测
        # results = model(img)[0]
        results = model(img)
        results = results[0]
        names   = results.names  # 获取类别名称
        boxes   = results.boxes.data.tolist()  # 获取边界框
    
        # 获取模型检测到的关键点
        keypoints = results.keypoints.cpu().numpy()
        
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
                    
        # print(pst_points)
        
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
            
        # print(f'w = {w} , h = {h}')
        
        # 定义输出图像四个点位置（目标点，这里是一个矩形）  
        dst = np.float32([[0, 0],[w,0],[w, h],[0,h]])  

        # 计算透视变换矩阵  
        M = cv.getPerspectiveTransform(np.float32(pst_points), dst)  
        
        height, width = 400, 300
        # 透视变换  
        transformed_image = cv.warpPerspective(source_img, M, (int(w), int(h)))
        
        bk_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        # 计算变换后图像在白色背景上的位置
        x_offset = (width - transformed_image.shape[1]) // 2
        y_offset = (height - transformed_image.shape[0]) // 2
        bk_image[y_offset:y_offset+transformed_image.shape[0], x_offset:x_offset+transformed_image.shape[1]] = transformed_image
    
        out = self.ocr.ocr_for_single_line(transformed_image)
        # print(out)
        number = out.get('text')
        print(number)

        # result = self.ppocr_v3.predict(transformed_image)
        # print(result)
        # for index, text in enumerate(result.text):
        #   print(text)
          
        # # # 接口绘制内容  Visuliaze the results.
        # vis_im = fd.vision.vis_ppocr(im, result)
        
        if save_result_flag:
            d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
            name = os.path.basename(img_path)
            save_folder = os.path.join(d, 'data', 'results')
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, name)
            save_path_number = os.path.join(save_folder, name[:-4] + f'_num_{number}.jpg')
            cv.imwrite(save_path, img)
            cv.imwrite(save_path_number, transformed_image)
            print(f'save {name} success!')
        
        # # # 显示校正后的图像  
        # cv.imshow('src', img)  
        # # cv.imshow('dst', transformed_image)  
        # cv.imshow('bk_image', bk_image)  
        
        # cv.waitKey(0)
        
  


if __name__ == '__main__':
    
    d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    image_folder = os.path.join(d, 'data', 'images')
    
    demo = ImageCorrection()
    
    
    assert os.path.isdir(image_folder), 'input path is not folder!'
        
    img_lis = glob.glob('{}/*.jpg'.format(image_folder))
    img_path = img_lis[21]
    print(img_path)
    
    # 单张图像
    demo.yolo_pose_detect_oneimage(img_path, save_result_flag=True)
    
    # # 文件夹：使用yolov8pose进行推理检测得到关键点信息，之后进行校正得到文本信息
    # demo.yolo_pose_detect(image_folder, save_result_flag=False)
    
    
    
    
    
    pass






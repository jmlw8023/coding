#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   ppocr3_class.py
@Time    :   2024/06/07 10:57:12
@Author  :   hgh 
@Version :   1.0
@Desc    :   使用ppocr识别检测图像文本，之后把相关信息写入到json中
'''

# import module
import os
import json
import cv2 as cv
import numpy as np
from collections import defaultdict
import fastdeploy as fd

# pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html

class TextDet(object):
    
    def __init__(self) -> None:
        
        # 获取当前py文件路径
        self.curr_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
        # 需要检测的文件夹目录
        self.detect_folder = os.path.join(self.curr_dir, 'data', 'images', '20240606')
        # self.detect_folder = os.path.join(self.curr_dir, 'data', 'images', '01')
        # self.detect_folder = os.path.join(self.curr_dir, 'data', 'images', '03')
        
        self.is_save_draw_image = True
        
        self.img_format_lists = [".tif", ".tiff", ".jpg", ".jpeg", ".gif", ".png", ".eps", ".raw", ".cr2", ".nef", ".orf", ".sr2", ".bmp", ".ppm", ".heif"]

        self.init_model()
        
    # 初始化检测模型
    def init_model(self):
        cls_bs = 1
        rec_bs = 6
        # device = 'cpu'

        rec_label_file = r'../data/weights/ppocr_keys_v1.txt'
        rec_model = r'../data/weights/ch_PP-OCRv3_rec_infer'
        det_model = r'../data/weights/ch_PP-OCRv3_det_infer'

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
        
        self.rec_model = fd.vision.ocr.Recognizer(rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)
        self.det_model = fd.vision.ocr.DBDetector(det_model_file, det_params_file, runtime_option=det_option)


        self.det_model.preprocessor.max_side_len = 960
        self.det_model.postprocessor.det_db_thresh = 0.3
        self.det_model.postprocessor.det_db_box_thresh = 0.6
        self.det_model.postprocessor.det_db_unclip_ratio = 1.5
        self.det_model.postprocessor.det_db_score_mode = "slow"
        self.det_model.postprocessor.use_dilation = False
        
        # det_model.preprocessor.max_side_len = 960
        # det_model.postprocessor.det_db_thresh = 0.3
        # det_model.postprocessor.det_db_box_thresh = 0.6
        # det_model.postprocessor.det_db_unclip_ratio = 1.5
        # det_model.postprocessor.det_db_score_mode = "slow"  # "middle"
        # det_model.postprocessor.use_dilation = False
        # # cls_model.postprocessor.cls_thresh = 0.9
        
        # 创建 PP-OCRv3 model
        self.ppocr_v3 = fd.vision.ocr.PPOCRv3(det_model=self.det_model,  rec_model=self.rec_model)
        self.ppocr_v3.cls_batch_size = cls_bs
        self.ppocr_v3.rec_batch_size = rec_bs
    
    # 检测
    def detect(self):
        assert os.path.isdir(self.detect_folder), '检测的文件夹有问题！'
        self.save_folder = os.path.join(self.curr_dir, 'data/results')
        folder_name = os.path.basename(self.detect_folder)
        self.save_json_path = os.path.join(self.save_folder,  folder_name, folder_name + '.json' )
        os.makedirs(self.save_folder, exist_ok=True)
        
        # 创建存储绘制图像文件夹
        if self.is_save_draw_image:
            img_save_folder = os.path.join(self.save_folder, folder_name, 'draw_images')
            os.makedirs(img_save_folder, exist_ok=True)
            
        data_lis = []
        # data = defaultdict(dict)
        
        count = 0
        img_path_lis = os.listdir(self.detect_folder)
        for img_name in img_path_lis:
            img_path = os.path.join(self.detect_folder, img_name)

            assert os.path.isfile(img_path), '输入文件有问题!'
            
            if img_path.lower().endswith(tuple(self.img_format_lists)):
                # Read the input image
                im = cv.imread(img_path)
                
                data = {}
                
                number_lis = []
                if im is not None:
                    
                    data['imgName'] = img_name
                    # 预测
                    result = self.ppocr_v3.predict(im)
                            
                    text_len = 3        # 文字长度
                    if len(result.text) > 0:
                        for index, text in enumerate(result.text):
                            if self.contains_digit(text):
                            # if self.is_all_digits(text):
                                num_len = self.digit_length(text)
                                if (num_len) <= text_len:
                                        number_dict = {}
                                        # print(index, text)
                                        # # text = keep_digits_and_letters(text)
                                        # contect += text
                                        # ui.textEdit_result.append(text)
                                        box = result.boxes[index]       # 4个坐标点，顺序为左下，右下，右上，左上                            
                                        left_up = box[6], box[7]
                                        right_down = box[2], box[3]
                                        
                                        number_dict['pointLx'] = box[6]
                                        number_dict['pointLy'] = box[7]
                                        number_dict['pointRx'] = box[2]
                                        number_dict['pointRy'] = box[3]
                                        number_dict['value'] = text
                                        number_lis.append(number_dict)
                                        count += 1
                                        # 生成一个随机的BGR颜色
                                        random_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

                                        cv.rectangle(im, left_up, right_down, random_color, thickness=2, lineType=cv.LINE_AA)
                    data['numbers'] = number_lis
                    data_lis.append(data)
                    if self.is_save_draw_image:
                        name, shuffix = os.path.splitext(os.path.basename(img_path))
                        cv.imwrite("{}".format(os.path.join(img_save_folder, name + '.png')), im)
                        print(f"Visualized result {name} save in {img_save_folder}")

            
            with open(self.save_json_path, 'w', encoding='utf-8') as f:
                json.dump(data_lis, f, ensure_ascii=False, indent=4)

    # 只包含数字的字符串  
    def is_all_digits(self, s):  
        import re
        # \d+ : 一个或多个数字  
        # ^ 和 $ 分别表示字符串的开始和结束，确保整个字符串都是数字  
        if re.match("^\d+$", s):  
            return True  
        else:  
            return False  
        
    # 检测是否包含数字
    def contains_digit(self, s):
        return any(c.isdigit() for c in s)
    
    # 检测是否包含数字，返回长度
    def digit_length(self, s):
        digit_count = 0
        current_count = 0
        for c in s:
            if c.isdigit():
                current_count += 1
            elif current_count > 0:
                digit_count += current_count
                current_count = 0
        if current_count > 0:
            digit_count += current_count
        return digit_count



if __name__ == '__main__':
    
    demo = TextDet()
    
    demo.detect()
    
    
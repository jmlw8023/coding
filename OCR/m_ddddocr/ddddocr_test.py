#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   ddddocr_test.py
@Time    :   2024/05/16 13:05:45
@Author  :   hgh
@Version :   1.0
@Desc    :   使用ddddocr  可以识别中文、英文、数字
'''

# import module
import os
import cv2 as cv
import ddddocr

# git 网站：https://github.com/sml2h3/ddddocr   https://gitcode.com/sml2h3/ddddocr/overview      


img_path = r'../data/images/car_number.jpg'


############################# 分类 ######################################
# # # 文字识别
ocr = ddddocr.DdddOcr()
image = open(img_path, "rb").read()
result = ocr.classification(image)
print(result)

############################# 检测 ######################################

det = ddddocr.DdddOcr(det=True)


im = cv.imread(img_path)

# # 文字区域检测
text_boxes = det.detection(image)
# # 检测框
# print(text_boxes)

# 绘制检测框
for box in text_boxes:
    x1, y1, x2, y2 = box
    im = cv.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

# 可视化显示
cv.imshow('res', cv.resize(im, (720, 480)))
# 一直显示
cv.waitKey(0)
cv.destroyAllWindows()



print('finish!')

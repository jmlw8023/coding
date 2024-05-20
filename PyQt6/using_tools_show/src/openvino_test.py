#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   openvino_test.py
@Time    :   2024/05/16 14:17:22
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

""" openvino 模型
.xml包含神经网络拓扑信息
.bin包含网络的权重和偏置二进制数据

"""

# import module
import os
import numpy as np
import cv2 as cv


import openvino as ov
# from openvino.runtime import Core


# # ie = Core()
# ie = ov.Core()
# # 查看系统可用设备
# devices = ie.available_devices
# for device in devices:
#     device_name = ie.get_property(device,'FULL_DEVICE_NAME')
#     print(f'{device}:{device_name}')


import torchvision
model = torchvision.models.resnet18(weights='DEFAULT')
ov_model = ov.convert_model(model)
compiled_model = ov.compile_model(ov_model, "AUTO")

print(compiled_model)

# model_xml = '../data/weights/b113_yolov8n_openvino_model/b113_yolov8n.xml'

# ie = ov.Core()
# model  = ie.read_model(model=model_xml)
# compiled_model = ie.compile_model(model=model,device_name='CPU')

# # 查看模型信息

# model.inputs

# input_layer = model.input(0)
# input_layer.any_name

# print(f'input precision:{input_layer.element_type}')
# print(f'input shape:{input_layer.shape}')

# model.outputs

# output_layer = model.output(0)
# output_layer.any_name

# print(f'output precision:{output_layer.element_type}')
# print(f'output shape:{output_layer.shape}')


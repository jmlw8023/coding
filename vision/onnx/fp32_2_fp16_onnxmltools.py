#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   fp32_2_fp16_onnxmltools.py
@Time    :   2024/05/24 14:29:48
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''
# link: https://zhuanlan.zhihu.com/p/412528771

# pip install onnxconverter_common
# pip install onnxmltools   # Successfully installed onnxmltools-1.12.0

# import module
import os
import onnx

import numpy as np
from onnxmltools.utils import float16_converter


onnx_path = '../weights/yolov8n.onnx'


onnx_model = onnx.load(onnx_path)
save_folder = os.path.dirname(onnx_path)
print('load model success!')
trans_fp16_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)

onnx.save_model(trans_fp16_model, os.path.join(save_folder, 'yolov8_fp16.onnx'))
print('transform model success!')




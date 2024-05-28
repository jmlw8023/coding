# -*- encoding: utf-8 -*-
'''
@File    :   fuse_conv_bn.py
@Time    :   2023/02/08 16:56:12
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
import os
import onnx
import onnxsim
from onnxmltools.utils import float16_converter
# 



onnx_path = '../weights/yolov8n.onnx'


model = onnx.load(onnx_path)

print(model)



save_folder = os.path.dirname(onnx_path)
print('load model success!')
trans_fp16_model = float16_converter.convert_float_to_float16(model, keep_io_types=True)


onnx.save_model(trans_fp16_model, os.path.join(save_folder, 'yolov8_fp16.onnx'))





















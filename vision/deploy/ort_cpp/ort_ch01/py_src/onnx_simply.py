# import module
import os
import onnx

import numpy as np
from onnxmltools.utils import float16_converter


onnx_path = './yolov5n.onnx'


onnx_model = onnx.load(onnx_path)
save_folder = os.path.dirname(onnx_path)
print('load model success!')
trans_fp16_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)

onnx.save_model(trans_fp16_model, os.path.join(save_folder, 'yolov5n_fp16.onnx'))
print('transform model success!')



# 终端再优化 ： onnxsim yolov5n_fp16.onnx yolov5n_fp16.onnx








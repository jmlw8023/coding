#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   yolo.py
@Time    :   2024/05/24 14:10:15
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''
# link: 
# import module
from ultralytics import YOLO

# ###########################################################################################################
# # 模型结构路径，不同模型通过这个进行指定
# # model_path = r'./ultralytics/cfg/models/v8/yolov8.yaml'  
# model_path = r'./ultralytics/cfg/models/v8/yolov8n-pose.yaml'  
# # model_path = r'./ultralytics/cfg/models/v9/yolov9e.yaml'
# # 更改模型可以在这个地方找，或者自己去定义结构：/root/source/code/ultralytics/ultralytics/cfg/models/v8
# # 模型具体网络结构在：/root/source/code/ultralytics/ultralytics/nn/modules/block.py 或者 conv.py 或者head.py等

# ##########################################################################################
# # 数据集存放位置路径，不改变数据集，这个不用改
# data_path = r'./mydata.yaml'

# # # ############################################ 训练模型 ###################################################
# # # 加载模型
# # model = YOLO("yolov8n.yaml")  # build a new model from scratch
# # # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO(model_path).load('yolov8n-pose.pt')  # build from YAML and transfer weights

# # 模型训练
# # 参数： 数据集路径， 训练迭代次数， 训练尺寸， 保存路径名称（默认runs/detect/里面，找对应name的名称）
# model.train(data=data_path, epochs=100, imgsz=640, name='v8n_pose_mydata')  # train the model

# # 模型验证  name就是在runs/detect/里面，名称为 name名称
# metrics = model.val(name='v8n_pose_mydata_val')  # evaluate model performance on the validation set
# path = model.export(format="onnx", simplify=True)  # export the model to ONNX format


# ############################################ 单独测试 ###################################################

# # 指定pt文件路径
pt_model_path = r'runs/detect/b_v8n/weights/b113_yolov8n.onnx'

# # 加载指定权重模型
model_test = YOLO(pt_model_path)
# # name就是在runs/detect/里面，名称为 name名称， 比如现在指定的名称  my_v8n_val
results_val = model_test.val(name='v8n_val') 
# results_val = model_test.predict(source='D:/source/code/datasets/data_b113/train/images', name='v8n_detect_traindata', save=True) 



# 导出 openvino 模型
# yolo export model=runs/detect/b_v8n/weights/b113_yolov8n.pt format=openvino  simplify=True dynamic=True half=False













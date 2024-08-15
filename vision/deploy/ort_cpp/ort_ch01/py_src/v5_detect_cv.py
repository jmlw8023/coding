# -*- encoding: utf-8 -*-
'''
@File    :   v5_detect.py
@Time    :   2024/06/19 22:22:28
@Author  :   jolly 
@Version :   python3
@Contact :   jmlw8023@163.com
'''

# import packets
import os
import cv2 as cv
# import onnxruntime
# import argparse
import numpy as np


class Det(object):

    def __init__(self, model_path=None, model_shape=(640, 640)) -> None:

        # self.__class_names = ['plane']
        self.__class_names = []
        # self.__class_names = ['plane']
        self._get_classes()
        self.__img_input_shape = model_shape
        self.__score_threshold = 0.2  
        self.__nms_threshold = 0.5
        self.__confidence_threshold = 0.2   

        self.__img_shuffix = ["bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"]
        self.model_path = model_path
        # self.net = cv.dnn.readNet(self.model_path)
        self.net = cv.dnn.readNetFromONNX(self.model_path)
    
    def _get_classes(self, cls_path=r'./data/coco.names'):
        assert os.path.isfile(cls_path), f'{cls_path} is error file!'
        
        self.__class_names.clear()
        with open(cls_path, 'r', encoding='utf-8') as f:
            for name in f.readlines():
                # print(name.strip())
                self.__class_names.append(name.strip())
        
        # print(self.__class_names)

    # 填充
    def _letterbox(self, im, new_shape=(416, 416), color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))    
        dw, dh = (new_shape[1] - new_unpad[0])/2, (new_shape[0] - new_unpad[1])/2  # wh padding 
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        if shape[::-1] != new_unpad:  # resize
            im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
        im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
        return im

    # 缩放尺寸
    def _scale_boxes(self, boxes, shape):
        # Rescale boxes (xyxy) from img_input_shape to shape
        gain = min(self.__img_input_shape[0] / shape[0], self.__img_input_shape[1] / shape[1])  # gain  = old / new
        pad = (self.__img_input_shape[1] - shape[1] * gain) / 2, (self.__img_input_shape[0] - shape[0] * gain) / 2  # wh padding
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes

    # 绘制检测结果
    def _draw(self, image, box_data):
        box_data = self._scale_boxes(box_data, image.shape)
        boxes = box_data[...,:4].astype(np.int32) 
        scores = box_data[...,4]
        classes = box_data[...,5].astype(np.int32)
    
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            print('class: {}, score: {}, coordinate: [{}, {}, {}, {}]'.format(self.__class_names[cl], score, top, left, right, bottom))
            cv.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 1)
            cv.putText(image, '{0} {1:.2f}'.format(self.__class_names[cl], score), (top, left), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # 过滤检测框
    def _filter_box(self, outputs): 
        boxes = []
        scores = []
        class_ids = []

        for i in range(outputs.shape[1]):
            data = outputs[0][i]
            objness = data[4]
            if objness > self.__confidence_threshold:
                score = data[5:] * objness
                _, _, _, max_score_index = cv.minMaxLoc(score)
                max_id = max_score_index[1]
                if score[max_id] > self.__score_threshold:
                    x, y, w, h = data[0].item(), data[1].item(), data[2].item(), data[3].item()
                    boxes.append(np.array([x-w/2, y-h/2, x+w/2, y+h/2]))
                    scores.append(score[max_id]*objness)
                    class_ids.append(max_id)

        indices = cv.dnn.NMSBoxes(boxes, scores, self.__score_threshold, self.__nms_threshold)
        output = []
        for i in indices:
            output.append(np.array([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], class_ids[i]]))
        output = np.array(output)
        return output
    
    # raw格式转为模型需要输入的格式
    def raw_convert_format(self, file_path, size=(512, 640)):

        shape = size[0] * size[1]
        bayer = np.zeros(shape=shape, dtype='uint16')
        with open(file_path, "rb") as f:
            for i in range(0, len(f.read()), 2):
                f.seek(i)
                raw = f.read(2)
                a1 = int((raw[0] / 4) % 64)  # 14位有效位的高八位
                a2 = int(((raw[0] % 4) * 16) + (raw[1] / 16))  # 14位有效位的低六位
                a3 = int(raw[1] % 16)  # 14位有效位的低四位
                value = (a1 << 10) + (a2 << 4) + a3  # 组合成14位值
                bayer[int(i / 2)] = value

        # bayer = bayer.reshape(size)
        image = np.frombuffer(bayer, dtype=np.uint8).reshape(size[0], size[1], -1).astype(np.float32)  
        # img = np.dstack((image[:, :, 1], image[:, :, 1], image[:, :, 1])) 
        img = np.dstack((image[:, :, 1], image[:, :, 1])) 

        return img

    # img_path: 检测图像路径
    # save_dir: 存储文件夹
    # is_use_raw: 是否是raw格式,
    # img_size:   为raw格式时需要传入的图像尺寸 rows，cols 
    # is_save:  是否检测存储
    def detect(self, img_path=None, save_dir='./results/images', is_use_raw=True, img_size=(512, 640), is_save=True):

        if img_path is None:
            raw_img_path = r'./data/image/1/image_0_31.raw' 
        else:
            raw_img_path = img_path
        
        if is_use_raw:
            img_shuffix = 'raw'
        else:
            img_shuffix = self.__img_shuffix 
        
        assert raw_img_path.lower().endswith(tuple(img_shuffix)), 'img format error!'

        if is_use_raw:
            image = self.raw_convert_format(raw_img_path, size=img_size)
            assert (image is not None), 'raw_convert_format error!'
        else:
            image = cv.imread(raw_img_path)
            assert (image is not None), 'image error!'
                
        img_name, shuffix = os.path.splitext(os.path.basename(raw_img_path))

        img_input = self._letterbox(image, self.__img_input_shape)
        # print('---------- img_input.shape = ', img_input.shape)
        blob = cv.dnn.blobFromImage(img_input, 1/255., size=self.__img_input_shape, swapRB=True, crop=False)
        
        self.net.setInput(blob)
        outputs = self.net.forward()
        boxes = self._filter_box(outputs)
        self._draw(image, boxes)
        if is_save:
            os.makedirs(save_dir, exist_ok=True)
            if is_use_raw:
                save_img_path = os.path.join(save_dir, img_name + '_raw.png')
            else:
                save_img_path = os.path.join(save_dir, img_name + '.png')
                
            cv.imwrite(save_img_path, image)
            print(f'save {save_img_path}  successful!')

        # cv.imshow('res', image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

def raw_to_png(raw_file_path, output_png_path=r'./', rows=160, cols=160, channels=16):  
    img = np.fromfile(raw_file_path, dtype='uint16')	# 图像的位深度
    img = img.reshape(rows, cols, -1)
    cv.imwrite(output_png_path, img )	
    
    
if __name__=="__main__":  

    # model_shape=(320, 320)
    model_shape=(640, 640)
    # model_path = r"./yolov5n_fp16.onnx"
    model_path = r"./yolov5n.onnx"
    raw_img_path = r"./data/images/bus.jpg"
    
    # raw_to_png(raw_img_path)

    demo = Det(model_path=model_path, model_shape=model_shape)
    
    # img_path: 检测图像路径
    # save_dir: 存储文件夹
    # is_use_raw: 是否是raw格式,
    # img_size:   为raw格式时需要传入的图像尺寸 rows，cols 
    # is_save:  是否检测存储
    demo.detect(img_path=raw_img_path, is_use_raw=False)
    
    
    
    
    
    

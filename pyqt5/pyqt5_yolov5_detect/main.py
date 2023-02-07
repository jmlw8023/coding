# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/12/26 10:16:23
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os
import sys
import time
import random
import numpy as np
from pathlib import Path

import cv2 as cv
from PIL import Image, ImageDraw, ImageFont

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDesktopWidget, QFileDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOU root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ui.detect_GUI import Ui_MainWindow
from ui.login_GUI import Ui_Login_win
from utils.yolov5_onnx import YOLOV5, filter_box




class Detect(QMainWindow):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.imgsz = 640            # 预测图尺寸大小
        self.conf_thres = 0.25      # NMS置信度
        self.iou_thres = 0.50       # IOU阈值
        self.img_name = None
        self.weight = r'weights/yolov5s.onnx'
        self.classes_path = r'data/coco.names'
        self.model = YOLOV5(self.weight)
      
        # self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # 从模型中获取各类别名称
        self.names = self.get_classes(self.classes_path)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]               # 给每一个类别初始化颜色
        self.show()

        self.initDir()
        self.initUI()
        self.init_signal_solt()

    def initUI(self):

        # self.setGeometry(300, 300, 1200, 800)
        self.ui.text_res.setReadOnly(True)  # 设置文本不可编辑
        # self.center()
        self.setWindowTitle('YOLO 识别检测系统')
 
    # 
    def center(self):
        # 窗口大小
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        # 本窗体运动
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)
    # 
    def initDir(self):
        if not os.path.exists(os.path.join(ROOT, 'data')):
            os.makedirs(os.path.join(ROOT, 'data'))
        if not os.path.exists(os.path.join(ROOT, 'results')):
            os.makedirs(os.path.join(ROOT, 'results'))
        if not os.path.exists(os.path.join(ROOT, 'weights')):
            os.makedirs(os.path.join(ROOT, 'weights'))
    # 
    def get_classes(self, name_file='data/coco.names'):
        with open(name_file, 'r', encoding='utf-8') as f:
            classes = []
            for name in f.readlines():
                name = name.strip()
                classes.append(name)
            return classes
    # 
    def init_signal_solt(self):
        self.ui.btn_img.clicked.connect(self.image_show)
        self.ui.btn_weight.clicked.connect(self.weight_choose)
        # t_begin = time.time()
        self.ui.btn_run.clicked.connect(self.run)
        self.ui.btn_cls.clicked.connect(self.classes_choose)
        self.ui.btn_video.clicked.connect(self.video_choose)
        self.ui.exit_pushButton.clicked.connect(self.close_win)
    
    def close_win(self):
        mgs_box = QMessageBox.question(
            None, '提示信息', '是否真的退出检测系统！',
            QMessageBox.Yes |
            QMessageBox.No 
        )

        if mgs_box == QMessageBox.Yes:
            self.close()
        elif mgs_box == QMessageBox.No:
            pass
    # 
    def image_show(self):
        print('open image ---> ')
        self.img_name = QFileDialog.getOpenFileName(
            self, '选择图片', './images', '*.jpg;;*.jpeg;;*.png;;All Files(*)'
        )
        print('open image ---> ', self.img_name[0])

        if self.img_name[0] is not None and len(self.img_name[0]):
            pixmap = QPixmap(self.img_name[0])
            self.ui.q_pre.setPixmap(pixmap)
            self.ui.q_pre.setScaledContents(True)
        else:
            pass

    # 
    def weight_choose(self):

        self.weight_file = QFileDialog.getOpenFileName(
            self, '选择权重文件', './weights', '*.onnx;;*.pt;;*.pth;;All Files(*)'
        )
        print(self.weight_file)
        if self.weight_file[0] is not None and len(self.weight_file[0]):
            print('open weight ---> ', self.weight_file[0])
            self.weight = self.weight_file[0]
            self.model = YOLOV5(self.weight)
        else:
            pass
    # 
    def classes_choose(self):
        self.classes_file = QFileDialog.getOpenFileName(
            self, '选择类别文件', './data', '*.names;;*.txt;;All Files(*)'
        )

        if self.classes_file[0] is not None and len(self.classes_file[0]):
            print('open classes file ---> ', self.classes_file[0])
            self.classes_path = self.classes_file[0]
            self.names = self.get_classes(self.classes_path)
            print(self.names)
        else:
            pass
    # 
    def video_choose(self):
        self.video_file = QFileDialog.getOpenFileName(
            self, '选择视频文件', './images', '*.mp4;;*.flv;;*.avi;;All Files(*)'
        )

        if self.video_file[0] is not None and len(self.video_file[0]):
            print('open classes file ---> ', self.video_file[0])
            self.video_path = self.video_file[0]
            print(self.video_file)
        else:
            pass
    
    def msg_run(self):
        # msg = QMessageBox.warning(
        #     self, 
        #     'Unselected image',
        #     'please choose image!!!',
        #     QMessageBox.Yes | QMessageBox.No,
        #     QMessageBox.Yes
        # )
        msg_box = QMessageBox.warning(self, 'Unselected image', 'please choose image!!!')
        print(msg_box)
        # msg_box.exec_()

    def run(self):
        # try:
        #     import torch
        #     assert hasattr(torch, '__version__')
        # except (ImportError, AssertionError):
        #     # torch = None
        #     pass
        # self.result_txt = '<h3>检测结果: </h3>\n <h4>类别 | 分值</h4>\n' 

        self.t_start = time.time()
        if self.img_name is not None :
            image = cv.imread(self.img_name[0])
            output, img  = self.model.inference(image)
            self.t_infer = time.time()
            outbox = filter_box(output, self.conf_thres, self.iou_thres)
            self.t_nms = time.time()
            if outbox is not None and len(outbox):
                self.result_txt = '检测结果:{}个目标 \n 类别 | 分值\n' .format(outbox.shape[0])
                boxes = outbox[...,:4].astype(np.int32)     #取整方便画框
                scores = outbox[...,4]
                classes = outbox[...,5].astype(np.int32)    #下标取整            
                line_thickness = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

                for box, score, cls in zip(boxes, scores, classes):
                    top, left, right, bottom = box
                    color = [random.randint(0, 255) for _ in range(3)]
                    # print(color, 'color')
                    # img = np.ascontiguousarray(img)
                    # cv.rectangle(img, (top, left), (right, bottom), self.colors, -1, cv.LINE_AA)    # filled
                    cv.rectangle(img, (top, left), (right, bottom), color, thickness=line_thickness, lineType=cv.LINE_AA)    # filled
        
        
                if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
                    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img)    # 创建绘制图像   

                    print('The categories detected are as follows :')
                    for box, score, cls in zip(boxes, scores, classes):
                        top, left, right, bottom = box
                        print('-' * 15)
                        # print(box)
                        # print(cls)
                        # print('=' * 15)
                        txt = '{0} {1:.2f}'.format(self.names[int(cls)], score)
                        print('====>  ', txt, '\t ', box)
                        # self.result_txt.append(txt)
                        self.result_txt += txt + '\n'
                        # fontstype = ImageFont.truetype("data/myfont.ttf", 20, encoding="utf-8")
                        fontstype = ImageFont.truetype(font='data/Arial.ttf', size=20, encoding="utf-8")
                        # draw.text((top, left-25), txt, (0, 255, 0), font=fontstype)  # 绘制文本
                        draw.text((top+25, left-25), txt, tuple(color), font=fontstype)  # 绘制文本
                    print('-------------------Detected over!!---------------------')
                else:
                    assert 'error'     
                
                img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
                cv.imwrite('results/{}'.format(os.path.basename(self.img_name[0])), img)
            print(f'save to results/{os.path.basename(self.img_name[0])}  success!!')
            self.result = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
            # qt_img = QImage(result.data, result.shape[1], result.shape[0], QImage.Format_RGB32)
            self.qt_img = QImage(self.result.data, self.result.shape[1], self.result.shape[0], QImage.Format_RGB32)

            self.ui.q_res.setPixmap(QPixmap.fromImage(self.qt_img))
            self.ui.q_res.setScaledContents(True)
            # print(self.result_txt)
            self.ui.text_res.setText(self.result_txt)
        else:
            # self.ui.btn_run.clicked.connect(self.msg_run)
            msg_box = QMessageBox.warning(self, 'Unselected image', 'please choose image!!!')
            # print(msg_box)
        
        self.t_end = time.time()


        time_context = "推理时间:{:.2f}ms \nNMS时间:{:.2f}ms \n总运行时间:{:.2f}ms.".format((self.t_infer-self.t_start)*1000, (self.t_nms-self.t_infer)*1000, (self.t_end-self.t_start)*1000)
        print(time_context)
        self.ui.time_label.setText(time_context)









class Login_win(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.ui = Ui_Login_win()
        self.ui.setupUi(self)

        self.initUI()
        self.init_signal_solt()


    def initUI(self):
        self.set_button_title()

    def init_signal_solt(self):
       
        self.ui.btn_exit.clicked.connect(self.close_win)
        self.ui.btn_login.clicked.connect(self.show_main_win)


    def set_button_title(self):
        img_name = r'./images/2.jpeg'
        img_name_path = r'./images/pigeon/09171200302.jpg'
        # 获取窗口大小
        # screen = QDesktopWidget().screenGeometry()
        # size = self.geometry()
        # 本窗体运动
        # self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

        # 标题
        self.ui.title_label.setText('基于YOLO的智能检测系统')
        # 文本居中
        self.ui.title_label.setAlignment(Qt.AlignCenter)
        # 文本中使用图片背景
        # self.ui.title_label.setStyleSheet("border-image:url({})".format(img_name))
        # 文本背景颜色
        self.ui.title_label.setStyleSheet('background-color: rgb(25, 151, 30)')
        self.ui.btn_exit.setStyleSheet('background-color: rgb(25, 151, 30)')
        self.ui.btn_login.setStyleSheet('background-color: rgb(25, 151, 30)')
        # 
        self.ui.bg_img_label.setScaledContents(True)
        # 加载图片
        img = QPixmap(img_name).scaled(self.ui.bg_img_label.width(), self.ui.bg_img_label.height())
        self.ui.bg_img_label.setPixmap(img)


        
    
    def show_main_win(self):
        self.m_win = Detect()
        self.m_win.show()
    
    def close_win(self):
        mgs_box = QMessageBox.question(
            None, '提示信息', '是否真的退出系统！',
            QMessageBox.Yes |
            QMessageBox.No 
        )

        if mgs_box == QMessageBox.Yes:
            self.close()
        elif mgs_box == QMessageBox.No:
            pass

    def info_print(self):
        print('打印信息~~~~~!!!')
        







if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    # det = Detect()
    # det.show()
    # main_win = Detect()
    login_win = Login_win()

    login_win.show()


    sys.exit(app.exec())


    



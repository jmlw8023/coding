
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2024/04/12 14:33:29
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module
import os
import sys  

import cv2 as cv
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox)
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap, QFont, QResizeEvent

from ui.ui_home import Ui_MainWindow



# 将 OpenCV 的 Mat 转换成 QImage
def cv_mat_to_qimage(mat):
    
    height, width = mat.shape[:2]  # 只取前两个值，适应灰度图或彩色图
    if len(mat.shape) == 2:     # 灰度图
        bytes_per_line = width
        format = QImage.Format.Format_Grayscale8
        
    elif len(mat.shape) == 3:   # 彩色图
        bytes_per_line = mat.strides[0]
        format = QImage.Format.Format_RGB888
    
    elif len(mat.shape) == 4:   # 彩色图
        bytes_per_line = mat.strides[0]
        format = QImage.Format.Format_RGB32
        
    qimg =  QImage(mat.data, width, height, bytes_per_line, format)
    qimg = qimg.rgbSwapped()  # 对于OpenCV的BGR格式，需要转换为Qt的RGB格式

    return qimg

class Home_win(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.initUI()
        self.init_signal_solt()
        
        self.im = None
        self.img_show = False
        self.img_path = None

    # 初始化页面
    def initUI(self):
        self.setWindowTitle('系统')
        self.format_lists = ['.jpg', '.png', '.jpeg', '.BMP ', '.WebP']
        
        self.ui.label_v_src.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_v_dst.setAlignment(Qt.AlignmentFlag.AlignCenter)


    # 初始化 信号与槽
    def init_signal_solt(self):

        self.ui.btn_open_img.clicked.connect(self.open_img_path)       
        # self.ui.btn_v_play.clicked.connect(self.detect_img)       
        self.ui.btn_v_clear.clicked.connect(self.clear_page)       
        
        self.ui.spinBox.valueChanged.connect(self.ui.horizontalSlider.setValue)     # spinbox信号发送给滑竿

        self.ui.horizontalSlider.valueChanged.connect(self.threshold_set_img)
        self.ui.horizontalSlider.valueChanged.connect(self.ui.spinBox.setValue)
        

    # 打开图像
    def open_img_path(self):
        img_path = QFileDialog.getOpenFileName(
            self, '选择图片', '.', 'All Files(*);;*.png;;*.jpg'
        )
        # print('open image ---> ', self.img_path[0])
        self.img_path = img_path[0]

        if self.img_path is not None and len(self.img_path):    
            # 如果选择了文件夹，则将其路径设置到lineEdit中
     
            if self.img_path.endswith(tuple(self.format_lists)):
                self.im = cv.imread(self.img_path)
                if self.im is not None:
                    img = cv.cvtColor(self.im, cv.COLOR_BGR2RGB)
                    pixmap = QPixmap.fromImage(cv_mat_to_qimage(img))
                    # self.ui.label_v_src.setPixmap(pixmap.scaled(self.ui.label_v_src.size()))
                    self.ui.label_v_src.setPixmap(pixmap.scaled(self.width()//2, int(self.height()/1.3)))
                    self.img_show = True
                    
                    value_spin = self.ui.spinBox.value()
                    print(value_spin)
                    self.threshold_set_img(value_spin)
                    
                else:
                    QMessageBox.information(self, '信息', '载入的图像有问题!')
                    
            else:
                QMessageBox.information(self, '信息', '请先载入图像文件!')

        else:
            self.ui.label_v_src.clear()
            self.ui.label_v_dst.clear()
            self.img_path = None
            QMessageBox.information(self, '错误', '请先载入正确路径!')
    
    def threshold_set_img(self, value_spin):
       
        if self.img_show:
            
            if self.im is not None: 
                gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)

                # blur = cv.medianBlur(gray,7)
                
                # value_spin = self.ui.spinBox.value()
                print('spinBox value = {}'.format(value_spin))

                _, thres = cv.threshold(gray, value_spin, 255, cv.THRESH_BINARY_INV)
                # cv.imshow("thresh", cv.resize(thres, (720, 640)))
                print('---------------thres shape = ', thres.shape)
                pixmap = QPixmap.fromImage(cv_mat_to_qimage(thres))
                self.ui.label_v_dst.setPixmap(pixmap.scaled(self.width()//2, int(self.height()/1.3)))

            else:
                QMessageBox.information(self, '信息', '载入的图像有问题!')

        else:
            QMessageBox.information(self, '信息', '请先载入图像文件!')
    # 清理页面
    def clear_page(self):
        if self.img_path:
            self.ui.label_v_dst.clear()
            self.ui.label_v_src.clear()
            
            self.ui.horizontalSlider.setValue(20)
 
            self.img_path = None
            self.img_show = False


    # 窗口大小发生变化
    def resizeEvent(self, event):
        # return super().resizeEvent(a0)
        # new_size = event.size()
        # print(f"Window resized to: {new_size.width()}x{new_size.height()}")

        if self.im is not None:
            img = cv.cvtColor(self.im, cv.COLOR_BGR2RGB)
            pixmap = QPixmap.fromImage(cv_mat_to_qimage(img))
            # self.ui.label_v_src.setPixmap(pixmap.scaled(self.ui.label_v_src.size()))
            self.ui.label_v_src.setPixmap(pixmap.scaled(self.width()//2, int(self.height()/1.3)))
            self.img_show = True
            
            self.threshold_set_img(self.ui.spinBox.value())

if __name__ == '__main__':
    

    app = QApplication(sys.argv)
    # screen = app.primaryScreen()
    # screen_size = screen.size()
    home_win = Home_win()
    home_win.show()

    sys.exit(app.exec())










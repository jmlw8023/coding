
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
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
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
    
    data_sig = pyqtSignal(list) #自定义信号
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.initUI()
        self.init_signal_solt()
        
        self.im = None
        self.img_show = False
        self.img_path = None
        self.yolo_img_path = None
        self.yolo_weight_path = None
        self.model = None
        self.yolo_image = None
        
        self.CLASSES = ['addB', 'addS', 'pitN', 'pitF']
        self.random_colors = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.video_format_lists = ['.mp4', '.avi', '.mkv', '.mpeg', '.mov']
        self.img_format_lists = ['.jpg', '.png', '.bpm', '.jpeg', '.webp']
        self.weights_format_lists = ['.pt', '.onnx', '.torchscript', '.vino']

    # 初始化页面
    def initUI(self):
        self.setWindowTitle('系统')
        ##################################################################################################
        # self.ui.label_yolo_src.setScaledContents(True)
        
        ##################################################################################################
        self.ui.label_v_src.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_v_dst.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.ui.btn_home_img_process.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.btn_home_count.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.btn_home_yolo.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.btn_home_other.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
        ##################################################################################################
        ##################################################################################################
        
        # self.ui.btn_side.setText('<<')


    # 初始化 信号与槽
    def init_signal_solt(self):

        ##################################################################################################
        self.ui.btn_yolo_open_file.clicked.connect(self.open_yolo_file)
        self.ui.btn_yolo_open_weight.clicked.connect(self.open_yolo_weight)
        self.ui.btn_yolo_detect.clicked.connect(self.yolo_detect)
        self.ui.btn_yolo_save.clicked.connect(self.save_yolo_detect)
        self.ui.btn_yolo_save_file.clicked.connect(self.open_save_yolo_folder)
        
        ##################################################################################################
        self.ui.btn_open_img.clicked.connect(self.open_img_path)       
        # self.ui.btn_v_play.clicked.connect(self.detect_img)       
        self.ui.btn_v_clear.clicked.connect(self.clear_page)       
        
        self.ui.spinBox.valueChanged.connect(self.ui.horizontalSlider.setValue)     # spinbox信号发送给滑竿

        self.ui.horizontalSlider.valueChanged.connect(self.threshold_set_img)
        self.ui.horizontalSlider.valueChanged.connect(self.ui.spinBox.setValue)
        
    # 选择 权重文件
    def open_yolo_weight(self):
        
        file_path = QFileDialog.getOpenFileName(self, '选择图片', './', 'All Files(*);;*.onnx;;*.pt;;')
        
        yolo_weight_path = file_path[0]
        if yolo_weight_path is not None and len(yolo_weight_path) > 0:
            
            self.ui.lineEdit_yolo_weight_path.clear()
            if yolo_weight_path.lower().endswith(tuple(self.weights_format_lists)):
                self.yolo_weight_path = yolo_weight_path
                print('open weight ---> ', yolo_weight_path)
                self.ui.lineEdit_yolo_weight_path.setText(self.yolo_weight_path)
                
                if yolo_weight_path.lower().endswith('.onnx'):
                    self.model: cv.dnn.Net = cv.dnn.readNetFromONNX(self.yolo_weight_path)
            
            else:
                self.ui.lineEdit_yolo_weight_path.clear()
                QMessageBox.information(self, '信息', '请先载入权重文件!')
                
        else:
            self.ui.lineEdit_video_path.clear()
            QMessageBox.information(self, '错误', '请先载入正确路径!')
            
            
    # 选择图像文件
    def open_yolo_file(self):
        
        file_path = QFileDialog.getOpenFileName(self, '选择图片', './', 'All Files(*);;*.jpg;;*.png;;')
        yolo_img_path = file_path[0]
        if yolo_img_path is not None and len(yolo_img_path) > 0:
            
            self.ui.lineEdit_yolo_file_path.clear()
            if yolo_img_path.lower().endswith(tuple(self.img_format_lists)):
                self.yolo_img_path = yolo_img_path
                print('open image ---> ', yolo_img_path)
                self.ui.lineEdit_yolo_file_path.setText(self.yolo_img_path)
                
                self.ui.label_yolo_src.setPixmap(QPixmap(yolo_img_path).scaled(self.width()//2, int(self.height()/1.3)))
                
                self.yolo_image: np.ndarray = cv.imread(self.yolo_img_path)
            
            else:
                self.ui.lineEdit_yolo_file_path.clear()
                QMessageBox.information(self, '信息', '请先载入图像文件!')
                
        else:
            self.ui.lineEdit_video_path.clear()
            QMessageBox.information(self, '错误', '请先载入正确路径!')
            
            
    def yolo_detect(self):
        
        if self.model is not None and self.yolo_image is not None:
            self.yolo_im = self.yolo_image.copy()
            h, w = self.yolo_image.shape[:2]

            length = max((h, w))

            image = np.zeros((length, length, 3), np.uint8)
            image[0:h, 0:w] = self.yolo_image

            scale = length / 640

            # Preprocess the image and prepare blob for model
            blob = cv.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
            self.model.setInput(blob)

            outputs = self.model.forward()

            # Prepare output array
            outputs = np.array([cv.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []

            # Iterate through output to collect bounding boxes, confidence scores, and class IDs
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv.minMaxLoc(classes_scores)
                if maxScore >= 0.25:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2],
                        outputs[0][i][3],
                    ]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)           
                

            result_boxes = cv.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

            detections = []

            # Iterate through NMS results to draw bounding boxes and labels
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    "class_id": class_ids[index],
                    "class_name": self.CLASSES[class_ids[index]],
                    "confidence": scores[index],
                    "box": box,
                    "scale": scale,
                }
                detections.append(detection)
                self.draw_bounding_box(
                    self.yolo_im,
                    class_ids[index],
                    scores[index],
                    round(box[0] * scale),
                    round(box[1] * scale),
                    round((box[0] + box[2]) * scale),
                    round((box[1] + box[3]) * scale),
                )

            
            pixmap = QPixmap.fromImage(cv_mat_to_qimage(self.yolo_im))
            # self.ui.label_v_src.setPixmap(pixmap.scaled(self.ui.label_v_src.size()))
            self.ui.label_yolo_dst.setPixmap(pixmap.scaled(self.width()//2, int(self.height()/1.3)))

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        """
        Draws bounding boxes on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the bounding box on.
            class_id (int): Class ID of the detected object.
            confidence (float): Confidence score of the detected object.
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        """
        label = f"{self.CLASSES[class_id]} ({confidence:.2f})"
        color = self.random_colors[class_id]
        cv.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv.putText(img, label, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1.8, color, 2)
        # cv.putText(img, label, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    def save_yolo_detect(self):
        if self.yolo_img_path is not None:
            if self.yolo_im is not None:
                # save_folder = os.path.dirname(self.yolo_im)
                d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
                save_folder = os.path.join(d, 'results')
                os.makedirs(save_folder, exist_ok=True)
                img_name, shuffix = os.path.splitext(os.path.basename(self.yolo_img_path))
                
                if self.yolo_image is not None:
                    cv.imwrite('{}'.format(os.path.join(save_folder, img_name + '.png')), self.yolo_image)
                    QMessageBox.information(self, '信息', '文件存储成功!')
                
    def open_save_yolo_folder(self):
        # if self.yolo_img_path is not None:
            # save_path = os.path.join(os.path.dirname(self.yolo_img_path), 'results')
            # 获取当前py文件路径
            d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
            save_path = os.path.join(d, 'results')
            os.makedirs(save_path, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(save_path))
            # QDesktopServices.openUrl(QUrl.fromLocalFile(self.yolo_img_path))
        

    # 打开图像
    def open_img_path(self):
        img_path = QFileDialog.getOpenFileName(
            self, '选择图片', '../results', 'All Files(*);;*.png;;*.jpg'
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
                # _, thres = cv.threshold(gray, value_spin, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
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










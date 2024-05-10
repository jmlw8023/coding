#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main_pyqt.py
@Time    :   2024/04/30 09:06:50
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module
import os
import sys  
import glob 
import time
import random
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import timedelta, datetime
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import serial
# from serial.tools import list_ports
from PyQt6.QtCore import Qt, QTimer, QPointF, QDateTime, QTime
from PyQt6.QtSerialPort import QSerialPortInfo, QSerialPort
from PyQt6.QtWidgets import (QApplication, QWidget, QFrame, QMainWindow, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene)
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap, QFont, QPainter, QPen, QBrush 
from qt_material import apply_stylesheet

import pyqtgraph as pg

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import ui.res
# import PySide6

# 图像背景样式：https://blog.csdn.net/leidawangzi/article/details/110942910

# pyrcc5 -o res.py mdata.qrc 

from ui.ui_main import Ui_Form

# # 获取当前py文件路径
# d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# 读取文本
# text = open(os.path.join(d, 'constitution.txt')).read()


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


class TimeAxisItem(pg.AxisItem):
    """Internal timestamp for x-axis"""
    def __init__(self, *args, **kwargs):
        super(TimeAxisItem, self).__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        """Function overloading the weak default version to provide timestamp"""

        return [time.strftime("%H:%M:%S", time.localtime(value/1000)) for value in values]
        
class Home_win(QFrame):
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.initUI()

        self.init_signal_solt()
        
        # 初始化串口
        # self.port = None
        self.ser = serial.Serial()
        
        # 设置定时器  
        self.timer = QTimer(self)  
        self.timer.timeout.connect(self.update_data)  
        self.timer.start(1000)  # 每秒更新一次 
        
        # 初始化数据  
        self.x_data = []
        self.x_data_second = []
        self.y_data = []  
        self.filtered_y_data = []  
        
        self.threshold = 33
        self.t_base = time.time()*1000

        # self.plot_widget = pg.PlotWidget()
        self.plot_widget_temp_src = pg.GraphicsLayoutWidget()
        self.ui.graph_layout_temp_src.addWidget(self.plot_widget_temp_src)
        self.plot_widget_temp_dst = pg.GraphicsLayoutWidget()
        self.ui.graph_layout_temp_dst.addWidget(self.plot_widget_temp_dst)

        self.plot_widget_temp_src.setBackground('w') # 白色背景
        # self.plot_widget_temp_dst.setBackground('w') # 白色背景
        self.flag_temp_ols = True
        self.flag_temp_ms = True
        # self.ui.btn_temp_ms.setChecked(True)
        # self.ui.btn_temp_ols.setChecked(True)
        
        # self.temp_plot_item_src = self.plot_widget_temp_src.addPlot(title="实时曲线", axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.temp_plot_item_src = self.plot_widget_temp_src.addPlot(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.temp_plot_item_dst = self.plot_widget_temp_dst.addPlot(axisItems={'bottom': TimeAxisItem(orientation='bottom')})

        self.temp_src = self.temp_plot_item_src.plot(pen='g', symbol='o', name='原始数据')
        self.temp_src_ms = self.temp_plot_item_src.plot(pen='r', symbol='o', name='方差过滤')
        self.temp_dst = self.temp_plot_item_dst.plot(pen='g', symbol='o', name='原始数据')
        self.temp_dst_ols = self.temp_plot_item_dst.plot(pen='b', symbol='o', name='最小二乘法')

        # 创建图例并添加到图形布局中
        # legend = pg.LegendItem(offset=(100, 10))  # 设置图例偏移量
        legend_src = pg.LegendItem(offset=(-10, -10))  # 设置图例偏移量
        legend_dst = pg.LegendItem(offset=(-10, -10))  
        legend_src.setParentItem(self.temp_plot_item_src.graphicsItem())
        legend_dst.setParentItem(self.temp_plot_item_dst.graphicsItem())
        
        # 添加曲线到图例
        legend_src.addItem(self.temp_src, self.temp_src.name())
        legend_src.addItem(self.temp_src_ms, self.temp_src_ms.name())
        legend_dst.addItem(self.temp_dst, self.temp_dst.name())  
        legend_dst.addItem(self.temp_dst_ols, self.temp_dst_ols.name())  
        
        # # 初始化 DateAxisItem  
        # self.time_axis = pg.DateAxisItem(orientation='bottom')  
        # self.plot_item.setAxisItems({'bottom': self.time_axis})  
        # # 添加曲线  
        # self.curve = pg.PlotCurveItem()  
        # self.plot_item.addItem(self.curve)  

        # self.plotDataItem = self.plotWidget.plot([], [], pen='r')
   
        
        # self.plotWidget.setLabel('left', '度', units='Value')
        # self.plotWidget.setLabel('bottom', 'Time', units='HH:MM:SS')
        # self.plotWidget.showGrid(x=True, y=True)
        # self.plotWidget.enableAutoRange('xy', True)


        # self.p1, self.p2 = self.set_graph_ui()  # 设置绘图窗口
        

    def set_graph_ui(self):

        pg.setConfigOptions(antialias=True)  # pg全局变量设置函数，antialias=True开启曲线抗锯齿

        win = pg.GraphicsLayoutWidget()  # 创建pg layout，可实现数据界面布局自动管理

        # pg绘图窗口可以作为一个widget添加到GUI中的graph_layout，当然也可以添加到Qt其他所有的容器中
        self.ui.graph_layout_temp_src.addWidget(win)

        p1 = win.addPlot(title="原始曲线")  # 添加第一个绘图窗口
        p1.setLabel('left', text='度', color='#ffffff')  # y轴设置函数
        p1.showGrid(x=True, y=True)  # 栅格设置函数
        p1.setLogMode(x=False, y=False)  # False代表线性坐标轴，True代表对数坐标轴
        p1.setLabel('bottom', text='time', units='s')  # x轴设置函数
        # p1.addLegend()  # 可选择是否添加legend

        win.nextRow()  # layout换行，采用垂直排列，不添加此行则默认水平排列
        p2 = win.addPlot(title="处理后曲线")
        p2.setLabel('left', text='度', color='#ffffff')
        p2.showGrid(x=True, y=True)
        p2.setLogMode(x=False, y=False)
        p2.setLabel('bottom', text='time', units='s')
        # p2.addLegend()

        return p1, p2

    # 初始化页面
    def initUI(self):

        self.setWindowTitle('净化空调机组上位机')
        self.format_lists = ['.jpg', '.png', '.jpeg', '.bmp', '.WebP']

        # self.ui.stackedWidget.currentIndex(0)
        self.ui.btn_temp.clicked.connect(self.solt_temp)
        self.ui.btn_humidity.clicked.connect(self.solt_humidity)
        self.ui.btn_wind_speed.clicked.connect(self.solt_wind_speed)
        self.ui.btn_cleanliness.clicked.connect(self.solt_cleanliness)


        self.ui.btn_home.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        # self.ui.btn_img.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))

        # self.ui.comboBox_port.currentIndexChanged.connect(self.update_port_names)
       
        img_bg_path = r'./ui/background.jpg'
        self.bg_img = cv.imread(img_bg_path)
        # 在显示窗口之前，最大化窗口  
        self.showMaximized()  

        w, h = self.width(), self.height()
        img_h, img_w  = self.bg_img.shape[:2]

        y = self.ui.stackedWidget.currentWidget().height()
        x = self.ui.stackedWidget.currentWidget().width()
        # print(img_w, img_h)
        # print(x, y)
        # print(w, h)

        x_radio = w / img_w
        y_radio = h / img_h
        # x_radio = img_w / x
        # y_radio = img_h / y

        # print('x_radio = ', x_radio, ' y_radio = ', y_radio)

        # # cv.rectangle(self.bg_img)
        # # # self.ui.btn_cleanliness.move(10, 92)
        # self.ui.btn_cleanliness.move(205*x_radio, 82*y_radio)
        # self.ui.btn_temp.move(585*x_radio, 110*y_radio)
        # self.ui.btn_humidity.move(532*x_radio, 82*y_radio)
        # self.ui.btn_wind_speed.move(660*x_radio, 88*y_radio)
        # # self.ui.btn_temp.move(568*x_radio, 120*y_radio)
        # # self.ui.btn_humidity.move(520*x_radio, 88*y_radio)
        # # self.ui.btn_wind_speed.move(650*x_radio, 92*y_radio)

        self.update_port_names()
        


    # 初始化 信号与槽
    def init_signal_solt(self):
        pass
        # self.ui.btn_serial_open.clicked.connect(self.port_open)       
        self.ui.btn_temp_ms.clicked.connect(self.temp_meam_squre)       
        self.ui.btn_temp_ols.clicked.connect(self.temp_ols)       
           
    
    
    def temp_meam_squre(self, flag):
        
       if not flag:
           self.flag_temp_ms = flag
           self.ui.btn_temp_ms.setStyleSheet("QPushButton{background-color:red;border-radius:5px;}")
       else:
           self.flag_temp_ms = flag
           self.ui.btn_temp_ms.setStyleSheet("QPushButton{background-color:rgb(255, 255, 255);border-radius:5px;}")
        
    def temp_ols(self, flag):
        
       if not flag:
           self.flag_temp_ols = flag
           self.ui.btn_temp_ols.setStyleSheet("QPushButton{background-color:red;border-radius:5px;}")
       else:
           self.flag_temp_ols = flag
           self.ui.btn_temp_ols.setStyleSheet("QPushButton{background-color:rgb(255, 255, 255);border-radius:5px;}")
        
    
  
    def update_data(self):  
        # 生成随机温度值  
        temp = random.randint(20, 31)  
        if len(self.y_data) % 10 == 0:
            temp = random.randint(28, 45) 
            
        self.y_data.append(temp)  
        t_now = time.time()
        self.x_data.append(int(t_now*1000))
        self.x_data_second.append(int(self.x_data[-1] - self.t_base))
        

        # 如果数据点过多，删除旧的数据点以保持图表清晰  
        if len(self.x_data) > 100:  
            self.x_data.pop(0)  
            self.y_data.pop(0) 
            
        # 原始数据线
        self.temp_src.setData(np.array(self.x_data), np.array(self.y_data))
        
        if self.flag_temp_ms:
            
            print(len(self.x_data))
            print(len(self.y_data))
            print('-'*30)
            # 方差检测和过滤
            if len(self.y_data) > 5:  # 至少需要5个数据点来计算方差                
                # 过滤异常值
                filtered_data = self.filter_outliers_by_variance(self.y_data)
                print((np.array(self.x_data)).shape)
                print((np.array(self.y_data)).shape)
                print((np.array(filtered_data)).shape)
                print('#'*30)
            #     filtered_data = self.filter_outliers_by_variance(self.y_data)
                self.temp_src_ms.setData(np.array(self.x_data), np.array(filtered_data))
            else:
                 # 数据点不足
                self.temp_src_ms.setData(np.array(self.x_data), np.array(self.y_data))

        # 最小二乘法
        if self.flag_temp_ols:
            slope, intercept = np.polyfit(self.x_data_second, np.array(self.y_data), 1)
            fitted_temps = slope * np.array(self.x_data_second) + intercept
            self.temp_dst.setData(np.array(self.x_data), np.array(self.y_data))
            self.temp_dst_ols.setData(np.array(self.x_data), np.array(fitted_temps))
        else:
            self.temp_dst.setData(np.array([]), np.array([]))
            self.temp_dst_ols.setData(np.array([]), np.array([]))
  
  
  
        # # current_time = QDateTime.currentDateTime().toMSecsSinceEpoch() / 1000  # Get current time in seconds
        # # self.x_data.append(current_time)  
        
        # now = QDateTime.currentDateTime()  
        # current_t = now.toMSecsSinceEpoch() / 1000.0  
        # self.x_data.append(current_t)
        # current_t =  datetime.now().strftime('%H:%M:%S')   # strftime('%H:%M:%S.%f')
        # # current_t = QTime.currentTime().hour() * 60 + QTime.currentTime().minute() * 60 + QTime.currentTime().second()
        # print(current_t)
        # # current_t = QDateTime.currentDateTime().toString("hh:mm:ss")
        # # self.x_data.append((datetime.now() - datetime(1970, 1, 1)).total_seconds())

        # current_time = time.strftime('%H:%M:%S', time.localtime())
        # new_y_value = random.randint(0, 100)
        

        # # 更新曲线数据，并仅重绘改变的部分  
        # self.curve.setData(x=self.xdata, y=self.ydata)  
  
        # 如果数据改变很大，可能需要强制重绘整个图表  
        # self.plot_widget.autoRange()  
        # x_values.append(current_time)
        # y_values.append(new_y_value)
        
        # self.plotDataItem.setData(x_values, y_values)
        # self.plotWidget.setTitle(f"Real-time Data ({current_time})")

        # t = np.linspace(0, 20, 200)
        # y_sin = np.sin(t)
        # y_cos = np.cos(t)
        # self.p1.plot(t, y_sin, pen='g', name='sin(x)', clear=True)
        # self.p2.plot(t, y_cos, pen='g', name='con(x)', clear=True)


    # 使用最小二乘法进行线性拟合
    def linear_least_squares(self, x, y):
        m, b = np.polyfit(x, y, 1)
        return m, b

    def filter_outliers_by_variance(self, data, threshold=2, fill_method='mean'):
        """
        根据方差过滤异常值。
        parm:
        data -> 一个包含数值的列表或数组。
        threshold -> 异常值判断的标准,基于标准差的倍数. 默认为2(即超过均值加减2倍标准差的数据被视为异常)
        return:
        filtered_data -> 过滤异常值后的数据。
        """
        # 计算平均值和标准差
        mean = np.mean(data)
        std_dev = np.std(data)
        
        # 正常值范围
        lower_bound = mean - threshold * std_dev
        upper_bound = mean + threshold * std_dev
        # # 过滤异常值
        # filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
        # 异常值
        is_outlier = np.logical_or(data < lower_bound, data > upper_bound)
        # 选择填补策略
        if fill_method == 'mean':
            fill_value = mean
        elif fill_method == 'median':
            fill_value = np.median(data)
        else:
            raise ValueError("Unsupported fill method. Choose from 'mean' or 'median'.")
        
        # 填补异常值
        filled_data = np.where(is_outlier, fill_value, data)
        
        return filled_data

    def draw_curve(self, scene, x_data, y_data, color):  
        # 清除之前的曲线  
        # scene.clear()  
  
        # # 绘制新的曲线  
        # pen = QPen(Qt.GlobalColor.black, 2)  
        # brush = QBrush(Qt.GlobalColor(getattr(Qt, color.upper())))  
  
        # points = [QPointF(x, y) for x, y in zip(x_data, y_data)]  
        # poly = QPolygonF(points)  
  
        # item = scene.addPolygon(poly, pen, brush)  
        # item.setPos(-x_data[0], 0)  # 调整位置以匹配 x 轴  
  
        # 根据需要添加最小二乘法拟合（这里省略了详细实现）  

    
        pass
    
    
    
    
    
      # 打开串口
    def port_open(self):
        
        self.ser.port        = self.ui.comboBox_port.currentText()      # 串口号
        self.ser.baudrate    = int(self.ui.comboBox_baudrate.currentText()) # 波特率

        flag_data = int(self.ui.comboBox_databits.currentText())  # 数据位
        if flag_data == 5:
            self.ser.bytesize = serial.FIVEBITS
        elif flag_data == 6:
            self.ser.bytesize = serial.SIXBITS
        elif flag_data == 7:
            self.ser.bytesize = serial.SEVENBITS
        else:
            self.ser.bytesize = serial.EIGHTBITS

        flag_data = self.ui.comboBox_parity.currentText()  # 校验位
        if flag_data == 'None':
            self.ser.parity = serial.PARITY_NONE
        elif flag_data == 'OddParity':
            self.ser.parity = serial.PARITY_ODD
        elif flag_data == 'EvenParity':
            self.ser.parity = serial.PARITY_EVEN
        elif flag_data == 'SpaceParity':
            self.ser.parity = serial.PARITY_SPACE           
        else:
            self.ser.parity = serial.PARITY_MARK

        flag_data = int(self.ui.comboBox_stopbits.currentText()) # 停止位
        if flag_data == 1:
            self.ser.stopbits = serial.STOPBITS_ONE
        elif flag_data == 2:
            self.ser.stopbits = serial.STOPBITS_TWO
        else:
            self.ser.stopbits = serial.STOPBITS_ONE_POINT_FIVE

        # # flag_data = self.ui.comboBox_flow.currentText()  # 流控
        # if flag_data == "No Ctrl Flow":
        #     self.ser.xonxoff = False  #软件流控
        #     self.ser.dsrdtr  = False  #硬件流控 DTR
        #     self.ser.rtscts  = False  #硬件流控 RTS
        # elif flag_data == "SW Ctrl Flow":
        #     self.ser.xonxoff = True  #软件流控
        # else:         
        #     if self.Checkbox3.isChecked():
        #         self.ser.dsrdtr = True  #硬件流控 DTR
        #     if self.Checkbox4.isChecked():
        #         self.ser.rtscts = True  #硬件流控 RTS
                
                
        try:
            time.sleep(0.1)
            self.ser.open()
        except:
            QMessageBox.critical(self, "串口异常", "此串口不能被打开！")
            return None

        # 串口打开后，切换开关串口按钮使能状态，防止失误操作        
        if self.ser.isOpen():
            self.ui.btn_serial_open.setEnabled(False)
            self.ui.btn_serial_open.setText('close')


        # 定时器接收数据
        self.timer = QTimer()
        self.timer.timeout.connect(self.data_receive)
        # 打开串口接收定时器，周期为1ms
        self.timer.start(1)  

    # 发送数据
    def data_send(self):
        
        pass
    
        # 接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()
            
            if num > 0:
                time.sleep(0.1)
                num = self.ser.inWaiting()  #延时，再读一次数据，确保数据完整性
        except:
            QMessageBox.critical(self, '串口异常', '串口接收数据异常，请重新连接设备！')
            self.port_close()
            return None
        
        if num > 0:
            data = self.ser.read(num)
            num = len(data)
            
            # 时间显示
            if self.Checkbox5.isChecked():
                self.Text1.insertPlainText((time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + " ")
                
            # HEX显示数据
            if self.Checkbox2.checkState():
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '{:02X}'.format(data[i]) + ' '
                    
                self.Text1.insertPlainText(out_s)
            # ASCII显示数据
            else:
                self.Text1.insertPlainText(data.decode('utf-8'))

            # 接收换行              
            if self.Checkbox6.isChecked():
                self.Text1.insertPlainText('\r\n')
                    
            # 获取到text光标
            textCursor = self.Text1.textCursor()
            # 滚动到底部
            textCursor.movePosition(textCursor.End)
            # 设置光标到text中去
            self.Text1.setTextCursor(textCursor)

            # 统计接收字符的数量
            self.data_num_received += num
            self.Lineedit3.setText(str(self.data_num_received))
        else:
            pass
 
 
 
     # 关闭串口
    def port_close(self):
        try:
            self.timer.stop()
            # self.timer_send.stop()
            
            self.ser.close()
        except:
            QMessageBox.critical(self, '串口异常', '关闭串口失败，请重启程序！')
            return None

        # 切换开关串口按钮使能状态和定时发送使能状态
        self.ui.btn_serial_open.setEnabled(True)
        self.ui.btn_serial_open.setText('open')

        
        # 发送数据和接收数据数目置零
        self.data_num_sended = 0
        # self.Lineedit2.setText(str(self.data_num_sended))
        self.data_num_received = 0
        # self.Lineedit3.setText(str(self.data_num_received))
  
 
    
    def update_port_names(self):
        # 获取可用串口名称
        self.ui.comboBox_port.clear()
        comInfo = QSerialPortInfo.availablePorts()
        # print(comInfo)
        for item in comInfo:
            print(item.portName())
            self.ui.comboBox_port.addItem(item.portName())
        # self.ui.comboBox_port.addItems(comInfo)

    def solt_temp(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        
    def solt_humidity(self):
        self.ui.stackedWidget.setCurrentIndex(2)     

    def solt_wind_speed(self):
        self.ui.stackedWidget.setCurrentIndex(3)  

    def solt_cleanliness(self):
        self.ui.stackedWidget.setCurrentIndex(4)     

        
if __name__ == '__main__':


    app = QApplication(sys.argv)
    # screen = app.primaryScreen()
    # screen_size = screen.size()
    # apply_stylesheet(app, theme='dark_cyan.xml')
    # apply_stylesheet(app, theme='light_red.xml', invert_secondary=True)
    home_win = Home_win()

    home_win.show()

    sys.exit(app.exec())


     
        
        
        
        
        





# from wordcloud import WordCloud
# ham_msg_cloud = WordCloud(width =520, height =260,max_font_size=50, background_color ="black", colormap='Blues').generate(原文本语料)

# plt.figure(figsize=(16,10))
# plt.imshow(ham_msg_cloud, interpolation='bilinear')
# plt.axis('off') # turn off axis
# plt.show()




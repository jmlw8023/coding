





import os, sys
import numpy as np
import random
from pathlib import Path

import cv2 as cv
from PIL import Image, ImageDraw, ImageFont


from PyQt5.QtWidgets import (QApplication, QMainWindow, QDesktopWidget, QFileDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOU root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ui.detect_GUI import Ui_MainWindow
from ui.login_GUI import Ui_Login_win



class Main_win(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    # def show_win(self):
    #     self.show()



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
        img = QPixmap(img_name_path).scaled(self.ui.bg_img_label.width(), self.ui.bg_img_label.height())
        self.ui.bg_img_label.setPixmap(img)


        
    
    def show_main_win(self):
        self.m_win = Main_win()
        self.m_win.show()
    
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

    def info_print(self):
        print('打印信息~~~~~!!!')
        







if __name__ == '__main__':
    app = QApplication(sys.argv)


    main_win = Main_win()
    login_win = Login_win()

    login_win.show()

    # login_win.btn_login.clicked.connect(main_win.show)
    # login_win.btn_exit.clicked.connect(login_win.clos_win)

    sys.exit(app.exec_())
    






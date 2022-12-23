# -*- encoding: utf-8 -*-
'''
@File    :   showimage.py
@Time    :   2022/12/21 14:37:57
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog

from imgshow import Ui_Form



class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.initUI()

    def initUI(self):
        self.ui.openbtn.clicked.connect(self.showImg)
        

    def showImg(self):
        img_name, img_type = QFileDialog.getOpenFileName(self, '打开图片', '', '*.jpg;;*.png;;AllFiles(*)')
        img = QtGui.QPixmap(img_name).scaled(self.ui.label.width(), self.ui.label.height())
        self.ui.img_pre.setPixmap(img)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
    



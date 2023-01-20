# -*- encoding: utf-8 -*-
'''
@File    :   mainwindow_demo.py
@Time    :   2023/01/20 09:21:26
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
import sys

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QIcon


class MainWin(QMainWindow):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)


    def initUI(self):

        self.setWindowTitle('MainWindow')

        self.setGeometry(30, 30, 860, 480)






if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    
    app.setWindowIcon(QIcon('../favicon.ico'))



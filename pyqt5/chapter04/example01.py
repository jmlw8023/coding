# -*- encoding: utf-8 -*-
'''
@File    :   example.py
@Time    :   2022/12/23 12:11:23
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
# import os


import sys

from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QToolTip)


class Example(QWidget):

    def __init__(self, parent=None):
        super().__init__()

        self.initUI()
    
    def initUI(self):
        # self.setGeometry(300, 300, 320, 280)
        # self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('favicon.ico'))
        # self.show()

        QToolTip.setFont(QFont('SansSerif', 10))
        # QToolTip.setFont(QFont('Monospace', 20))

        self.setToolTip('This is a <b> QWidget </b> widget')

        btn = QPushButton('Button', self)
        btn.setToolTip('This is a <b> QPushButton </b> widget')     # 创建提示框可以使用富文本格式的内容
        btn.resize(btn.sizeHint())  # sizeHint()方法提供了一个默认的按钮大小
        btn.move(50, 60)

        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(QCoreApplication.instance().quit)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 120)  

        self.setGeometry(300, 300, 320, 280)
        self.setWindowTitle('Tooltips')
        self.show()
    





if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    exam = Example()

    sys.exit(app.exec_())
    


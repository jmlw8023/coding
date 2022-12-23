# -*- encoding: utf-8 -*-
'''
@File    :   test_pyqt.py
@Time    :   2022/10/20 08:59:01
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os, sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QPushButton, QApplication, QWidget, QMainWindow








class WinForm(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setGeometry(300, 300, 400, 350)
        self.setWindowTitle('click button close window')
        quit = QPushButton('close', self)
        quit.setGeometry(50, 100, 50, 30)
        # quit.setStyleSheet('background-color : blue')
        quit.clicked.connect(self.close)



from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('加载外部网页的例子')
        self.setGeometry(5,30,1355,730)
        self.browser=QWebEngineView()
        #加载外部的web界面
        self.browser.load(QUrl('https://blog.csdn.net/Xin11099'))
        self.setCentralWidget(self.browser)


def testQt5():
    import sys
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)

    widget = QtWidgets.QWidget()

    widget.resize(400, 300)

    widget.setWindowTitle('hello, pyqt5')

    widget.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # win = WinForm()
    # win.show()

    w = MainWindow()
    w.show()

    sys.exit(app.exec_())









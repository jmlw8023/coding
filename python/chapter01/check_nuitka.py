# -*- encoding: utf-8 -*-
'''
@File    :   check_nuitka.py
@Time    :   2022/12/27 14:20:42
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import sys

from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow

#  使用Nuitka 进行打包测试，通过pyqt进行
class Demo(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.initUI()
        self.init_signal_solt()

    def initUI(self):
        self.setFixedSize(480, 320)
        self.setWindowTitle('Pyqt GUI for nuitka packet')

        self.btn = QPushButton('click', self)
        self.btn.resize(80, 20)
        self.btn.move(120, 80)

    def init_signal_solt(self):
        self.btn.clicked.connect(self.show_print)
        

    def show_print(self):
        print('click~~~')


if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    win = Demo()
    win.show()
    sys.exit(app.exec_())
    



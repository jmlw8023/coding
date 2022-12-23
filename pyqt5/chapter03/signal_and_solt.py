# -*- encoding: utf-8 -*-
'''
@File    :   signal_and_solt.py
@Time    :   2022/12/21 15:10:52
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os, sys

from PyQt5.QtWidgets import QPushButton, QApplication, QWidget, QVBoxLayout, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal




class Widgets(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.initUI()


    def initUI(self):   
        layout = QVBoxLayout()
        self.setGeometry(300, 300, 280, 270)
        self.setWindowTitle('信号和槽的例子') 
        self.btn = QPushButton('open', self)
        self.btn.setToolTip('this is a button!')
        self.btn.move(100, 70)
        # signal and solt
        self.btn.clicked.connect(self.show_msg)
        layout.addWidget(self.btn)


    def show_msg(self):
        QMessageBox.information(self, '信息提示框', '弹窗成功！！')
        


class WinForm(QWidget):

    btn_clickd_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()

        self.setWindowTitle('自定义信号和槽函数')
        self.resize(300, 200)
        self.btn = QPushButton('close', self)
        # 链接信号与槽函数
        self.btn.clicked.connect(self.btn.clicked)
        # 接收信号，连接到自定义槽函数
        self.btn_clickd_signal.connect(self.btn_close)
    
    def btn_clicked(self):
        # 发送信号
        self.btn_clickd_signal.emit()

    def btn_close(self):
        self.close()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    # win = Widgets()

    win = WinForm()

    win.show()
    sys.exit(app.exec_())
    




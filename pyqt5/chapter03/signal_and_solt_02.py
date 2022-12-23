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



# 信号
class Signal(QObject):

    # 定义一个信号
    # send_msg = pyqtSignal(object)
    send_msg = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        

    def send(self):   
        self.send_msg.emit('hello pyqt5', 'other message')


class Solt(QObject):

    def __init__(self, parent=None):
        super().__init__(parent)
    
    def get(self, msg1, msg2):
        print('solt get messages: ' + msg1 + '\t' + msg2)




if __name__ == '__main__':
    s = Signal()
    t = Solt()

    print('把信号绑定到槽函数中~')
    s.send_msg.connect(t.get)
    s.send()

    print('把信号与槽函数的链接断开~!')
    s.send_msg.disconnect(t.get)
    s.send()
    




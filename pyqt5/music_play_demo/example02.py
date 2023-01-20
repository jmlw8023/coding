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
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QMessageBox, QDesktopWidget)


class Example(QWidget):

    def __init__(self, parent=None):
        super().__init__()

        self.initUI()
    
    def initUI(self):
        
        # self.setGeometry(300, 300, 320, 280)
        # self.setWindowTitle('Message box')
        # self.show()
        self.resize(300, 200)
        self.center()

        self.setWindowTitle('Center')
        self.show()
    
    def center(self):
        # 获取窗口大小
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        # 本窗体运动
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

        # 方法弃用
        # qr = self.frameGeometry()
        # # QDesktopWidget 提供了用户的桌面信息，包括屏幕的大小
        # cp = QDesktopWidget().availableGeometry.center()
        # qr.moveCenter(cp)
        # self.move(qr.topLeft())
    
    # 按下回车关闭窗口
    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Return:
            self.close()
    
    # 默认关闭窗口，提醒窗口
    def closeEvent(self, event):

        # 创建了一个消息框，上面有俩按钮：Yes和No
        reply = QMessageBox.question(self, 'Message', 
                                    '你确定要退出吗？', 
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            event.accept()  
        else:
            event.ignore()



if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    exam = Example()

    sys.exit(app.exec_())
    


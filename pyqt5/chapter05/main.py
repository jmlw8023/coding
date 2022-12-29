# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/12/29 09:41:13
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os, sys


from PyQt5.QtWidgets import QApplication, QWidget


from ui.mode import Ui_MyForm




class Demo(QWidget):
    '''@
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MyForm()
        self.ui.setupUi(self)

        self.init_signal_solt()

    def init_signal_solt(self):
        self.ui.hl_btn01.clicked.connect(self.btn_fun)
        self.ui.hl_btn02.clicked.connect(self.btn_fun)
        self.ui.vl_btn01.clicked.connect(self.btn_fun)
        self.ui.vl_btn02.clicked.connect(self.btn_fun)
        
    def btn_fun(self):
        print('我被点击了！！')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Demo()
    win.show()

    sys.exit(app.exec_())
    


# -*- encoding: utf-8 -*-
'''
@File    :   rename_files_GUI.py
@Time    :   2023/01/20 11:35:01
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''
# 功能： 实现Windows 文件夹批量重命名
# import packets
import os, sys
import shutil

from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QFileDialog, QMessageBox

from rename import Ui_Form




class Rename(QMainWindow):

    def __init__(self, parnet=None):
        super().__init__()

        # self.ui = Ui_Form()

        self.initUI()
        # 居中
        # self.center()

        self.init_solt_and_sinal()

    def initUI(self):
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.setWindowTitle('Windows 批量重命名程序！！')

    
    def center(self):

        screen = QDesktopWidget().screenGeometry()
        win_size = self.geometry()

        left = (screen.width() - win_size.width()) // 2
        top = (screen.height() - win_size.height()) // 2
        self.move(left, top)



    def folder_choose(self):
        self.folder_path = QFileDialog.getExistingDirectory()
        print(self.folder_path)

    def image_show(self):
        print('open image ---> ')
        self.img_name = QFileDialog.getOpenFileName(
            self, '选择图片', './images', '*.jpg;;*.jpeg;;*.png;;All Files(*)'
        )
        print('open image ---> ', self.img_name[0])
    
    def init_solt_and_sinal(self):

        self.ui.open_btn.clicked.connect(self.folder_choose)
        self.ui.sumit_btn.clicked.connect(self.message_box)


    def message_box(self):
    
        QMessageBox.information(self, '信息提示框', '修改成功！！')



if __name__ == '__main__':

    app = QApplication(sys.argv)

    win = Rename()

    win.show()

    sys.exit(app.exec())

        









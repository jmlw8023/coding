# -*- encoding: utf-8 -*-
'''
@File    :   login_GUI.py
@Time    :   2023/02/07 09:20:45
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
import os, sys


from PyQt5.QtWidgets import QMainWindow, QPushButton, QLineEdit, QFrame, QApplication, QDialog, QVBoxLayout
from PyQt5.QtCore import QCoreApplication



class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化参数属性
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.setWindowTitle('主功能页面')
        self.setFixedWidth(600)
        self.setFixedHeight(600)


class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化登录对话框的内容
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.setWindowTitle('欢迎登录')  # 设置标题
        self.resize(200, 200)  # 设置宽、高
        self.setFixedSize(self.width(), self.height())
        # self.setWindowFlags(Qt.WindowCloseButtonHint)  # 设置隐藏关闭X的按钮

        '''
        定义界面控件设置
        '''
        self.frame = QFrame(self)  # 初始化 Frame对象
        self.verticalLayout = QVBoxLayout(self.frame)  # 设置横向布局
        self.verticalLayout

        self.login_id = QLineEdit()  # 定义用户名输入框
        self.login_id.setPlaceholderText("请输入登录账号")  # 设置默认显示的提示语
        self.verticalLayout.addWidget(self.login_id)  # 将该登录账户设置添加到页面控件

        self.passwd = QLineEdit()  # 定义密码输入框
        self.passwd.setPlaceholderText("请输入登录密码")  # 设置默认显示的提示语
        self.verticalLayout.addWidget(self.passwd)  # 将该登录密码设置添加到页面控件

        self.button_enter = QPushButton()  # 定义登录按钮
        self.button_enter.setText("登录")  # 按钮显示值为登录
        self.verticalLayout.addWidget(self.button_enter)  # 将按钮添加到页面控件

        self.button_quit = QPushButton()  # 定义返回按钮
        self.button_quit.setText("返回")  # 按钮显示值为返回
        self.verticalLayout.addWidget(self.button_quit)  # 将按钮添加到页面控件

        # 绑定按钮事件
        self.button_enter.clicked.connect(self.button_enter_verify)
        self.button_quit.clicked.connect(
            QCoreApplication.instance().quit)  # 返回按钮绑定到退出

    def button_enter_verify(self):
        # 校验账号是否正确
        if self.login_id.text() != "admin":
            print("test1")
            return
        # 校验密码是否正确
        if self.passwd.text() != "admin":
            print("test2")
            return
        # 验证通过，设置QDialog对象状态为允许
        self.accept()











if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    login_win = LoginDialog()

    if login_win.exec_() == QDialog.accepted:

        mw_win = MainWindow()

        mw_win.show()

        sys.exit(app.exec_())
    






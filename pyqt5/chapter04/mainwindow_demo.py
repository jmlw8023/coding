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

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QDesktopWidget, QHBoxLayout, QWidget, QMessageBox, QVBoxLayout


StyleSheet = """
/*这里是通用设置，所有按钮都有效，后面设置可以覆盖这个*/

# QPushButton {
#     border: none; /*去掉边框*/
# }

QPushButton#RedButton {
    background-color: #f44336; /*背景颜色*/
}
#RedButton:hover {
    background-color: #e57373; /*鼠标悬停时背景颜色*/
}

#BlueButton {
    background-color: #2196f3;
    /*限制最小最大尺寸*/
    min-width: 96px;
    max-width: 96px;
    min-height: 96px;
    max-height: 96px;
    border-radius: 48px; /*圆形*/
}
#BlueButton:hover {
    background-color: #64b5f6;
}
#BlueButton:pressed {
    background-color: #bbdefb;
}

#OrangeButton {
    max-height: 48px;
    border-top-right-radius: 20px; /*右上角圆角*/
    border-bottom-left-radius: 20px; /*左下角圆角*/
    background-color: #ff9800;
}
#OrangeButton:hover {
    background-color: #ffb74d;
}
#OrangeButton:pressed {
    background-color: #ffe0b2;
}

"""

class MainWin(QMainWindow):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.initUI()

        # 窗口居中
        self.center()

        self.init_sigal_and_solt()


    def initUI(self):

        self.setWindowTitle('MainWindow')

        # self.move(120, 120)
        # self.resize(860, 480)
        # 等价于前面两个函数
        self.setGeometry(120, 120, 860, 480)
        # self.setGeometry(120, 120, 860, 680)  # 最后一个生效

        self.status = self.statusBar()
        # 秒数
        sec = 6000
        self.status.showMessage('This is the Status Bar, the message will only be displayed 6 seconds!'.format(sec), sec)

        # set button 
        # self.btn = QPushButton('close button', self)
        # # 相对于主窗口位置进行移动
        # self.btn.move(30, 30)
        
        self.btn = QPushButton('close button')

        self.btn2 = QPushButton('message button')
        self.msg_btn = QPushButton('弹窗')
        # 创建布局
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.btn2)
        h_layout.addWidget(self.btn)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.msg_btn)

        # h_layout.addChildLayout(v_layout)
        h_layout.addLayout(v_layout)

        main_frame = QWidget()

        # main_frame.setLayout(v_layout)
        main_frame.setLayout(h_layout)
        self.setCentralWidget(main_frame)



        # # 信号关联槽函数
        # self.btn.clicked.connect(self.btn_solt)


    def init_sigal_and_solt(self):
        # 信号关联槽函数
        self.btn.clicked.connect(self.btn_solt)
        # self.btn2.clicked.connect(self.show_terminal_msg)
        self.btn2.clicked.connect(self.more_msg_box)
        self.msg_btn.clicked.connect(self.message_box)



    def center(self):
        # 屏幕坐标
        screen = QDesktopWidget().screenGeometry()
        # 窗口坐标
        win_size = self.geometry()

        # left = (screen.width() - win_size.width()) / 2
        # top = (screen.height() - win_size.height()) / 2
        # self.move(int(left), int(top))

        left = (screen.width() - win_size.width()) // 2
        top = (screen.height() - win_size.height()) // 2
        self.move(left, top)

    def btn_solt(self):
        sender = self.sender()
        print('sender message : ', sender.text())
        instance_app = QApplication.instance()
        # exit
        instance_app.quit()
    
    def show_terminal_msg(self):
        print('button is clicked!!')

    
    def message_box(self):

        QMessageBox.information(self, '信息提示框', '弹窗成功！！')

    def more_msg_box(self):
        mgs_box = QMessageBox.information(
                None, '信息', 'message boxes',
                QMessageBox.Ok |
                QMessageBox.Open |
                QMessageBox.Save |
                QMessageBox.Cancel |
                QMessageBox.Close |
                QMessageBox.Discard |
                QMessageBox.Apply |
                QMessageBox.Reset |
                QMessageBox.RestoreDefaults |
                QMessageBox.Help |
                QMessageBox.SaveAll |
                QMessageBox.Yes |
                QMessageBox.YesToAll |
                QMessageBox.No |
                QMessageBox.NoToAll |
                QMessageBox.Abort |
                QMessageBox.Retry |
                QMessageBox.Ignore
        )
        


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    # 设置图标
    app.setWindowIcon(QIcon('../favicon.ico'))
    # 增加stype
    app.setStyleSheet(StyleSheet)
    # 对象创建
    win = MainWin()

    win.show()
    sys.exit(app.exec())



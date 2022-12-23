# -*- encoding: utf-8 -*-
'''
@File    :   musicPlayer.py
@Time    :   2022/12/23 15:33:40
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os, sys

from PyQt5.QtCore import QSize, Qt, QTimer, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import(QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton, QProgressBar)
import qtawesome as qta
import requests, traceback

class Music(QMainWindow):

    def __init__(self, parent=None):
        super().__init__()
        self.setFixedSize(480, 220)
        self.setWindowTitle('jolly music')
        self.initUI()
        self.customStyle()
        self.playing = False   #    播放默认初始化状态为 FALSE
        self.player = QMediaPlayer(self)

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.start()
        self.timer.timeout.connect(self.check_music_status)

    # 初始化UI
    def initUI(self):
        # 窗口布局
        self.main_widget = QWidget()
        self.main_widget.setObjectName('main_widget')
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        # 标题
        self.title = QLabel('不问红尘滚滚事，但听溪涧流水.....')

        # 关闭按钮
        self.close_btn = QPushButton('')
        self.close_btn.setFixedSize(15, 15)

        # 音乐状态按钮
        self.status_label = QLabel('')
        self.status_label.setFixedSize(15, 15)

        # 播放按钮
        play_icon = qta.icon("fa.play-circle",)
        self.play_btn = QPushButton(play_icon,"")
        self.play_btn.setIconSize(QSize(80, 80))
        self.play_btn.setFixedSize(82,82)
        self.play_btn.setObjectName("play_btn")
        self.play_btn.clicked.connect(self.play_music)

        # 下一首按钮
        next_icon = qta.icon("fa.play-circle-o")
        self.next_btn = QPushButton(next_icon,"")
        self.next_btn.setIconSize(QSize(80,80))
        self.next_btn.setFixedSize(82,82)
        self.next_btn.setObjectName("next_btn")
        self.next_btn.clicked.connect(self.next_music)

        # 进度条
        self.process_bar = QProgressBar()
        self.process_value = 0
        self.process_bar.setValue(self.process_value)
        self.process_bar.setFixedHeight(5)
        self.process_bar.setTextVisible(False)

        self.main_layout.addWidget(self.close_btn,0,0,1,1)
        self.main_layout.addWidget(self.title,0,1,1,1)
        self.main_layout.addWidget(self.status_label,1,0,1,1)
        self.main_layout.addWidget(self.play_btn, 1, 1, 1, 1)
        self.main_layout.addWidget(self.next_btn, 1, 2, 1, 1)
        self.main_layout.addWidget(self.process_bar,2,0,1,3)

        self.setCentralWidget(self.main_widget)

        # self.setWindowOpacity(0.9) # 设置窗口透明度
        # self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.setWindowFlag(Qt.FramelessWindowHint)  # 隐藏边框

    # 设置样式
    def customStyle(self):

        self.setStyleSheet('''
            #main_widget{
                border-radius:5px;
            }
            #play_btn,#pervious_btn,#next_btn{
                border:none;
            }
            #play_btn:hover,#pervious_btn:hover,#next_btn:hover{
                background:gray;
                border-radius:5px;
                cursor:pointer;
            }
        ''')
        self.close_btn.setStyleSheet('''
            QPushButton{
                background:#F76677;
                border-radius:5px;
                }
            QPushButton:hover{
                background:red;
                }''')
        self.status_label.setStyleSheet('''
            QLabel{
                background:#F7D674;
                border-radius:5px;
                }
        ''')

      # 关闭程序
    def close_btn_event(self):
        self.close()

    # 鼠标长按事件
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_drag = True
            self.m_DragPosition = event.globalPos() - self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))

    # 鼠标移动事件
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_drag:
            self.move(QMouseEvent.globalPos() - self.m_DragPosition)
            QMouseEvent.accept()

    # 鼠标释放事件
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_drag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    # 播放音乐
    def play_music(self):
        try:
            # 播放音乐
            if self.playing is False:
                self.playing = True # 设置播放状态为是
                self.play_btn.setIcon(qta.icon("fa.pause-circle")) # 设置播放图标
                player_status = self.player.mediaStatus() # 获取播放器状态
                # print("当前播放状态：",player_status)
                if player_status == 6:
                    # 设置状态标签为绿色
                    self.status_label.setStyleSheet('''QLabel{background:#6DDF6D;border-radius:5px;}''')
                    self.player.play()
                else:
                    self.next_music()
            # 暂停音乐
            else:
                # 设置状态为蓝色
                self.status_label.setStyleSheet('''QLabel{background:#0099CC;border-radius:5px;}''')
                self.playing = False
                self.play_btn.setIcon(qta.icon("fa.play-circle"))
                self.player.pause()
        except Exception as e:
            print(repr(e))

    # 下一首音乐
    def next_music(self):
        try:
            # 设置状态为黄色
            self.status_label.setStyleSheet('''
                QLabel{
                    background:#F7D674;
                    border-radius:5px;
                    }
            ''')
            self.playing = True # 设置播放状态为是
            self.play_btn.setIcon(qta.icon("fa.pause-circle")) # 修改播放图标
            self.process_value = 0 # 重置进度值

            # 获取网络歌曲
            self.get_music_thread = GetMusicThread()
            self.get_music_thread.finished_signal.connect(self.init_player)
            self.get_music_thread.start()
        except Exception as e:
            print(traceback.print_exc())

    # 设置播放器
    def init_player(self,url):
        # print("获取到音乐链接：",url)
        content = QMediaContent(QUrl(url))
        self.player.setMedia(content)
        self.player.setVolume(50)
        self.player.play()
        self.duration = self.player.duration()  # 音乐的时长
        # 设置状态为绿色
        self.status_label.setStyleSheet('''
            QLabel{
                background:#6DDF6D;
                border-radius:5px;
                }
        ''')

        # 进度条计时器
        self.process_timer = QTimer()
        self.process_timer.setInterval(1000)
        self.process_timer.start()
        self.process_timer.timeout.connect(self.process_timer_status)

    # 定时器
    def check_music_status(self):
        player_status = self.player.mediaStatus()
        player_duration = self.player.duration()
        # print("音乐时间：",player_duration)
        # print("当前播放器状态",player_status)
        if player_status == 7:
            self.next_music()

        if player_duration > 0:
            self.duration = player_duration

    # 进度条定时器
    def process_timer_status(self):
        try:
            if self.playing is True:
                self.process_value += (100 / (self.duration/1000))
                # print("当前进度：",self.process_value)
                self.process_bar.setValue(self.process_value)
        except Exception as e:
            print(repr(e))


# 异步子线程获取音乐链接
class GetMusicThread(QThread):
    finished_signal = pyqtSignal(str)
    # finished_signal = Signal(str) # PySide2的使用方法

    def __init__(self,parent=None):
        super().__init__(parent)

    def run(self):
        reps = requests.post("https://api.uomg.com/api/rand.music?format=json")
        print(reps.json())
        file_url = reps.json()['data']['url']
        self.finished_signal.emit(file_url)


def main():
    import sys
    app = QApplication(sys.argv)
    gui = Music()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


import sys

from PyQt5.QtCore import QFile
from PyQt5.QtGui import QFont, QImage, QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton








class LabelDemo(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(250, 350, 840, 680)
        self.setWindowTitle('Label demo')


        self.label()


    
    def label(self):

        self.label = QLabel('This is first Python GUI program', self)
        self.label.move(100, 300)
        self.label.setFixedSize(650, 680)
        # self.label.setLineWidth(15)

        self.label.setFont(QFont('Sanserif', 15))
        self.label.setStyleSheet('color:blue')

        img = QPixmap('../heart.jpg')
        img_label = QLabel('image label', self)
        img_label.setFixedWidth(500)
        img_label.setFixedHeight(450)
        img_label.move(80, 10)
        img_label.setPixmap(img)






        



        








if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    win = LabelDemo()

    win.show()

    sys.exit(app.exec_())

    







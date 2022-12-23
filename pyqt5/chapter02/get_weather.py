# -*- encoding: utf-8 -*-
'''
@File    :   get_weather.py
@Time    :   2022/12/21 09:27:36
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os, sys

import requests
from PyQt5.QtWidgets import QApplication, QMainWindow

from weather import Ui_Form




class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.initUI()

    def initUI(self):
        self.ui.queryBtn.clicked.connect(self.queryWeather)
        self.ui.clearBtn.clicked.connect(self.clear_result)


    def queryWeather(self):
        print('\t start queryWeather\t')
        city_name = self.ui.weatherComboBox.currentText()
        city_code = self.trans_city_name(city_name)
        url = r'http://www.weather.com.cn/data/sk/' + city_code + '.html'
        rep = requests.get(url)
        rep.encoding = 'utf-8'
        print(rep.json())

        msg1 = '城市: {} \n'.format(rep.json()['weatherinfo']['city']) 
        msg2 = '风向: {} \n'.format(rep.json()['weatherinfo']['WD']) 
        msg3 = '温度: {} \n'.format(rep.json()['weatherinfo']['temp']) 
        msg4 = '风力: {} \n'.format(rep.json()['weatherinfo']['WS']) 
        msg5 = '湿度: {} \n'.format(rep.json()['weatherinfo']['SD']) 
        result = msg1 + msg2 + msg3 + msg4 + msg5
        self.ui.resultText.setText(result)      

    def trans_city_name(self ,cityName):
        cityCode = ''
        if cityName == '北京' :
                cityCode = '101010100'
        elif cityName == '天津' :
                cityCode = '101030100'
        elif cityName == '上海' :	
                cityCode = '101020100'
        elif cityName == '广州' :
                cityCode = '101280101'

        return cityCode	

    def clear_result(self):
        print('\t clear result \t')
        self.ui.resultText.clear()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
    














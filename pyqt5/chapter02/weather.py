# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\source\code\pyqt\mycode\chapter02\weather.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(497, 450)
        self.centralwidget = QtWidgets.QWidget(Form)
        self.centralwidget.setObjectName("centralwidget")
        self.queryBtn = QtWidgets.QPushButton(self.centralwidget)
        self.queryBtn.setGeometry(QtCore.QRect(130, 300, 75, 23))
        self.queryBtn.setObjectName("queryBtn")
        self.clearBtn = QtWidgets.QPushButton(self.centralwidget)
        self.clearBtn.setGeometry(QtCore.QRect(240, 300, 75, 23))
        self.clearBtn.setObjectName("clearBtn")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(19, 19, 421, 271))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(40, 20, 41, 21))
        self.label.setObjectName("label")
        self.weatherComboBox = QtWidgets.QComboBox(self.groupBox)
        self.weatherComboBox.setGeometry(QtCore.QRect(120, 20, 131, 22))
        self.weatherComboBox.setObjectName("weatherComboBox")
        self.weatherComboBox.addItem("")
        self.weatherComboBox.addItem("")
        self.weatherComboBox.addItem("")
        self.weatherComboBox.addItem("")
        self.resultText = QtWidgets.QTextEdit(self.groupBox)
        self.resultText.setGeometry(QtCore.QRect(40, 60, 351, 191))
        self.resultText.setObjectName("resultText")
        Form.setCentralWidget(self.centralwidget)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "MainWindow"))
        self.queryBtn.setText(_translate("Form", "查询"))
        self.clearBtn.setText(_translate("Form", "清除"))
        self.label.setText(_translate("Form", "城市"))
        self.weatherComboBox.setCurrentText(_translate("Form", "广州"))
        self.weatherComboBox.setItemText(0, _translate("Form", "广州"))
        self.weatherComboBox.setItemText(1, _translate("Form", "北京"))
        self.weatherComboBox.setItemText(2, _translate("Form", "天津"))
        self.weatherComboBox.setItemText(3, _translate("Form", "上海"))

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\source\code\coding\pyqt5\chapter05\ui\mode.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MyForm(object):
    def setupUi(self, MyForm):
        MyForm.setObjectName("MyForm")
        MyForm.resize(813, 500)
        self.horizontalLayoutWidget = QtWidgets.QWidget(MyForm)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 320, 211))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.hl_btn01 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.hl_btn01.setObjectName("hl_btn01")
        self.horizontalLayout.addWidget(self.hl_btn01)
        self.hl_btn02 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.hl_btn02.setObjectName("hl_btn02")
        self.horizontalLayout.addWidget(self.hl_btn02)
        self.hl_btn03 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.hl_btn03.setObjectName("hl_btn03")
        self.horizontalLayout.addWidget(self.hl_btn03)
        self.hl_btn04 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.hl_btn04.setObjectName("hl_btn04")
        self.horizontalLayout.addWidget(self.hl_btn04)
        self.verticalLayoutWidget = QtWidgets.QWidget(MyForm)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(360, 20, 241, 211))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.vl_btn01 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.vl_btn01.setObjectName("vl_btn01")
        self.verticalLayout.addWidget(self.vl_btn01)
        self.vl_btn02 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.vl_btn02.setObjectName("vl_btn02")
        self.verticalLayout.addWidget(self.vl_btn02)
        self.vl_btn03 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.vl_btn03.setObjectName("vl_btn03")
        self.verticalLayout.addWidget(self.vl_btn03)
        self.vl_btn04 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.vl_btn04.setObjectName("vl_btn04")
        self.verticalLayout.addWidget(self.vl_btn04)
        self.horizontal_Layout = QtWidgets.QLabel(MyForm)
        self.horizontal_Layout.setGeometry(QtCore.QRect(90, 240, 161, 16))
        self.horizontal_Layout.setObjectName("horizontal_Layout")
        self.label_2 = QtWidgets.QLabel(MyForm)
        self.label_2.setGeometry(QtCore.QRect(420, 240, 161, 16))
        self.label_2.setObjectName("label_2")
        self.gridLayoutWidget = QtWidgets.QWidget(MyForm)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 290, 241, 151))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.grid_btn02 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.grid_btn02.setObjectName("grid_btn02")
        self.gridLayout.addWidget(self.grid_btn02, 1, 0, 1, 1)
        self.grid_btn01 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.grid_btn01.setObjectName("grid_btn01")
        self.gridLayout.addWidget(self.grid_btn01, 0, 0, 1, 1)
        self.grid_btn03 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.grid_btn03.setObjectName("grid_btn03")
        self.gridLayout.addWidget(self.grid_btn03, 0, 1, 1, 1)
        self.formLayoutWidget = QtWidgets.QWidget(MyForm)
        self.formLayoutWidget.setGeometry(QtCore.QRect(350, 290, 211, 151))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.form_btn01 = QtWidgets.QPushButton(self.formLayoutWidget)
        self.form_btn01.setObjectName("form_btn01")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.form_btn01)
        self.form_btn02 = QtWidgets.QPushButton(self.formLayoutWidget)
        self.form_btn02.setObjectName("form_btn02")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.form_btn02)
        self.form_btn03 = QtWidgets.QPushButton(self.formLayoutWidget)
        self.form_btn03.setObjectName("form_btn03")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.form_btn03)
        self.form_btn04 = QtWidgets.QPushButton(self.formLayoutWidget)
        self.form_btn04.setObjectName("form_btn04")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.form_btn04)
        self.form_btn05 = QtWidgets.QPushButton(self.formLayoutWidget)
        self.form_btn05.setObjectName("form_btn05")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.form_btn05)
        self.dateTimeEdit = QtWidgets.QDateTimeEdit(MyForm)
        self.dateTimeEdit.setGeometry(QtCore.QRect(620, 300, 194, 22))
        self.dateTimeEdit.setObjectName("dateTimeEdit")
        self.dateEdit = QtWidgets.QDateEdit(MyForm)
        self.dateEdit.setGeometry(QtCore.QRect(620, 260, 110, 22))
        self.dateEdit.setObjectName("dateEdit")
        self.dial = QtWidgets.QDial(MyForm)
        self.dial.setGeometry(QtCore.QRect(650, 360, 50, 64))
        self.dial.setObjectName("dial")
        self.horizontalScrollBar = QtWidgets.QScrollBar(MyForm)
        self.horizontalScrollBar.setGeometry(QtCore.QRect(430, 460, 341, 20))
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.toolButton = QtWidgets.QToolButton(MyForm)
        self.toolButton.setGeometry(QtCore.QRect(120, 460, 101, 18))
        self.toolButton.setObjectName("toolButton")

        self.retranslateUi(MyForm)
        QtCore.QMetaObject.connectSlotsByName(MyForm)

    def retranslateUi(self, MyForm):
        _translate = QtCore.QCoreApplication.translate
        MyForm.setWindowTitle(_translate("MyForm", "Form"))
        self.hl_btn01.setText(_translate("MyForm", "hl_btn01"))
        self.hl_btn02.setText(_translate("MyForm", "hl_btn02"))
        self.hl_btn03.setText(_translate("MyForm", "hl_btn03"))
        self.hl_btn04.setText(_translate("MyForm", "hl_btn04"))
        self.vl_btn01.setText(_translate("MyForm", "vl_btn01"))
        self.vl_btn02.setText(_translate("MyForm", "vl_btn02"))
        self.vl_btn03.setText(_translate("MyForm", "vl_btn03"))
        self.vl_btn04.setText(_translate("MyForm", "vl_btn04"))
        self.horizontal_Layout.setText(_translate("MyForm", "horizontal_Layout"))
        self.label_2.setText(_translate("MyForm", "vertical_Layout"))
        self.grid_btn02.setText(_translate("MyForm", "grid_btn02"))
        self.grid_btn01.setText(_translate("MyForm", "grid_btn01"))
        self.grid_btn03.setText(_translate("MyForm", "grid_btn03"))
        self.form_btn01.setText(_translate("MyForm", "form_btn01"))
        self.form_btn02.setText(_translate("MyForm", "form_btn02"))
        self.form_btn03.setText(_translate("MyForm", "form_btn03"))
        self.form_btn04.setText(_translate("MyForm", "form_btn04"))
        self.form_btn05.setText(_translate("MyForm", "form_btn05"))
        self.toolButton.setText(_translate("MyForm", "toolButton"))

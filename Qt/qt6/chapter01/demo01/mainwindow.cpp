#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // title
    this->setWindowTitle("Qt6 第一个应用程序");

    this->setMaximumSize(640, 480);

    //背景颜色
//    this->setStyleSheet("background:blue");
    // 隐藏标题栏
    this->setWindowFlag(Qt::WindowMinMaxButtonsHint);
//    this->setWindowFlag(Qt::WindowCloseButtonHint);
    // 通过按钮关闭窗口
    connect(ui->close_btn, SIGNAL(clicked()), this, SLOT(close()));


}

MainWindow::~MainWindow()
{
    delete ui;
}


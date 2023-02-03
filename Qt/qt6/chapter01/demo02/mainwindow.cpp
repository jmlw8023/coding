//#pragma execution_character_set("utf-8")
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QLabel>
#include <QMessageBox>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->setWindowTitle("Qt6窗口程序");

    // 创建一个QLabel控件
    QLabel *label = new QLabel(this);
    // Qlabel 中控件显示的文字内容
    label->setText("Hello Qt");
    // 文本显示的位置：X轴，Y轴，Qlabel 控件宽度和高度
    label->setGeometry(QRect(180, 300, 200, 300));

}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_login_btn_clicked()
{
//    accept();
    QMessageBox msgBox;
//    msgBox.setText(QString::fromLocal8Bit("登录的消息对话框~~"));
    msgBox.information(this, QString::fromLocal8Bit("login page"), QStringLiteral("登录的消息对话框~~"));
//    msgBox.exec();
    qDebug() << "be click!!" ;
}


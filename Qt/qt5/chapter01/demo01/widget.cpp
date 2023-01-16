#include "widget.h"
#include "ui_widget.h"
#include <iostream>

using namespace std;

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
}

Widget::~Widget()
{
    delete ui;
}


void Widget::on_btn_clicked()
{
    cout << "我被点击了！！！"  << endl;
}

#include "mainwindow.h"

#include <QApplication>
#include <QLabel>
#include <QDir>
#include <QDebug>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    qDebug() << QDir::currentPath();

//    QLabel label("this is label text!!");
//    label.show();

    return a.exec();
}

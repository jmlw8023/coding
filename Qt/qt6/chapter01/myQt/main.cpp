#include "mainwindow.h"

#include <QApplication>

#include <vector>

// OpenCV 相关
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>



int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    // test_SimpleBlobDetector();
    w.show();

    return a.exec();
}

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <QDebug>
#include <QFileDialog>
#include <QPixmap>

#include <QMessageBox>
#include <QDesktopServices>

#include <QPropertyAnimation>


#include <vector>

// OpenCV 相关
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "imgprocess.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void initBase();
    void initImageTest();

    void initImage();

    void testSideWindow();
    void btnSideWindowSolt(bool btnFlag);


private:


    QPropertyAnimation *m_propertyAnimation;
    QPropertyAnimation *m_propertyAnimation2;
    bool m_bSideflag = false;







    //SimpleBlobDetector
    cv::Mat srcImg;
    // cv::SimpleBlobDetector::Params pBLOBDetector;

    // int g_d = 15;
    // int g_sigmaColor = 20;
    // int g_sigmaSpace = 50;
    // cv::Mat img, image;
    QString filePath {};
    Imgprocess view;

    QPoint lastPos; // 记录鼠标最后的位置


protected:

    QImage matToQImage(const cv::Mat& mat);

    void setThresholdImage(int valueSpin);

    bool eventFilter(QObject *obj, QEvent *event) override ;
    void handleWheelEvent(QWheelEvent *event);
    void handleMouseEvent(QMouseEvent *event);
    void resizeEvent(QResizeEvent *enent);

    void wheelEvent(QWheelEvent *event) override;

    // 重写关闭窗口事件
    void closeEvent(QCloseEvent *event) override;

public:


    // void on_Trackbar(int, void*);
    void test_blur();
    void testBlobProcess();

    void videoTest();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H

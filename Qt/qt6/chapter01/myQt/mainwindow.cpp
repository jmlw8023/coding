#include "mainwindow.h"
#include "./ui_mainwindow.h"



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 页面切换
    initBase();

    initImage();

    initImageTest();

    // testBlobProcess();
    // test_blur();


    testSideWindow();


}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::testSideWindow()
{

    ui->pushButton->setText(">>");
    ui->pushButton->setCheckable(true);

    ui->widget_side->move(-ui->widget_side->width(),0);// 左侧停靠
    ui->pushButton->move(-1,ui->widget_side->height()/2);
    m_propertyAnimation = new QPropertyAnimation(ui->widget_side,"geometry");
    m_propertyAnimation->setEasingCurve(QEasingCurve::InOutSine);
    m_propertyAnimation->setDuration(800);
    m_propertyAnimation2 = new QPropertyAnimation(ui->pushButton,"geometry");
    m_propertyAnimation2->setEasingCurve(QEasingCurve::InOutSine);
    m_propertyAnimation2->setDuration(800);

    connect(ui->btn_click, &QPushButton::clicked, this, [&](){
        QMessageBox::critical(this, "警告", "critical 警告的内容");
    });
    connect(ui->btn_inside, &QPushButton::clicked, this, [&](){
        QMessageBox::information(this, "提示", "这个是内部按钮！");
    });
    connect(ui->pushButton, &QPushButton::clicked, this, &MainWindow::btnSideWindowSolt);

}

void MainWindow::btnSideWindowSolt(bool btnFlag)
{

    if (btnFlag)
    {
        qDebug() << "btn Flag = " << m_bSideflag;
        m_propertyAnimation->setStartValue(QRect(-this->rect().width(),0,ui->widget_side->width(),ui->widget_side->height()));
        m_propertyAnimation->setEndValue(QRect(0,0,ui->widget_side->width(),ui->widget_side->height()));
        m_propertyAnimation->start();
        m_propertyAnimation2->setStartValue(QRect(-1,ui->widget_side->height()/2-ui->pushButton->height()/2,ui->pushButton->width(),ui->pushButton->height()));
        m_propertyAnimation2->setEndValue(QRect(ui->widget_side->width()-2,ui->widget_side->height()/2-ui->pushButton->height()/2,ui->pushButton->width(),ui->pushButton->height()));
        m_propertyAnimation2->start();
        ui->pushButton->setText("<<");
        // m_bSideflag = !m_bSideflag;
    }
    else
    {
        qDebug() << "btn Flag = " << m_bSideflag;
        m_propertyAnimation->setStartValue(QRect(0,0,ui->widget_side->width(),ui->widget_side->height()));
        m_propertyAnimation->setEndValue(QRect(-this->rect().width(),0,ui->widget_side->width(),ui->widget_side->height()));
        m_propertyAnimation->start();
        m_propertyAnimation2->setStartValue(QRect(ui->widget_side->width()-2,ui->widget_side->height()/2-ui->pushButton->height()/2,ui->pushButton->width(),ui->pushButton->height()));
        m_propertyAnimation2->setEndValue(QRect(-1,ui->widget_side->height()/2-ui->pushButton->height()/2,ui->pushButton->width(),ui->pushButton->height()));
        m_propertyAnimation2->start();
        ui->pushButton->setText(">>");
        // m_bSideflag = !m_bSideflag;
    }

}




void MainWindow::initImage()
{
    // view.setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    // view.setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    // view.setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    ui->label_img_src->setScaledContents(true);
    ui->label_img_src->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->label_img_src->setAlignment(Qt::AlignCenter);
    ui->label_img_src->setMouseTracking(true);
    ui->label_img_src->setFocusPolicy(Qt::StrongFocus);

    // ui->scrollArea_img_src->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    // ui->scrollArea_img_src->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    // QDir dir("..");
    // QString path_excel = dir.absoluteFilePath("D:/source/code/datasets/b113/images");
    // QString path_excel = dir.absoluteFilePath("../../../datasets/b113/images");
    connect(ui->btn_img_open, &QPushButton::clicked, this, [&](){
        filePath = QFileDialog::getOpenFileName(
            this,
            "请选择一张图像",
            "../../../../datasets/b113/images",
            "Images (*.bmp *.png *.jpg *.tif *.GIF )"

            );

        if (filePath.isEmpty())
            return;

        else
        {
            QPixmap pixmap (filePath);

            // view.setImage(pixmap);
            // view.show();


            ui->label_img_src->setAlignment(Qt::AlignCenter);
            ui->label_img_src->setPixmap(pixmap.scaled(ui->label_img_src->size(), Qt::KeepAspectRatio));
            // ui->label_img_src->setPixmap(pixmap);

            ui->label_img_src->show();

            // ui->scrollArea_img_src->setWidget(ui->label_img_src);
            // ui->scrollArea_img_src->show();
            // ui->label_img_src->installEventFilter(&scrollArea);


        }

        // QObject::connect(ui->scrollArea_img_src, &QScrollArea::wheelEvent, [&](QWheelEvent *event) {
        //     qreal scaleFactor = 1.15;
        //     if (event->angleDelta().y() < 0)
        //         scaleFactor = 1.0 / scaleFactor;
        //     ui->label_img_src->setPixmap(ui->label_img_src->pixmap().scaled(ui->label_img_src->pixmap().size() * scaleFactor));
        // });

    });

    connect(ui->comboBox_img_feature, &QComboBox::currentIndexChanged, this, [&](int currentId){

                qDebug() << "currentId = " << currentId  << " !";

    });


    connect(ui->btn_img_open_dir, &QPushButton::clicked, this, [&](){

        // QFileInfo fileInfo(QDir::currentPath());
        // auto pathDir = fileInfo.path();
        QString strFilePath = QDir::currentPath() + "/RunResultSave";
        // 判断文件夹是否存在，如果不存在则创建
        if (!QDir().exists(strFilePath))
        {
            QDir().mkdir(strFilePath);
        }
        strFilePath = "file:///" + strFilePath;
        QDesktopServices::openUrl(QUrl(strFilePath));

    });




}


void MainWindow::handleWheelEvent(QWheelEvent *event) {

    int delta = event->angleDelta().y();
    qreal scaleFactor = 1.15;

    if (delta > 0) {
        ui->label_img_src->resize(ui->label_img_src->width() * scaleFactor, ui->label_img_src->height() * scaleFactor);
    } else {
        ui->label_img_src->resize(ui->label_img_src->width() / scaleFactor, ui->label_img_src->height() / scaleFactor);
    }
}

void MainWindow::handleMouseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        if (event->type() == QEvent::MouseButtonPress) {
            // 记录按下位置
                lastPos = event->pos();
        } else if (event->type() == QEvent::MouseMove) {
            // 计算偏移量并移动图像
                QPoint delta = event->pos() - lastPos;
            ui->label_img_src->move(ui->label_img_src->pos() + delta);
            lastPos = event->pos();
        }
    }
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    // const QSize size = event->size();

    // if (srcImg.data)
    // {
    //     cv::Mat gray;
    //     cv::cvtColor(srcImg, gray, cv::COLOR_BGR2GRAY);
    //     // QImage qimg = matToQImage(srcImg);
    //     QPixmap pixmap = QPixmap::fromImage(matToQImage(gray));
    //     ui->label_img_src_test->setPixmap(pixmap.scaled(width()/2, height()/1.3));

    //     setThresholdImage(ui->spinBox_img->value());
    // }


    // QMainWindow::rezizeEvent(event);
}


bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == ui->label_img_src) {
        if (event->type() == QEvent::Wheel) {
            handleWheelEvent(static_cast<QWheelEvent *>(event));
            return true;
        } else if (event->type() == QEvent::MouseButtonPress ||
                   event->type() == QEvent::MouseButtonRelease ||
                   event->type() == QEvent::MouseMove) {
            handleMouseEvent(static_cast<QMouseEvent *>(event));
            return true;
        }
    }
    return QWidget::eventFilter(obj, event);
}

void MainWindow::wheelEvent(QWheelEvent *event)
{

    // QPoint numDegrees;                                     // 定义指针类型参数numDegrees用于获取滚轮转角
    // numDegrees = event->angleDelta();                      // 获取滚轮转角
    // int step = 0;                                          // 设置中间参数step用于将获取的数值转换成整数型
    // if (!numDegrees.isNull())                              // 判断滚轮是否转动
    // {
    //     step = numDegrees.y();                             // 将滚轮转动数值传给中间参数step
    // }
    // event->accept();                                       // 获取事件
    // int currentWidth = ui->label_img_src->width();                  // 获取当前图像的宽
    // int currentHeight = ui->label_img_src->height();                // 获取当前图像的高
    // currentWidth += step;                                  // 对当前图像的高累加
    // currentHeight += step;                                 // 对当前图像的宽累加
    // if (step > 0)                                          // 判断图像是放大还是缩小
    // {
    //     QString imgsize = QString("图像放大,尺寸为：%1 * %2")
    //                           .arg(currentWidth).arg(currentHeight);
    //     qDebug() << imgsize;                               // 打印放大后的图像尺寸
    // }
    // else
    // {
    //     QString imgsize = QString("图像缩小,尺寸为：%1 * %2")
    //                           .arg(currentWidth).arg(currentHeight);
    //     qDebug() << imgsize;                                // 打印缩小后的图像尺寸
    // }
    // ui->label_img_src->resize(currentWidth, currentHeight);          // 通过更新图像显示控件的大小来更新图像大小

    int delta = event->angleDelta().y();
    qreal scaleFactor = 1.15;

    if (delta > 0) {
        ui->label_img_src->resize(ui->label_img_src->width() * scaleFactor, ui->label_img_src->height() * scaleFactor);
    } else {
        ui->label_img_src->resize(ui->label_img_src->width() / scaleFactor, ui->label_img_src->height() / scaleFactor);
    }

}

void MainWindow::closeEvent(QCloseEvent *event)
{
    // QMessageBox::StandardButton button = QMessageBox::question(this, "提示", "确定要退出吗？", QMessageBox::Yes | QMessageBox::No);
    // if (button == QMessageBox::Yes)
    // {
    //     // 用户选择"是"，关闭应用程序
    //     event->accept();
    // }
    // else
    // {
    //     // 用户选择"否"，取消关闭操作
    //     event->ignore();
    // }


}


void MainWindow::initBase()
{
    // 页面切换
    connect(ui->btn_home, &QPushButton::clicked, this, [&](){
        ui->stackedWidget->setCurrentWidget(ui->page_home);
    });

    connect(ui->btn_video, &QPushButton::clicked, this, [&](){
        ui->stackedWidget->setCurrentWidget(ui->page_video);
    });


    connect(ui->btn_img, &QPushButton::clicked, this, [&](){
        ui->stackedWidget->setCurrentWidget(ui->page_img);
    });

    connect(ui->btn_set, &QPushButton::clicked, this, [&](){
        ui->stackedWidget->setCurrentWidget(ui->page_set);
    });

    connect(ui->btn_other, &QPushButton::clicked, this, [&](){
        ui->stackedWidget->setCurrentWidget(ui->page_other);
    });


}

void MainWindow::initImageTest()
{
    connect(ui->btn_img_open_file, &QPushButton::clicked, this, [&](){
        QString img_path = QFileDialog::getOpenFileName(
            this,
            "请选择一张图像",
            "..",
            "Images (*.bmp *.png *.jpg *.tif *.GIF )"

            ) ;

        if (img_path.isEmpty()) return;

        qDebug() << img_path;
        srcImg = cv::imread(img_path.toStdString());

        if (!srcImg.data)
        {
            QMessageBox::information(this, "信息", "载入的图像有问题！");
            return;
        }

        cv::Mat gray;
        cv::cvtColor(srcImg, gray, cv::COLOR_BGR2GRAY);
        // QImage qimg = matToQImage(srcImg);
        QPixmap pixmap = QPixmap::fromImage(matToQImage(gray));
        ui->label_img_src_test->setPixmap(pixmap.scaled(this->width()/2, this->height()/1.3));

        // setThresholdImage(ui->spinBox_img->value());

    });

    connect(ui->spinBox_img, &QSpinBox::valueChanged, ui->horizontalSlider_img, &QSlider::setValue);
    connect(ui->horizontalSlider_img, &QSlider::valueChanged, ui->spinBox_img, &QSpinBox::setValue);
    connect(ui->horizontalSlider_img, &QSlider::valueChanged, this, &MainWindow::setThresholdImage);




}


QImage MainWindow::matToQImage(const cv::Mat& mat)
{
    // 8-bit, 3-channel image (CV_8UC3)
    if (mat.type() == CV_8UC3)
    {
        QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888);
        return image.rgbSwapped(); // OpenCV uses BGR order, convert to RGB
    }
    // 8-bit, single-channel image (CV_8UC1)
    else if (mat.type() == CV_8UC1)
    {
        QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Indexed8);
        // Set color table for grayscale image
        QVector<QRgb> colorTable(256);
        for (int i = 0; i < 256; ++i)
            colorTable[i] = qRgb(i, i, i);
        image.setColorTable(colorTable);
        return image;
    }
    else
    {
        qDebug() << "Unsupported image format!";
        return QImage();
    }
}

void MainWindow::setThresholdImage(int valueSpin)
{
    if (srcImg.data)
    {
        cv::Mat grayImg;
        cv::cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);
        qDebug() << "spinBox Vaule = " << valueSpin;
        cv::threshold(grayImg, grayImg, valueSpin, 255, cv::THRESH_BINARY_INV);
        QPixmap pixmap = QPixmap::fromImage(matToQImage(grayImg));
        ui->label_img_dst_test->setPixmap(pixmap.scaled(this->width()/2, this->height()/2));
    }
}


int g_d = 15;
int g_sigmaColor = 20;
int g_sigmaSpace = 50;
cv::Mat img, image;


void on_Trackbar(int, void*)
{
    bilateralFilter(img, image, g_d, g_sigmaColor, g_sigmaSpace);
    imshow("output", image);
}

void MainWindow::test_blur()
{

    cv::Mat src;
    src = cv::imread("D:/source/code/datasets/b113/images/20230329/19133_115_21_0910.bmp");
    img = src.clone();


    if (img.empty())
    {
        qDebug() << "Could not load image ... " ;
        return ;
    }

    cv::Mat image = cv::Mat::zeros(img.rows, img.cols, img.type());
    bilateralFilter(img, image, g_d, g_sigmaColor, g_sigmaSpace);

    cv::namedWindow("output");

    cv::createTrackbar("核直径","output", &g_d, 50, on_Trackbar);
    cv::createTrackbar("颜色空间方差","output", &g_sigmaColor, 100, on_Trackbar);
    cv::createTrackbar("坐标空间方差","output", &g_sigmaSpace, 100, on_Trackbar);

    imshow("input", img);
    // imshow("output", image);

    cv::waitKey(0);

}

void MainWindow::testBlobProcess()
{
    using namespace  cv;
    // 2.9 单个像素长度um  5倍
    double dbUnit = 2.9/(1000*5);

    // 定义显示窗口
    namedWindow("src", WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow("threshold", WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow("morphologyEx x1", WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow("morphologyEx x2", WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow("canny", WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow("dst", WINDOW_NORMAL|WINDOW_KEEPRATIO);
    resizeWindow("src", 1080,720);
    resizeWindow("threshold", 1080,720);
    resizeWindow("morphologyEx x1", 1080,720);
    resizeWindow("morphologyEx x2", 1080,720);
    resizeWindow("canny", 1080,720);
    resizeWindow("dst", 1080,720);

    //【1】载入图像
    Mat src, img ;
    src = imread("D:/source/code/datasets/b113/images/20230329/19133_115_21_0910.bmp");
    Mat src_clone = src.clone();


    if(src.empty()){
        qDebug()<<"图片为空";
        return ;
    }
    imshow("src",src);
    // 转灰度图
    Mat gray;
    cvtColor(src,gray,COLOR_BGR2GRAY);
    // 直方图均衡化
    cv::equalizeHist(gray, src);

    // imshow("gray",src);

    //【3】图像二值化
    threshold(gray,gray,130,190,THRESH_BINARY);
    imshow("threshold",gray);

    //【4】执行形态学开操作去除噪点
    Mat kernel = getStructuringElement(MORPH_RECT,Size(15,15),Point(-1,-1));
    morphologyEx(gray,gray,MORPH_CLOSE,kernel,Point(-1,-1),1);
    imshow("morphologyEx x1",gray);

    //【4】执行形态学开操作去除噪点
    Mat kernel1 = getStructuringElement(MORPH_RECT,Size(10,10),Point(-1,-1));
    morphologyEx(gray,gray,MORPH_CLOSE,kernel1,Point(-1,-1),1);
    imshow("morphologyEx x2",gray);

    //【5】边缘检测
    Canny(gray,gray,0,255);
    imshow("canny",gray);

    //【6】轮廓发现
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> her;
    findContours(gray,contours,her,RETR_TREE,CHAIN_APPROX_SIMPLE);

    Mat srcImg = src;
    //拟合椭圆：fitEllipse(）
    std::vector<RotatedRect> box(contours.size());
    Point2f rect[4];
    for (int i = 0; i<contours.size(); i++)
    {
        Rect rect = boundingRect(contours[i]);

        Point2f pRadius;
        if(contours[i].size()>105){
            box[i] = fitEllipse(Mat(contours[i]));

            //条件过滤
            if( box[i].size.aspectRatio()<0.8||box[i].size.area()>10000000||rect.width<300 )
                continue;

            float majorAxis = std::max(box[i].size.width, box[i].size.height);

            rectangle(srcImg,rect,Scalar(0, 0, 255));

            ellipse(srcImg, box[i], Scalar(255, 0, 0), 1, 8);

            float x = rect.width/2.0;
            float y = rect.height/2.0;
            //【8】找出圆心并绘制
            pRadius=Point2f(rect.x+x,rect.y+y);

            cv::String det_info = cv::format("[%d] %.1f,%.1f(%dx%d),%.5f mm, %.5f mm",i,
                                             pRadius.x, pRadius.y, rect.width, rect.height,dbUnit*rect.width, dbUnit*majorAxis);


            cv::Point bbox_points;
            bbox_points = cv::Point(rect.x, rect.y);
            bbox_points = cv::Point(rect.x + det_info.size() * 11, rect.y);
            bbox_points = cv::Point(rect.x + det_info.size() * 11, rect.y - 15);
            bbox_points = cv::Point(rect.x, rect.y - 15);

            cv::putText(srcImg, det_info, bbox_points, cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            circle(srcImg,pRadius,1,Scalar(0,0,255),1);

            pRadius=box[i].center;
            circle(srcImg,pRadius,1,Scalar(255,0,0),1);

        }
    }
    // 绘制结果
    cv::imshow("dst", srcImg);
    // 保存结果
    // imwrite("dst.png", srcImg);

    cv::waitKey(0);
    cv::destroyAllWindows();


}

void MainWindow::videoTest()
{
    cv::VideoCapture cap(0); // 打开默认摄像头

    if (!cap.isOpened())
    {
        std::cerr << "Error: 无法打开摄像头" << std::endl;
            return ;
    }

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter video("output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width, frame_height));

    while (true) {
        cv::Mat frame;
        cap >> frame; // 读取帧

        if (frame.empty()) {
            std::cerr << "Error: 视频流结束" << std::endl;
                break;
        }

        video.write(frame); // 写入帧到视频文件

        cv::imshow("Video", frame);
        if (cv::waitKey(1) == 27) // 按下ESC键退出
            break;
    }

    cap.release();
    video.release();
    cv::destroyAllWindows();

    return ;

}



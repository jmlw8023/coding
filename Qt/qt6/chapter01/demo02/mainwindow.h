#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMouseEvent>
#include <QPushButton>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    // 鼠标按下
    void mousePressEvent(QMouseEvent *e);
    // 鼠标移动
    void mouseMoveEvent(QMouseEvent *e);
    // 鼠标释放
    void mouseReleaseEvent(QMouseEvent *e);

private slots:
    void on_login_btn_clicked();

private:
    Ui::MainWindow *ui;
    QPushButton *btClose;
    QPoint last;
};



//class QTimer;







#endif // MAINWINDOW_H

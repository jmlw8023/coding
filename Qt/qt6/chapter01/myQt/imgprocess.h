#ifndef IMGPROCESS_H
#define IMGPROCESS_H

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QWheelEvent>

class Imgprocess : public QGraphicsView
{
public:
    Imgprocess(QWidget *parent = nullptr) : QGraphicsView(parent)
    {
        scene = new QGraphicsScene(this);
        setScene(scene);
        setRenderHint(QPainter::SmoothPixmapTransform);
    }

    void setImage(const QPixmap &pixmap)
    {
        scene->clear();
        item = scene->addPixmap(pixmap);
        fitInView(item, Qt::KeepAspectRatio);
    }


protected:
    void wheelEvent(QWheelEvent *event) override
    {
        qreal scaleFactor = 1.15;
        if (event->angleDelta().y() < 0)
            scaleFactor = 1.0 / scaleFactor;
        scale(scaleFactor, scaleFactor);
    }

private:
    QGraphicsScene *scene;
    QGraphicsPixmapItem *item;


};

#endif // IMGPROCESS_H

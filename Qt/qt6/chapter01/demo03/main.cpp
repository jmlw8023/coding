#include <iostream>

#include <vector>

using namespace std;

void PrintVector(const vector<int> &v);


int main(int argc, char const *argv[])
{
     vector<int> v1;

     // 默认构造
     for (int i = 0; i < 10; i++)
     {
         // 插入元素
         v1.push_back(i);
     }

    PrintVector(v1);

    vector<int> v2(5, 10);


      PrintVector(v2);


    return 0;
}




void PrintVector( const vector<int> & v)
{
    cout << "start print : " ;
    for (auto x : v)
    {
        cout << x << " ";
    }
    cout << endl;

}



















//#include "mainwindow.h"

//#include <QApplication>

//int main(int argc, char *argv[])
//{



////    QApplication a(argc, argv);
////    MainWindow w;
////    w.show();
////    return a.exec();

//    return 0;
//}

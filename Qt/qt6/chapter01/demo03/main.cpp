#include <iostream>

#include <vector>
#include <list>

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

    cout <<  "v2.size() " << v2.size() << endl;
    cout << "v2.capacity() " << v2.capacity()  << endl;

    vector<int> v3(v1.begin(), v1.end());
    PrintVector(v3);
    v3.insert(v3.begin(), 22);
    PrintVector(v3);
    v3.insert(v3.begin(), 3, 100);
    // 尾插
    v3.push_back(88);
//    PrintVector(v3);
//    PrintVector(v1);
    // 交换两个vector 元素
    v3.swap(v1);

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

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

    // 5个10 方式构造
    vector<int> v4(5, 10);
    PrintVector(v4);

    // 区间方式构造
    vector<int> v2(v1.begin(), v1.end());
    PrintVector(v2);

    // 拷贝构造
    vector<int> v3(v1);
    PrintVector(v3);




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



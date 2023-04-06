#include <iostream>

#include <vector>

using namespace std;

void PrintVector(const vector<int> &v);


int main(int argc, char const *argv[])
{
    // vector<int> v1;

    // // 默认构造
    // for (int i = 0; i < 10; i++)
    // {
    //     // 插入元素
    //     v1.push_back(i);
    // }

    cout << "abc " << endl;

    vector<int> v2(5, 10);
    cout << "abc " << endl;

    // // PrintVector(v2);
    // for (auto x : v2)
    // {
    //     cout << x << " ";
    // }
    
    cout << "abc " << endl;


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



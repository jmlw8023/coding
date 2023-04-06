
#include <iostream>

using namespace std;

void test02()
{
    int a;
    double b, c;

    cout << "a -> " << a << " b-> " << b << " c -> " << endl;

    return;
}

void test01()
{
    cout << "fist cout" << endl;

    char str[] = "cpp demo";

    char *p = &str[1];


    cout << "p --> " << p << endl;
    cout << "&p --> " << &p << endl;
    cout << "p[3] --> " << p[3] << endl;
    // cout << "++p --> " << ++p << endl;
    // cout << "p++ --> " << p++ << endl;
    cout << "*p --> " << *p << endl;
    char * q = nullptr;
    q = p;

    cout << "&q --> " << &q << endl;
    cout << "q --> " << q << endl;
    /**
     * @brief 
        p --> pp demo
        &p --> 0x27557ff698
        p[3] --> d
        *p --> p
        &q --> 0x27557ff690
        q --> pp demo     
     * 
     */

    string name = "this is string name!!";

    cout << "name.size() = "  << name.size() << endl; 

    name.append("append add some");
    cout << "name = " << name << endl;
    cout << "name.size() = "  << name.size() << endl; 

    return;
}


# -*- encoding: utf-8 -*-
'''
@File    :   try_except_demo.py
@Time    :   2023/01/17 16:43:07
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :
'''

# import packets
# import os










# 文件内容读取
def test02(file='data.txt'):
    import time
    # 一次性读取整个文件内容
    with open(file, 'r', encoding='utf-8') as f:
        print(f.read())

    # 通过for-in循环逐行读取
    with open(file, mode='r') as f:
        for line in f:
            print(line, end='')
            time.sleep(0.5)
    print()

    # 读取文件按行读取到列表中
    with open(file) as f:
        lines = f.readlines()
    print(lines)



# 测试文件读取异常处理
def test(file='data.txt'):
    f = None

    try:
        f = open(file, 'r', encoding='utf-8')
        data = f.read()
        print(data)
    except FileNotFoundError as e:
        print('无法打开指定的文件!')
        print(e)        # [Errno 2] No such file or directory: 'data.txt'

    except LookupError as e:
        print('指定了未知的编码!')
        print(e)

    except UnicodeDecodeError as e:
        print('读取文件时解码错误!')        
        print(e)








def main():
    try:
        test()
    except IOError as e:
        print(e)
    
    try:
        test02()
    except IOError as e:
        print(e)

    pass
    




if __name__ == '__main__':
    main()
    




# -*- encoding: utf-8 -*-
'''
@File    :   argparse_use_demo.py
@Time    :   2023/01/18 09:28:11
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :
'''
#功能： 通过指定参数，或者在运行中增加参数作为输入变量
# import packets
# import os
import argparse



def parser_opt(argu=False):
    # 创建参数对象
    parser = argparse.ArgumentParser()
    data = 'This is my data!!!'

    # 默认参数
    parser.add_argument('--data', type=str, default=data, help='initial data message!')
    parser.add_argument('--number', type=int, default=1024, help='initial number')
    parser.add_argument('--msg', type=str, default='this is my message!!', help='initial number')

    # 默认为False， 在指定的运行中增加变量名，运行时变为True
    parser.add_argument('--name', action='store_true', help='Runtime increment --name, name will become valid(name=True)')

    # 输入时必须增加的参数变量
    #  不论指定类型，默认为None
    parser.add_argument('--req', required=False, help='Parameter that is not required for input!')
    parser.add_argument('--reqs', type=float, required=False, help='Parameter that is not required for input!')
    # parser.add_argument('--req', required=True, help='Parameter that must be added when entering')

    # 默认不生效变量

    # 从命令行中结构化解析参数
    opt = parser.parse_known_args()[0] if argu else parser.parse_args()

    # print(parser.parse_args())
    # Namespace(data='This is my data!!!', msg='this is my message!!', name=False, number=1024)
    # print('#' * 20)

    print(opt)
    print(opt.data)
    print(opt.number)
    print(opt.name)
    print(opt.req)


def main():
    argument = True
    parser_opt(argument)




if __name__ == '__main__':

    main()
    




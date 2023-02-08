# -*- encoding: utf-8 -*-
'''
@File    :   fillter_value.py
@Time    :   2023/02/08 12:20:19
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
import os

import random
import numpy as np

# 过滤掉list中的负数
def demo_list():
    # 创建随机数
    l = [np.random.randint(-10, 10) for _ in range(10)]
    print(l)
    # l2 = np.random.randint(-10, 10, 10).tolist()
    # print(l2)

    # 使用函数式
    res = [x for x in l if x >= 0]
    print(res)

    # 采用内置函数fitter
    res_iter = filter(lambda x: x >= 0, l)   # 迭代器对象
    res_list = list(res_iter)
    # print(res_iter) # 迭代器对象
    print(res_list)


def demo_dict():
    # s = str('num').zfill()
    d_dict = {'teacher%02d' % x : np.random.randint(60, 100) for x in range(1, 21)}
    print(d_dict)
    # print(type(d_dict))

    # 方法一
    res_dict = { k : v for k, v in d_dict.items() if v >= 88}
    print(res_dict)
    # 方法二
    res_dict_iter = filter(lambda item : item[1] >= 88, d_dict.items())
    print(res_dict_iter)
    res_dict_ = dict(res_dict_iter)
    print(res_dict_)


def demo_tuple():
    # 内置方法，进行键值提取
    from collections import namedtuple
    Stu = namedtuple('Student', ['name', 'age', 'sex', 'email', 'score'])
    s = Stu('jolly', 27, 'male', 'jmlw8023@163.com', 89.66)
    print(s)
    print('isinstance s is tuple = ', isinstance(s, tuple))
    print('s[0] = ', s[0], ', s[1] = ', s[1])
    print('s.name = ', s.name, ', s.age = ', s.age)
    

# 对字典中的value 进行比较排序
def sort_dict():
    # 创建字典
    d = {k : random.randint(60, 99) for k in 'absljuxtoaaaaswd'}    # 重复均可
    print(d)    # {'a': 80, 'b': 69, 's': 82, 'l': 64, 'j': 60, 'u': 62, 'x': 67, 't': 80, 'o': 86, 'w': 60, 'd': 80}
    print(len(d))

    # 方法一
    res_d = [(v, k) for k, v in d.items()]
    print(res_d)
    s = sorted(res_d, reverse=True)
    print(s)
    # 方法二
    res = list(zip(d.values(), d.keys()))
    print(res)
    print(sorted(res, reverse=True))
    # 方法三
    vres = sorted(d.items(), key=lambda item : item[1], reverse=True)
    print(vres)

    p_sort = list(enumerate(vres, 1))
    print(p_sort)

    # for i, (k, v) in enumerate(vres, 1):
    #     d[k] = (i, v)
    # print(sorted(d.items(), key=lambda item : item[1]))
    # print(dict(sorted(d.items(), key=lambda item : item[1])))

    # key : (排名, score)
    dd = {k : (i, v) for i, (k, v) in enumerate(vres, 1)}
    # {'o': (1, 98), 'a': (2, 97), 't': (3, 93), 'd': (4, 92), 'w': (5, 80), 's': (6, 77), 'b': (7, 76), 'l': (8, 75), 'j': (9, 71), 'u': (10, 68), 'x': (11, 60)} 
    print(dd)

# 序列中出现频次最高的元素
def most_frequent_element():
    # 创建重复元素
    data = [random.randint(0, 10) for _ in range(20)]
    print(data)
    # 依据list元素为键，创建值为 0 的字典
    d = dict.fromkeys(data, 0)
    # print(d)
    for x in data:
        d[x] += 1
    
    print(d)
    # 方法一： 排序取值
    # (次数, 原始值key)
    # res = sorted([(v, k) for  k, v in d.items() ], reverse=True)    # 列表解析
    res = sorted(((v, k) for  k, v in d.items() ), reverse=True)    # 生成器解析
    print(res)
    # print('出现次数最多的是：{k} 共{v}次！'.format(res[0][1], res[0][0]))
    k, v = res[0][1], res[0][0]
    print('出现次数最多的是：{} 共{}次！'.format(k, v))

    # 方法二： 使用堆数据结构
    import heapq
    # 只取前3个 排序最高的元素
    result = heapq.nlargest(3, ((v, k) for  k, v in d.items() ))
    print(result)
    k, v = result[0][1], result[0][0]
    print('出现次数最多的是：{} 共{}次！'.format(k, v))



def main():

    # demo_list()
    # demo_dict()
    # demo_tuple()
    # sort_dict()
    most_frequent_element()




    pass




if __name__ == '__main__':
    main()
    

# -*- encoding: utf-8 -*-
'''
@File    :   random_demo.py
@Time    :   2023/01/18 10:23:33
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :
'''
# 通过random、numpy 等模块进行随机数生成

# import packets
# import os
import random
import numpy as np




# random module
def test_random():
    # 随机产生一个[0.0, 1.0]之间的数
    print(random.random())
    
    # 产生指定范围内生成随机小数
    print(random.uniform(-10, 10))
    # 保留三位小数
    print(round(random.uniform(-10, 10), 3))

    # 生成指定范围的整数    randint (low, high) high会被取到值
    print(random.randint(-10, 10))
    print(random.randint(-10, 10))

    # 从随机序列中获取一个随机元素list、tuple
    lis = [3, -6, -9, 4, -7, 4, 0, -1]
    print(random.choice(lis))

    # 从指定序列中获取指定个数的随机元素。不改变原有序列
    print(random.sample(lis, 4))

    # 将一个list中的元素打乱，随机排列, 改变元素组
    print(random.shuffle(lis))  # None
    print(lis)



def demo_np_random():
    #  生成指定范围和个数 的整数
    nums = np.random.randint(-10, 10, 8)
    
    print(list(nums))   # 打印时候list 中多了, 号
    


# 随机生成6位随机数验证码   A verification code is generated randomly
'''
字符0-9对应的是48-57
字符A-Z对应的是65-90
字符a-z对应的是97-122
'''
def generater_verification(bit_num=6):

    lis = []
    for num in range(int(bit_num)):
        # randint (low, high) high会被取到值
        up_letter = random.randint(65, 90)
        low_letter = random.randint(97, 122)
        number = random.randint(0, 9)
        # # chr 生成对应的字符
        lis.append(chr(up_letter))
        lis.append(chr(low_letter))
        lis.append(number)
        # # chr 生成对应的字符
        # print(up_letter)
        # print(chr(up_letter))
        # print(low_letter)
        # print(chr(low_letter))
        # print(number)
    
    # 打乱lis 中的元素
    random.shuffle(lis)
    target_lis = random.sample(lis, int(bit_num))

    ver_str = ''
    for par in target_lis:
        ver_str += str(par)
    
    print('The generated {}-bit verification code is: '.format(int(bit_num)), ver_str)



if __name__ == '__main__':
    
    # test_random()
    
    # demo_np_random()
    # 3~8位的 随机位数的随机验证码
    # bit_num = 6
    bit_num = random.randint(3, 8)
    print('{}-bit verification code'.format(bit_num))
    generater_verification(bit_num)

    pass



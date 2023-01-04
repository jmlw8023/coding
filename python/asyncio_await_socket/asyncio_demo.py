# -*- encoding: utf-8 -*-
'''
@File    :   asyncio_demo.py
@Time    :   2023/01/04 09:15:14
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os
import time
import asyncio



# 普通方法
def display(args):
    time.sleep(1)
    print(args)


def funtion01():
    for num in range(10, 20):
        display(num)

# funtion01() # 需要十秒钟的时间打印完结果 range(10, 20)


# 使用异步io
# 在函数前标记 async 关键字，再以 await 关键字调用它，程序突然就变成异步的
async def display02(num):
    # await asyncio.sleep(1)
    asyncio.sleep(1)
    print(num)

def method01():
    values = [display02(number) for number in range(0, 11)]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(values))
    loop.close()


# method01()      # 异步只需一秒钟打印结果






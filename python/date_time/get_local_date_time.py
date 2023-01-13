# -*- encoding: utf-8 -*-
'''
@File    :   get_local_date_time.py
@Time    :   2023/01/13 14:34:06
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :
'''

# import packets
# import os
import time
import datetime
import calendar

import numpy as np


# 获取当前时间
# now = datetime.datetime.now()
# now = now.strftime('%Y_%m_%d_%H_%M')    # 设置格式
# print(now)

# now  = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') # 不同格式 2023_01_13_15_00_29
now  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # 不同格式 2023-01-13 15:01:05
# print(now)

# 获取昨天日期

today = datetime.date.today()           # 2023-01-13
# print(today)
one_day = datetime.timedelta(days=1)    # 指定 1天的时间， 1 day, 0:00:00
# print(one_day)
yesterday = today - one_day             # 2023-01-12
# print(yesterday)

year_struct = time.localtime(time.time())        #  .strftime('%Y')
# print(type(year_struct))                       # <class 'time.struct_time'>
# print(year_struct.tm_year)                     # 获取年
mouth_random = np.random.randint(1, 12, 1)       # 随机获取月份
print(mouth_random) 

# 显示日期
# test_calendar = calendar.month(year_struct.tm_year, int(mouth_random))
test_calendar = calendar.month(year_struct.tm_year, int(mouth_random))
print(test_calendar)
'''
    August 2023
Mo Tu We Th Fr Sa Su
    1  2  3  4  5  6
 7  8  9 10 11 12 13
14 15 16 17 18 19 20
21 22 23 24 25 26 27
28 29 30 31
'''
# 计算天数
x, mouth_nums = calendar.monthrange(year_struct.tm_year, int(mouth_random))
# print(x, mouth_nums)       # (0, 31)
print(int(mouth_random) , ' 月共有 {} 天！'.format(mouth_nums))     # 4  月共有 30 天！


# # 通过时间戳获取时间
# statu_time = 1643893140
# status_date = datetime.datetime.utcfromtimestamp(statu_time)
# # print(status_date)      # 2022-02-03 12:59:00







print()

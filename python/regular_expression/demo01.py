# -*- encoding: utf-8 -*-
'''
@File    :   demo01.py
@Time    :   2023/01/13 10:18:20
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :
'''

# import packets
# import os

import re

da_str = ['aegoiuawgv', 'aegaegjbgj', 'gwaifeae', 'wger']

# 匹配 'ae' 开头的
# result = re.match('ae', 'aegaegjbgj')

# 匹配 0 ~ 5
result = re.match('[0-5]', 'ae8ga1 486eg5jb 8g464j')

# 匹配任意一个字符
# result = re.match('..', 'aegaegjbgj')

res = result.group()


print(res)





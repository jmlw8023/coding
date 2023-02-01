# -*- encoding: utf-8 -*-
'''
@File    :   cnn.py
@Time    :   2023/02/01 15:34:39
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
# import os









# 计算Iou的值
'''
rect1: 坐标点1 left, top, right, bottom
rect2: 坐标点2 left, top, right, bottom
'''
def computer_IoU(rect1, rect2):
    # x0, y0, x1, y1 = left, top, right, bottom
    rect1_s = (rect1[3] - rect1[1]) * (rect1[2] - rect1[0])
    rect2_s = (rect2[3] - rect2[1]) * (rect2[2] - rect2[0])

    s_all = rect1_s + rect2_s
    left_limit = max(rect1[0], rect2[0])
    right_limit = min(rect1[2], rect2[2])
    top_limit = max(rect1[1], rect2[1])
    bottom_limit = max(rect1[3], rect2[3])

    if left_limit >= right_limit or top_limit >= bottom_limit:
        return 0
    else:
        intersect_s = (right_limit - left_limit) * (bottom_limit -top_limit)
        return (intersect_s / (s_all - intersect_s)) * 1.0




    










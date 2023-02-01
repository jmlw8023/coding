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




# 计算nms非极大值抑制
""" 步骤：
# 一、设置两个值：score阈值和IoU阈值
# 二、针对每一类对象，遍历该类所有的候选框，过滤掉score值低于score阈值候选框，并根据候选框类别进行类别概率排序：A<B<C<D<E<F
# 三、标记最大概率的矩形框是需要保留下来的候选框
# 四、从最大概率矩形框F分别计算A~E与F的交并比IoU是否大于IoU阈值，若A、C与F的重叠超过IoU阈值，则去除A、C
# 五、从剩下的B、D、E中，选择概率最大的E，标记为留下的候选框，判断B、D与E的重叠度，去除重叠度超过设定阈值的矩形框
# 六、照此以往，直至排除完所有的矩形框，并标记保留下来的矩形框
# 七、每一类处理完后，返回步骤二重新进行下一类对象
#  """
import numpy as np
def non_max_suppression_cpu(dets, thresh):
    # x0, y0 右上角坐标， x1、y1 左下角坐标
    x0 = dets[:, 2]
    y0 = dets[:, 3]
    x1= dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 4]

    # 单个候选框面积
    areas = (x0 - x1 + 1) * (y0 - y1 + 1)
    # 按照score排序，获取索引值
    index_order = scores.argsort()[::-1]

    keep = []
    while index_order.size > 0:
        i = index_order[0]
        keep.append(i)
        # 计算当前概率最大的矩形框与其他矩形框的相交框坐标
        xx1 = np.maximum(x1[i], x1[index_order[1:]])
        yy1 = np.maximum(y1[i], y1[index_order[1:]])
        xx0 = np.maximum(x0[i], x0[index_order[1:]])
        yy0 = np.maximum(y0[i], y0[index_order[1:]])


        # 计算相交框面积，不相交时为0
        w = np.maximum(0., xx0 - xx1 + 1)
        h = np.maximum(0., yy0 - yy1 + 1)
        intersect = w * h

        # 重叠IoU
        s_iou = intersect / (areas[i] + areas[index_order[1:]] - intersect)

        # 不高于阈值的矩形框索引
        index = np.where(s_iou < thresh)[0]
        # 更新序列
        index_order = index_order[index + 1]
    
    return keep












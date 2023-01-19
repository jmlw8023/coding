# -*- encoding: utf-8 -*-
'''
@File    :   augment_datasets.py
@Time    :   2023/01/19 14:30:57
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
import os
import math
import random
import numpy as np

import cv2 as cv











# 色域空间增强Augment colorspace
# def augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v']):
#     pass
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """用在LoadImagesAndLabels模块的__getitem__函数
    hsv色域增强  处理图像hsv，不对label进行任何处理
    :param img: 待处理图片  BGR [736, 736]
    :param hgain: h通道色域参数 用于生成新的h通道
    :param sgain: h通道色域参数 用于生成新的s通道
    :param vgain: h通道色域参数 用于生成新的v通道
    :return: 返回hsv增强后的图片 img
    """
    if hgain or sgain or vgain:
        # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数  用于生成新的hsv通道
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))  # 图像的通道拆分 h s v
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)         # 生成新的h通道
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 生成新的s通道
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 生成新的v通道

        # 图像的通道合并 img_hsv=h+s+v  随机调整hsv之后重新组合hsv通道
        # cv.LUT(hue, lut_hue)   通道色域变换 输入变换前通道hue 和变换后通道lut_hue
        img_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val)))
        # no return needed  dst:输出图像
        cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR, dst=img)  # no return needed  hsv->bgr



def translation_img():
    nL = len(labels)  # number of labels
    # 平移增强 随机左右翻转 + 随机上下翻转
    if augment:
        # 随机上下翻转 flip up-down
        if random.random() < hyp['flipud']:
            img = np.flipud(img)  # np.flipud 将数组在上下方向翻转。
            if nL:
                labels[:, 2] = 1 - labels[:, 2]   # 1 - y_center  label也要映射

        # 随机左右翻转 flip left-right
        if random.random() < hyp['fliplr']:
            img = np.fliplr(img)   # np.fliplr 将数组在左右方向翻转
            if nL:
                labels[:, 1] = 1 - labels[:, 1]   # 1 - x_center  label也要映射





def load_mosaic(self, index):
    """用在LoadImagesAndLabels模块的__getitem__函数 进行mosaic数据增强
    将四张图片拼接在一张马赛克图像中  loads images in a 4-mosaic
    :param index: 需要获取的图像索引
    :return: img4: mosaic和随机透视变换后的一张图片  numpy(640, 640, 3)
             labels4: img4对应的target  [M, cls+x1y1x2y2]
    """
    # labels4: 用于存放拼接图像（4张图拼成一张）的label信息(不包含segments多边形)
    # segments4: 用于存放拼接图像（4张图拼成一张）的label信息(包含segments多边形)
    labels4, segments4 = [], []
    s = self.img_size  # 一般的图片大小
    # 随机初始化拼接图像的中心点坐标  [0, s*2]之间随机取2个数作为拼接图像的中心坐标
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    # 从dataset中随机寻找额外的三张图像进行拼接 [14, 26, 2, 16] 再随机选三张图片的index
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    # 遍历四张图像进行拼接 4张不同大小的图像 => 1张[1472, 1472, 3]的图像
    for i, index in enumerate(indices):
        # load image   每次拿一张图片 并将这张图片resize到self.size(h,w)
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left  原图[375, 500, 3] load_image->[552, 736, 3]   hwc
            # 创建马赛克图像 [1472, 1472, 3]=[h, w, c]
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)   w=736  h = 552  马赛克图像：(x1a,y1a)左上角 (x2a,y2a)右下角
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)  图像：(x1b,y1b)左上角 (x2b,y2b)右下角
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将截取的图像区域填充到马赛克图像的相应位置   img4[h, w, c]
        # 将图像img的【(x1b,y1b)左上角 (x2b,y2b)右下角】区域截取出来填充到马赛克图像的【(x1a,y1a)左上角 (x2a,y2a)右下角】区域
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算pad(当前图像边界与马赛克边界的距离，越界的情况padw/padh为负值)  用于后面的label映射
        padw = x1a - x1b   # 当前图像与马赛克图像在w维度上相差多少
        padh = y1a - y1b   # 当前图像与马赛克图像在h维度上相差多少

        # labels: 获取对应拼接图像的所有正常label信息(如果有segments多边形会被转化为矩形label)
        # segments: 获取对应拼接图像的所有不正常label信息(包含segments多边形也包含正常gt)
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            # normalized xywh normalized to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)      # 更新labels4
        segments4.extend(segments)  # 更新segments4

    # Concat/clip labels4 把labels4（[(2, 5), (1, 5), (3, 5), (1, 5)] => (7, 5)）压缩到一起
    labels4 = np.concatenate(labels4, 0)
    # 防止越界  label[:, 1:]中的所有元素的值（位置信息）必须在[0, 2*s]之间,小于0就令其等于0,大于2*s就等于2*s   out: 返回
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

    # 测试代码  测试前面的mosaic效果
    # cv.imshow("mosaic", img4)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print(img4.shape)   # (1280, 1280, 3)

    # 随机偏移标签中心，生成新的标签与原标签结合 replicate
    # img4, labels4 = replicate(img4, labels4)
    #
    # # 测试代码  测试replicate效果
    # cv.imshow("replicate", img4)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print(img4.shape)   # (1280, 1280, 3)

    # Augment
    # random_perspective Augment  随机透视变换 [1280, 1280, 3] => [640, 640, 3]
    # 对mosaic整合后的图片进行随机旋转、平移、缩放、裁剪，透视变换，并resize为输入大小img_size
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    # 测试代码 测试mosaic + random_perspective随机仿射变换效果
    # cv.imshow("random_perspective", img4)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print(img4.shape)   # (640, 640, 3)

    return img4, labels4




def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1,
                       scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """这个函数会用于load_mosaic中用在mosaic操作之后
    随机透视变换  对mosaic整合后的图片进行随机旋转、缩放、平移、裁剪，透视变换，并resize为输入大小img_size
    :params img: mosaic整合后的图片img4 [2*img_size, 2*img_size]
    如果mosaic后的图片没有一个多边形标签就使用targets, segments为空  如果有一个多边形标签就使用segments, targets不为空
    :params targets: mosaic整合后图片的所有正常label标签labels4(不正常的会通过segments2boxes将多边形标签转化为正常标签) [N, cls+xyxy]
    :params segments: mosaic整合后图片的所有不正常label信息(包含segments多边形也包含正常gt)  [m, x1y1....]
    :params degrees: 旋转和缩放矩阵参数
    :params translate: 平移矩阵参数
    :params scale: 缩放矩阵参数
    :params shear: 剪切矩阵参数
    :params perspective: 透视变换参数
    :params border: 用于确定最后输出的图片大小 一般等于[-img_size, -img_size] 那么最后输出的图片大小为 [img_size, img_size]
    :return img: 通过透视变换/仿射变换后的img [img_size, img_size]
    :return targets: 通过透视变换/仿射变换后的img对应的标签 [n, cls+x1y1x2y2]  (通过筛选后的)
    """
    # 设定输出图片的 H W
    # border=-s // 2  所以最后图片的大小直接减半 [img_size, img_size, 3]
    height = img.shape[0] + border[0] * 2  # # 最终输出图像的H
    width = img.shape[1] + border[1] * 2   # 最终输出图像的W

    # ============================ 开始变换 =============================
    # 需要注意的是，其实opencv是实现了仿射变换的, 不过我们要先生成仿射变换矩阵M
    # Center 设置中心平移矩阵
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective  设置透视变换矩阵
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale  设置旋转和缩放矩阵
    R = np.eye(3)    # 初始化R = [[1,0,0], [0,1,0], [0,0,1]]    (3, 3)
    # a: 随机生成旋转角度 范围在(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    # s: 随机生成旋转后图像的缩放比例 范围在(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    # cv.getRotationMatrix2D: 二维旋转缩放函数
    # 参数 angle:旋转角度  center: 旋转中心(默认就是图像的中心)  scale: 旋转后图像的缩放比例
    R[:2] = cv.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear  设置剪切矩阵
    S = np.eye(3)  # 初始化T = [[1,0,0], [0,1,0], [0,0,1]]
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation 设置平移矩阵
    T = np.eye(3)  # 初始化T = [[1,0,0], [0,1,0], [0,0,1]]    (3, 3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix  @ 表示矩阵乘法  生成仿射变换矩阵M
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    # 将仿射变换矩阵M作用在图片上
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            # 透视变换函数  实现旋转平移缩放变换后的平行线不再平行
            # 参数和下面warpAffine类似
            img = cv.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            # 仿射变换函数  实现旋转平移缩放变换后的平行线依旧平行
            # image changed  img  [1472, 1472, 3] => [736, 736, 3]
            # cv.warpAffine: opencv实现的仿射变换函数
            # 参数： img: 需要变化的图像   M: 变换矩阵  dsize: 输出图像的大小  flags: 插值方法的组合（int 类型！）
            #       borderValue: （重点！）边界填充值  默认情况下，它为0。
            img = cv.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize 可视化
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    # 同样需要调整标签信息
    n = len(targets)
    if n:
        # 判断是否可以使用segment标签: 只有segments不为空时即数据集中有多边形gt也有正常gt时才能使用segment标签 use_segments=True
        #                          否则如果只有正常gt时segments为空 use_segments=False
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))  # [n, 0+0+0+0]
        # 如果使用的是segments标签(标签中含有多边形gt)
        if use_segments:  # warp segments
            # 先对segment标签进行重采样
            # 比如说segment坐标只有100个，通过interp函数将其采样为n个(默认1000)
            # [n, x1y2...x99y100] 扩增坐标-> [n, 500, 2]
            # 由于有旋转，透视变换等操作，所以需要对多边形所有角点都进行变换
            segments = resample_segments(segments)
            for i, segment in enumerate(segments):  # segment: [500, 2]  多边形的500个点坐标xy
                xy = np.ones((len(segment), 3))   # [1, 1+1+1]
                xy[:, :2] = segment  # [500, 2]
                # 对该标签多边形的所有顶点坐标进行透视/仿射变换
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # 根据segment的坐标，取xy坐标的最大最小值，得到边框的坐标  clip
                new[i] = segment2box(xy, width, height)  # xy [500, 2]
        # 不使用segments标签 使用正常的矩形的标签targets
        else:  # warp boxes
            # 直接对box透视/仿射变换
            # 由于有旋转，透视变换等操作，所以需要对四个角点都进行变换
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform 每个角点的坐标
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip  去除太小的target(target大部分跑到图外去了)
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates  过滤target 筛选box
        # 长和宽必须大于wh_thr个像素 裁剪过小的框(面积小于裁剪前的area_thr)  长宽比范围在(1/ar_thr, ar_thr)之间的限制
        # 筛选结果 [n] 全是True或False   使用比如: box1[i]即可得到i中所有等于True的矩形框 False的矩形框全部删除
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        # 得到所有满足条件的targets
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


























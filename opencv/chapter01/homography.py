# -*- encoding: utf-8 -*-
'''
@File    :   homography.py
@Time    :   2023/02/15 11:12:52
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
import os
import numpy as np
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置中文
mpl.rcParams['font.sans-serif'] = ['STSong']
# 显示负号
mpl.rcParams['axes.unicode_minus'] = False

img_root = r'../data/images'
img_path1 = os.path.join(img_root, 'milk1.jpg')
img_path2 = os.path.join(img_root, 'milk2.jpg')





im1 = cv.imread(img_path1)
im2 = cv.imread(img_path2)


# src_points = np.array([[581, 297], [1053, 173], [1041, 895], [558, 827]])
# dst_points = np.array([[571, 257], [963, 333], [965, 801], [557, 827]])

# H, _ = cv.findHomography(src_points, dst_points)

# h, w = im2.shape[:2]

# im2_warp = cv.warpPerspective(im2, H, (w, h))


# 计算SURF特征点和对应的描述子，kp存储特征点坐标，des存储对应描述子
surf = cv.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(im1, None)
kp2, des2 = surf.detectAndCompute(im2, None)

# 匹配特征点描述子
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 提取匹配较好的特征点
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# 通过特征点坐标计算单应性矩阵H
# （findHomography中使用了RANSAC算法剔除错误匹配）
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

# 使用单应性矩阵计算变换结果并绘图
h, w, d = im1.shape
pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
dst = cv.perspectiveTransform(pts, H)
img2 = cv.polylines(im2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

im3 = cv.drawMatches(im1, kp1, im2, kp2, good, None, **draw_params)



# img1 = plt.imread(img_path1)
# img2 = plt.imread(img_path2)

fig, ax = plt.subplots(2, 3, figsize=(10, 8))

# plt.imshow(img1)
# ax[0][0].imshow(img1)
# ax[0][1].imshow(img2)
ax[0][0].imshow(im1)
ax[0][1].imshow(img2)
ax[0][2].imshow(im3)

plt.show()


















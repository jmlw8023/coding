import cv2 as cv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STSong']
# 可以显示负号
mpl.rcParams['axes.unicode_minus'] = False


cols = 224
rows = 224
channel = 3

# 设置画布  2行2列
# fig, ax = plt.subplots(2, 2, figsize=(6, 5)) 

# 设置画布  3行3列
fig, ax = plt.subplots(3, 3, figsize=(6, 5)) 


img_path = r'../data/images/milk1.jpg'
img = cv.imread(img_path)

# assert (img is not None), 'read image error!' # 判断 中断
if img is None:
    img = np.random.randint(0, 256, (224, 224, 3))  # 随机生成

w, h, c = img.shape
print(img.shape)
print(img)

im = img.copy()

# im[0] = w - img[0]
im[:, :, 1] = 0
# im[1] = 0
# im[2] = 0


# img.resize(380, 200, 3)
# 第二行第一列
ax[1][0].imshow(img[:, :, [2, 1, 0]])
# 第一行第二列
ax[0][1].imshow(im[:, :, [2, 1, 0]])

# 
# ax[2][2].imshow(img)
r, g, b = cv.split(img)     # 分裂通道
ax[2, 0].imshow(r)
ax[2, 1].imshow(g)
# ax[2, 2].imshow(b)

m = np.random.randint(0, 256, (rows, cols)).astype(np.uint8)

image = cv.cvtColor(m, cv.COLOR_GRAY2BGR)
# ax[2, 2].imshow(image)

image[:, :, :] = 0      # 全黑
# image[:, :, 0] = 255     # 红色
# image[:, :, 1] = 255     # 绿色
# image[:, :, 2] = 255     # 蓝色
image[image.shape[0]//4:image.shape[0]//2:, image.shape[1]//4:image.shape[1]//2, :] = 255     # 白色

ax[2, 2].imshow(image)
# print(im.shape)

plt.show()







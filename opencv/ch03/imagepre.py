import cv2 as cv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STSong']
# 可以显示负号
mpl.rcParams['axes.unicode_minus'] = False


# 设置画布
fig, ax = plt.subplots(2, 2, figsize=(6, 5)) 



img_path = r'../data/images/milk1.jpg'


img = cv.imread(img_path)

w, h, c = img.shape
print(img.shape)
print(img)

im = img.copy()

# im[0] = w - img[0]
im[:, :, 1] = 0
# im[1] = 0
# im[2] = 0




# img.resize(380, 200, 3)
ax[1][0].imshow(img[:, :, [2, 1, 0]])
ax[0][1].imshow(im[:, :, [2, 1, 0]])


print(im.shape)



plt.show()







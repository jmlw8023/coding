# -*- encoding: utf-8 -*-
'''
@File    :   get_image_time.py
@Time    :   2023/02/06 09:24:56
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''

# import packets
import os

import cv2 as cv
import matplotlib.pyplot as plt



def plt_show_img(img):

    if img.dim == 2:

    

def get_rpn_area(img_path):

    if os.path.isfile(img_path):

        img = cv.imread(img_path)

        w, h, d = img.shape
        # print(w, h, d)  #   610 808 3









def main():

    # root_path = r'./images'
    root_path = r'E:\w\qun\datasets\res'
    file_name = r'trucks_00001.png'

    img_path = os.path.join(root_path, file_name)

    get_rpn_area(img_path)


    pass
    



if __name__ == '__main__':
    main()
    






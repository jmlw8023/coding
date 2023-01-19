# -*- encoding: utf-8 -*-
'''
@File    :   argmentor.py
@Time    :   2023/01/19 15:14:03
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :  https://github.com/jmlw8023/coding
'''
# 参考： https://github.com/mdbloice/Augmentor
# import packets
import os

import Augmentor






def demo01(root_path = r'./data/images', xml_path = r'./data/Annotatios'):

    img_path = os.path.join(root_path, 'bus.jpg')

    p = Augmentor.Pipeline(root_path)
    p.ground_truth(xml_path)

    # 旋转
    # p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=10)

    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)

    # 放大缩小
    # p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)


    # 生成的图片个数
    # p.sample(3)
    p.sample(5, multi_threaded=False)
    # p.process()



def demo02(image_path = r'./data/images', xml_path = r'./data/Annotatios'):

    # img_path = os.path.join(root_path, 'bus.jpg')
    

    p = Augmentor.Pipeline(image_path)

    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)

    p.sample(10)

    # p = Augmentor.DataPipeline(img_path, y)
    # p.rotate(1, max_left_rotation=5, max_right_rotation=5)
    # p.flip_top_bottom(0.5)
    # p.zoom_random(1, percentage_area=0.5)

    # augmented_images, labels = p.sample(100)


def demo03(image_path = r'./data/images', xml_path = r'./data/Annotatios'):

    p = Augmentor.Pipeline(image_path)

    # 旋转
    # p.rotate90(probability=0.5)
    # p.rotate270(probability=0.5)

    # 上下翻转
    # p.flip_left_right(probability=0.8)
    # p.flip_top_bottom(probability=0.6)

    # 指定区域进行裁剪 
    # p.crop_centre(probability=0.8, percentage_area=0.6)
    # 随机裁剪
    p.crop_random(probability=1, percentage_area=0.5)

    p.resize(probability=1, width=1040, height=840)


    p.sample(10)



def demo04(image_path = r'./data/images', xml_path = r'./data/Annotatios'):
    from urllib.request import urlretrieve

    img_url = r'https://isic-archive.com:443/api/v1/image/5436e3abbae478396759f0cf/download'
    urlretrieve(img_url, "ISIC_0000000.jpg")
    






if __name__ == '__main__':
    
    root_path = r'../paper/data/images'
    xml_path = r'../paper/data/Annotatios'
    
    # demo01(root_path, xml_path)
    # demo02(root_path, xml_path)
    demo03(root_path, xml_path)
    # demo04()







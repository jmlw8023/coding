#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   image_format_conver.py
@Time    :   2024/05/17 09:56:16
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module
import os






# bmp转为png图像
def bmp_to_png(bmp_folder, png_save_folder):
    import os
    import time
    from PIL import Image
    
    # os.makedirs(png_save_folder, exist_ok=True)
    assert os.path.isdir(png_save_folder), 'save folder is not folder'
    # 文件
    if os.path.isfile(bmp_folder):
        save_path = os.path.join(png_save_folder, os.path.basename(bmp_folder)[:-4] + '.png')
        with Image.open(bmp_to_png) as img:
            
            img.save(save_path, format='PNG')
            print('convert {} to png format to {} file'.format(bmp_folder, png_save_folder))
    # 文件夹
    elif os.path.isdir(bmp_folder):
        cont = 0
        for root, dirs, files in os.walk(bmp_folder):
            for file in files:
                if file.lower().endswith('.bmp'):
                    # print(os.path.join(root, file))
                    name_path = os.path.join(root, file)
                    save_path = os.path.join(png_save_folder, file[:-3]+'png')
       
                    with Image.open(name_path, 'r') as img:
                        time.sleep(0.3)
                        img.save(save_path, format='PNG', quality=90)
                    cont += 1
                    print('convert {} to png format to {} file'.format(file, png_save_folder))

        print('A total of {} files were convered!'.format(cont))
















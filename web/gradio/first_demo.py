#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   first_demo.py
@Time    :   2024/06/14 09:15:07
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module

import gradio as gr     # pip install gradio  -i https://pypi.tuna.tsinghua.edu.cn/simple

# link: https://www.gradio.app/docs/gradio



  
# def multiply(x, y):  
#     return x * y  
  
# iface = gr.Interface(fn=multiply, inputs=["number", "number"], outputs="number")  
# iface.launch()


def image_classifier(inp):
    return {'cat': 0.3, 'dog': 0.7}

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch()





# -*- encoding: utf-8 -*-
'''
@File    :   jolly.py
@Time    :   2022/10/26 16:40:43
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets
import os
import cv2 as cv
import functools
from flask import Blueprint, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename




jolly = Blueprint('jolly', __name__)


# basedir = os.path.abspath(os.path.dirname(__file__))



@jolly.route('/index')
def index():
    name = 'Gibs'
    return render_template('index.html', name=name)

@jolly.route('/home', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':
        f = request.files.get('file')

        # print((f.filename))
        print(os.path.dirname(__file__))




    return render_template('home.html')

@jolly.route('/start')
def start():

    return render_template('start.html')


@jolly.route('/test')
def test():

    return render_template('test.html')




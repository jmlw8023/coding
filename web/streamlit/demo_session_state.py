#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   demo_session_state.py
@Time    :   2024/06/04 17:24:32
@Author  :   hgh 
@Version :   1.0
@Desc    :    多页面
'''

# import module
import os
import streamlit as st

def page_home():
    st.title('Home Page')
    # 在Home页面中显示数据和功能组件

def page_demo():
    st.title('demo Page')
    # 在demo页面中显示数据和功能组件

def main():
    # 设置初始页面为Home
    session_state = st.session_state
    if 'page' not in session_state:
        session_state['page'] = 'Home'

    # 导航栏
    page = st.sidebar.radio('Navigate', ['Home', 'demo'])

    if page == 'Home':
        page_home()
    elif page == 'demo':
        page_demo()



if __name__ == '__main__':
    main()



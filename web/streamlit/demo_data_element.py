#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   demo_data_element.py
@Time    :   2024/07/02 14:23:42
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module
import os
import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image


def test_image():
    
    im = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    # r = np.random.randint(0, 255, (224, 224)).astype(np.float32)
    # g = np.random.randint(0, 255, (224, 224)).astype(np.float32)
    # b = np.random.randint(0, 255, (224, 224)).astype(np.float32)
    # im2 = cv.merge((b, g, r))
    
    # alpha = 0.5  
    # beta = 1.0 - alpha  # 或者直接写0.5，因为alpha+beta通常设为1  
    # gamma = 0.3 
    # im = cv.addWeighted(im1, alpha, im2, beta, gamma)   # cv.addWeighted进行加权叠加  
    
    # rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    # print(rgb_im.shape)
    # cv.imshow('image', rgb_im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # pillow_image = Image.fromarray(rgb_im) 
    # pillow_image = Image.fromarray(im[..., ::-1]) 
    pillow_image = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB)) 
    
    # pillow_image.show('demo')
    st.image(pillow_image, caption='random create image', width=320)


def test_pd():
    df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))

    # st.dataframe(df)
    
    st.dataframe(df.style.highlight_min(axis=0))


def test_editer():
    # df = pd.DataFrame(
    #     [
    #     {"command": "st.selectbox", "rating": 4, "is_widget": True},
    #     {"command": "st.balloons", "rating": 5, "is_widget": False},
    #     {"command": "st.time_input", "rating": 3, "is_widget": True},
    # ]
    # )
    # edited_df = st.data_editor(df)

    # favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
    # st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

    data_df = pd.DataFrame(
        {
            "category": [
                "📊 Data Exploration",
                "📈 Data Visualization",
                "🤖 LLM",
                "📊 Data Exploration",
            ],
        }
    )

    st.data_editor(
        data_df,
        column_config={
            "category": st.column_config.SelectboxColumn(
                "App Category",
                help="The category of the app",
                width="medium",
                options=[
                    "📊 Data Exploration",
                    "📈 Data Visualization",
                    "🤖 LLM",
                ],
            )
        },
        hide_index=True,
    )


def main():
    
    # test_pd()
    # test_editer()
    test_image()
   
    pass


if __name__ == '__main__':
    
    main()
    
    pass







#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   test_qwen2.py
@Time    :   2024/07/12 17:44:11
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module
import os

import numpy as np
# import matplotlib.pyplot as plt

import torch

import transformers     # 



from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen2-7B-Instruct'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_type='auto',
        device_map='auto'
        
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)












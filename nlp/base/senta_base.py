

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   senta_base.py
@Time    :   2024/04/15 19:24:42
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# import module

# pip install paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple
# 
# hub install senta_cnn
#  pip install Senta -i https://pypi.tuna.tsinghua.edu.cn/simple 


import paddlehub as hub


test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
# ernie = hub.Module(name='ernie')


name = 'senta_lstm' # ['senta_bow', 'senta_cnn', 'senta_lstm', 'senta_gru', 'senta_bilstm']
senta = hub.Module(name="senta_bow")
results = senta.sentiment_classify(texts=test_text, 
                                   use_gpu=False,
                                   batch_size=1)
print(len(results))
print('#'*30)
  
for result in results:
    print(result['text'])
    print(result['sentiment_label'])
    print(result['sentiment_key'])
    print(result['positive_probs'])
    print(result['negative_probs'])
    print('-'*30)
# 这家餐厅很好吃 1 positive 0.7902 0.2098
# 这部电影真的很差劲 0 negative 0.0343 0.9657










# from senta import Senta

# my_senta = Senta()

# # 获取目前支持的情感预训练模型, 我们开放了以ERNIE 1.0 large(中文)、ERNIE 2.0 large(英文)和RoBERTa large(英文)作为初始化的SKEP模型
# print(my_senta.get_support_model()) # ["ernie_1.0_skep_large_ch", "ernie_2.0_skep_large_en", "roberta_skep_large_en"]

# # 获取目前支持的预测任务
# print(my_senta.get_support_task()) # ["sentiment_classify", "aspect_sentiment_classify", "extraction"]

# # 选择是否使用gpu
# use_cuda = True # 设置True or False

# # 预测中文句子级情感分类任务
# my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="sentiment_classify", use_cuda=use_cuda)
# texts = ["中山大学是岭南第一学府"]
# result = my_senta.predict(texts)
# print(result)

# # 预测中文评价对象级的情感分类任务
# my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="aspect_sentiment_classify", use_cuda=use_cuda)
# texts = ["百度是一家高科技公司"]
# aspects = ["百度"]
# result = my_senta.predict(texts, aspects)
# print(result)

# # 预测中文观点抽取任务
# my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="extraction", use_cuda=use_cuda)
# texts = ["唐 家 三 少 ， 本 名 张 威 。"]
# result = my_senta.predict(texts, aspects)
# print(result)

# # 预测英文句子级情感分类任务（基于SKEP-ERNIE2.0模型）
# my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="sentiment_classify", use_cuda=use_cuda)
# texts = ["a sometimes tedious film ."]
# result = my_senta.predict(texts)
# print(result)

# # 预测英文评价对象级的情感分类任务（基于SKEP-ERNIE2.0模型）
# my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="aspect_sentiment_classify", use_cuda=use_cuda)
# texts = ["I love the operating system and the preloaded software."]
# aspects = ["operating system"]
# result = my_senta.predict(texts, aspects)
# print(result)

# # 预测英文观点抽取任务（基于SKEP-ERNIE2.0模型）
# my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="extraction", use_cuda=use_cuda)
# texts = ["The JCC would be very pleased to welcome your organization as a corporate sponsor ."]
# result = my_senta.predict(texts)
# print(result)

# # 预测英文句子级情感分类任务（基于SKEP-RoBERTa模型）
# my_senta.init_model(model_class="roberta_skep_large_en", task="sentiment_classify", use_cuda=use_cuda)
# texts = ["a sometimes tedious film ."]
# result = my_senta.predict(texts)
# print(result)

# # 预测英文评价对象级的情感分类任务（基于SKEP-RoBERTa模型）
# my_senta.init_model(model_class="roberta_skep_large_en", task="aspect_sentiment_classify", use_cuda=use_cuda)
# texts = ["I love the operating system and the preloaded software."]
# aspects = ["operating system"]
# result = my_senta.predict(texts, aspects)
# print(result)

# # 预测英文观点抽取任务（基于SKEP-RoBERTa模型）
# my_senta.init_model(model_class="roberta_skep_large_en", task="extraction", use_cuda=use_cuda)
# texts = ["The JCC would be very pleased to welcome your organization as a corporate sponsor ."]
# result = my_senta.predict(texts)
# print(result)







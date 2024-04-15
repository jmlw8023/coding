
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   textblob_test.py
@Time    :   2024/04/15 13:20:11
@Author  :   hgh 
@Version :   1.0
@Desc    :    
'''

# pip install    -i https://pypi.tuna.tsinghua.edu.cn/simple 


# textBlob 官方文档： https://textblob.readthedocs.io/en/latest/quickstart.html

# import module

from textblob import TextBlob






# 基于情感词典的方法
def word_dict_method():

    import jieba
    import pandas as pd
    # 加载情感词典
    posdict = pd.read_excel('positive_words.xlsx', header=None)[0].tolist()
    negdict = pd.read_excel('negative_words.xlsx', header=None)[0].tolist()
    # 分词
    text = '今天天气真好，心情非常愉快。'
    words = jieba.lcut(text)
    # 计算情感得分
    poscount = 0
    negcount = 0
    for word in words:
        if word in posdict:
            poscount += 1
        elif word in negdict:
            negcount += 1
    score = (poscount - negcount) / len(words)
    print(score)


# 机器学习方法
def ml_method():
    import jieba
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    # 加载训练数据
    posdata = pd.read_excel('positive_data.xlsx', header=None)[0].tolist()
    negdata = pd.read_excel('negative_data.xlsx', header=None)[0].tolist()
    data = posdata + negdata
    labels = [1] * len(posdata) + [0] * len(negdata)
    # 分词
    words = [' '.join(jieba.lcut(text)) for text in data]
    # 特征提取
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(words)
    # 训练分类器
    clf = MultinomialNB()
    clf.fit(X, labels)
    # 预测情感
    text = '今天天气真好，心情非常愉快。'
    test_X = vectorizer.transform([' '.join(jieba.lcut(text))])
    score = clf.predict_proba(test_X)[0][1]
    print(score)



def dl_method():
    import jieba
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
    # 加载训练数据
    posdata = pd.read_excel('positive_data.xlsx', header=None)[0].tolist()
    negdata = pd.read_excel('negative_data.xlsx', header=None)[0].tolist()
    data = posdata + negdata
    labels = [1] * len(posdata) + [0] * len(negdata)
    # 分词
    words = [jieba.lcut(text) for text in data]
    # 构建词向量
    word2vec = {}
    with open('sgns.weibo.bigram', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            vec = [float(x) for x in line[1:]]
            word2vec[word] = vec
    embedding_matrix = []
        # 特征提取
    # vectorizer = 
    for word in vectorizer.get_feature_names():
        if word in word2vec:
            embedding_matrix.append(word2vec[word])
        else:
            embedding_matrix.append([0] * 300)
    # 构建模型
    model = Sequential()
    model.add(Embedding(len(vectorizer.get_feature_names()), 300, weights=[embedding_matrix], input_length=100))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # 训练模型
    X = vectorizer.transform([' '.join(words[i][:100]) for i in range(len(words))]).toarray()
    model.fit(X, labels, epochs=10, batch_size=32)
    # 预测情感
    text = '今天天气真好，心情非常愉快。'
    test_X = vectorizer.transform([' '.join(jieba.lcut(text)[:100])]).toarray()
    score = model.predict(test_X)[0][0]
    print(score)



# 基于知识图谱
def knowledge_method():
    
    import jieba
    import pandas as pd
    from pyhanlp import *
    # 加载情感知识图谱
    graph = pd.read_excel('emotion_graph.xlsx')
    # 分词
    text = '今天天气真好，心情非常愉快。'
    words = jieba.lcut(text)
    # 计算情感得分
    poscount = 0
    negcount = 0
    for word in words:
        if word in graph['词语'].tolist():
            index = graph[graph['词语'] == word].index[0]
            if graph.loc[index, '情感分类'] == '正面':
                poscount += 1
            elif graph.loc[index, '情感分类'] == '负面':
                negcount += 1
    score = (poscount - negcount) / len(words)
    print(score)


# 情感神经网络
def netword_method():
    import jieba
    import pandas as pd
    import numpy as np
    from keras.models import load_model
    # 加载情感神经网络
    model = load_model('emotion_network.h5')
    # 加载情感词典
    posdict = pd.read_excel('positive_words.xlsx', header=None)[0].tolist()
    negdict = pd.read_excel('negative_words.xlsx', header=None)[0].tolist()
    # 分词
    text = '今天天气真好，心情非常愉快。'
    words = jieba.lcut(text)
    # 构建输入向量
    X = np.zeros((1, len(words)))
    for i, word in enumerate(words):
        if word in posdict:
            X[0, i] = 1
        elif word in negdict:
            X[0, i] = -1
    # 预测情感
    score = model.predict(X)[0][0]
    print(score)


# txt = '很想知道这个会怎么评价，分析这个句话的情感是什么？'


# text = TextBlob(txt)

# print(text.tags)





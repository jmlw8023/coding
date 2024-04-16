
# link: https://gitee.com/hansermarblue/dazhongdianping/blob/master/daZhongFood/Emotion/Emotion_model.py


# author:HanserMarBlue
# Date:2024/3/11 20:40
import re
import jieba
import emoji
import itertools
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import imageio
FILE_PATH = "C:\\Users\\86180\\Desktop\\DaZhongdianpingbishe\\daZhongFood\\data\\training_data.xlsx"
STOP_WORDS = "C:\\Users\\86180\\Desktop\\DaZhongdianpingbishe\\daZhongFood\\stopwords\\cn_stopwords.txt"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel(FILE_PATH)
data.head()


# 去除表情，数字，符号
def remove_emojis(text):
    text = emoji.replace_emoji(text, '')
    return text


def remove_digits(text):
    text = re.sub(r'\d+', '', text)
    return text


def remove_symbols(text):
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'  # remove symbol but not #
    text = re.sub('[' + my_punctuation + ']+', ' ', text)
    return text


def clean_word(text):
    text = remove_emojis(text)
    text = remove_digits(text)
    text = remove_symbols(text)

    return text


data['clean_text'] = data['comment'].apply(clean_word)

with open(STOP_WORDS, 'r', encoding='utf-8') as f:
    stop = f.read()
stop = stop.split()
stop = [' ', '', 'text', '☆', '×', '…', '\n'] + stop

# 分词和去除停用词
data['words_list'] = data['clean_text'].apply(jieba.lcut).apply(lambda x: [i for i in x if i not in stop])


def get_wc(df, filename):
    num = pd.Series(list(itertools.chain(*list(df)))).value_counts()
    pic = imageio.imread('C:\\Users\\86180\\Desktop\\DaZhongdianpingbishe\\daZhongFood\\image\\4.png')
    wc = WordCloud(font_path='C:\\Users\\86180\\Desktop\\DaZhongdianpingbishe\\daZhongFood\\Emotion/SimHei.ttf', background_color='White', mask=pic)
    wc2 = wc.fit_words(num)
    plt.imshow(wc2)
    plt.axis('off')
    plt.savefig(filename)
    plt.show()


get_wc(data['words_list'], '词云图.png')



def get_top_10(df):
    p = df[df['target'] == 0]
    n = df[df['target'] == 1]
    p_num = pd.Series(list(itertools.chain(*list(p['words_list'])))).value_counts()
    n_num = pd.Series(list(itertools.chain(*list(n['words_list'])))).value_counts()
    ptop10 = p_num.head(10)
    ntop10 = n_num.head(10)
    pt = pd.DataFrame({'words': ptop10.index, 'counts': ptop10.values})
    nt = pd.DataFrame({'words': ntop10.index, 'counts': ntop10.values})

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sub = sns.barplot(x='words', y='counts', data=pt, color='lightblue', ax=ax[0])
    sub.set_title("积极评价前10词")
    sub = sns.barplot(x='words', y='counts', data=nt, color='lightblue', ax=ax[1])
    sub.set_title("消极评价前10词")
    plt.savefig("top10.png")
    plt.show()
    return p_num.head(10), n_num.head(10)


ptop10, ntop10 = get_top_10(data)



def get_top_seller(df, target, filename):
    tmp = df[df['target'] == target]
    # top_seller = tmp['sellerId'].value_counts()
    # top_comment = tmp[tmp['sellerId'] == top_seller.index[0]]
    # get_wc(top_comment['words_list'], filename)
    # return top_seller.index[0]
    get_wc(tmp['words_list'], filename)
    return tmp.index[0]


get_top_seller(data, 0, '积极商家词云图.png')
get_top_seller(data, 1, '消极商家词云图.png')
# 构建数据集

data['recomment'] = data['words_list'].apply(lambda x: ",".join(x))
dataset = data[['target', 'recomment']]

x, y = dataset['recomment'].values, dataset['target'].values

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vectorizer = CountVectorizer(min_df=2)
tfidf_transformer = TfidfTransformer()
word_count_x = count_vectorizer.fit_transform(x)
tfidf_x = tfidf_transformer.fit_transform(word_count_x)
print(tfidf_x.shape)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(128, 8), random_state=42)
scores = cross_val_score(clf, tfidf_x, y, cv=5)
print('五折交叉验证准确率 {:.4f}'.format(scores.mean()))

# predict
test = pd.read_excel('C:\\Users\\86180\\Desktop\\DaZhongdianpingbishe\\daZhongFood\\data\\test.xlsx')

test['recomment'] = test['comment'].apply(clean_word).apply(jieba.lcut).apply(
    lambda x: [i for i in x if i not in stop]
).apply(lambda x: ",".join(x))
test_x = test['recomment'].values

word_test_x = count_vectorizer.transform(test_x)
tfidf_test_x = tfidf_transformer.transform(word_test_x)

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(128, 8), random_state=42)
clf.fit(tfidf_x, y)

pred_y = clf.predict(tfidf_test_x)
test['target'] = pred_y
del test['recomment']

test.to_excel('C:\\Users\\86180\\Desktop\\DaZhongdianpingbishe\\daZhongFood\\data\\test.xlsx', index=None)

print('tfidf_ SVM train accuracy %s' % accuracy_score(y, clf.predict(tfidf_x)))



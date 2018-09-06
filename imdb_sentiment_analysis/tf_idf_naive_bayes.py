import pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import numpy as np
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_val_score


def split_html_label(text):

    example = BeautifulSoup(text, "html.parser").get_text()
    letters = re.sub('[^a-zA-Z]', " ", example)
    word = letters.lower().split()
    return word


def load_data(dataset):

    data = []
    for i in range(len(dataset['review'])):
        data.append(" ".join(split_html_label(dataset['review'][i])))
    return data


def tf_idf_process():

    """

    ngram_range:比如'Python is useful'这个句子中，
                ngram_range(1,3)之后可得到'Python'  'is'  'useful'  'Python is'  'is useful' 和'Python is useful'
                如果是ngram_range (1,1) 则只能得到单个单词'Python'  'is'和'useful'

    """

    tfidf = TFIDF(min_df=2,
                  max_features=None,
                  strip_accents='unicode',
                  analyzer='word',
                  token_pattern=r'\w{1,}',
                  ngram_range=(1, 3),  # 二元文法模型
                  use_idf=1,
                  smooth_idf=1,
                  sublinear_tf=1,
                  stop_words='english')  # 去掉英文停用词

    return tfidf


if __name__ == '__main__':

    data_dir = "../data/imdb/"
    train = pd.read_csv("{}{}".format(data_dir, "labeledTrainData.tsv"), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("{}{}".format(data_dir, "testData.tsv"), header=0, delimiter="\t", quoting=3)
    print(train.shape)
    print(train.columns.values)
    print(test.columns.values)

    x_train = load_data(train)
    x_test = load_data(test)
    label = train['sentiment']

    train_len = len(x_train)
    # 数据拼接
    X = x_train+x_test

    tf_idf = tf_idf_process()
    tf_idf.fit(X)

    # 转换成tfidf
    x_data_set = tf_idf.transform(X)

    train_x = x_data_set[:train_len]
    test_x = x_data_set[train_len:]

    # 朴素贝叶斯
    model = MNB()
    model.fit(train_x, label)

    print("cross validation scores:  \n", cross_val_score(model, train_x, label, cv=10, scoring='roc_auc'))
    print("the mean value of cross validation scores: ", np.mean(cross_val_score(model, train_x,
                                                              label, cv=10, scoring='roc_auc')))


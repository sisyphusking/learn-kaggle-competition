import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import pickle
import os
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.ensemble import RandomForestClassifier


def clean_data(review, remove_stopwords=True):

    review = BeautifulSoup(review, "html.parser").get_text()
    raw = tokenizer.tokenize(review)

    sentences = []
    for i in raw:
        letters = re.sub('[^a-zA-Z]', " ", i)
        words = letters.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        sentences.extend(words)
    return sentences


def process_data(data):
    # data_set = [clean_data(review) for review in data["review"]]
    data_set = []
    for review in data["review"]:
        data_set.append(clean_data(review))
    return data_set


def word2vec_model(data, num_features=300, num_workers=8, path="../data/imdb/word2vec.h5"):
    min_word_count = 40
    context = 10
    downsampling = 1e-3
    model = Word2Vec(data, size=num_features, workers=num_workers,
                     min_count=min_word_count, window=context, sample=downsampling)
    model.save(path)
    return model


def pickle_data(obj, file):

    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def reload_pickle(file):

    with open(file, 'rb')as f:
        data = pickle.load(f)
    return data


def convert_to_vec(data, model, path=None):
    num_features = 300

    def feature_vec(words, model):

        # 300维的词向量，list
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0.
        index2word_set = set(model.wv.index2word)
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                # 句子中的每个词向量相加
                featureVec = np.add(featureVec, model[word])

        featureVec = np.divide(featureVec, nwords)
        return featureVec

    features = np.zeros((len(data), num_features), dtype="float32")

    for index, d in enumerate(data):
        features[index] = feature_vec(d, model)

    pickle_data(features, path)
    return features


if __name__ == '__main__':

    data_dir = "../data/imdb/"
    tokenizer = nltk.data.load('../data/imdb/english.pickle')
    train = pd.read_csv("{}{}".format(data_dir, "labeledTrainData.tsv"),
                        header=0, delimiter="\t", quoting=3).fillna(" ")
    unlabeled_train = pd.read_csv("{}{}".format(data_dir, "unlabeledTrainData.tsv"),
                                  header=0, delimiter="\t", quoting=3).fillna(" ")
    test = pd.read_csv("{}{}".format(data_dir, "testData.tsv"), header=0, delimiter="\t", quoting=3).fillna(" ")
    print("training set shape: ", train.shape)

    features_path = "../data/imdb/features.pkl"
    model_path = "../data/imdb/word2vec.h5"

    if not os.path.exists(model_path):

        data = pd.concat([train['review'], unlabeled_train['review']])
        data_set = []
        for i in data:
            data_set.append(clean_data(i))
        print(np.shape(data_set))
        model = word2vec_model(data_set, path=model_path)

    model = reload_pickle(model_path)

    if not os.path.exists(features_path):

        data = process_data(train)
        features = convert_to_vec(data, model, features_path)
        pickle_data(features, features_path)

    features = reload_pickle(features_path)
    print("features shape: ", np.shape(features))
    labels = train['sentiment']

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)

    rf_model.fit(np.array(features), labels)
    scores = cross_val_score(rf_model, features, labels, cv=10, scoring='roc_auc')

    print("random forest scores: ", scores)
    print(" mean scores: ", np.mean(scores))

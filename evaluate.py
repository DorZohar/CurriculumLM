import os

from EvaluateW2V.datasets.utils import _fetch_file, _get_dataset_dir
from EvaluateW2V.evaluate import evaluate_on_all
from gensim.models.keyedvectors import KeyedVectors
import sys
from time import time
import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from scipy.stats import norm
import pandas as pd
import shutil

from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords

import keras


tokenizer = WordPunctTokenizer()
stop = set(stopwords.words('english'))


def string_to_nbow(text, w2v):
    tokens = tokenizer.tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop]
    avg = [w2v[word] for word in tokens if word in w2v]

    return avg


def extrinsic_test__sentiment_analysis(w2v, alpha, folds = 5):

    MAX_LEN = 500

    dataset_dir = _get_dataset_dir("sentiment")

    if not os.path.isdir("%s/aclImdb" % dataset_dir):
        path = _fetch_file(url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                           data_dir="sentiment",
                           uncompress=True,
                           #move='aclImdb',
                           verbose=0)

        shutil.move("%s/aclImdb" % path, dataset_dir)

    t = time()

    train_data = load_files("%s/aclImdb/train" % dataset_dir, encoding='utf-8', shuffle=True, random_state=42)
    test_data = load_files("%s/aclImdb/test" % dataset_dir, encoding='utf-8')

    print("Done loading dataset after %d seconds" % (time() - t))

    t = time()

    word2index = {word: index + 1 for index, word in enumerate(w2v.index2word)}

    train_features = [[word2index.get(word.lower(), 0) for word in tokenizer.tokenize(text)] for text in train_data.data]
    train_features = keras.preprocessing.sequence.pad_sequences(train_features,
                                                                maxlen=MAX_LEN,
                                                                padding='pre',
                                                                truncating='post',
                                                                )
    test_features = [[word2index[word.lower()] for word in tokenizer.tokenize(text)] for text in test_data.data]
    test_features = keras.preprocessing.sequence.pad_sequences(test_features,
                                                               maxlen=MAX_LEN,
                                                               padding='pre',
                                                               truncating='post',
                                                              )

    train_X, valid_X, train_y, valid_y = train_test_split(train_features, train_data.target, test_size=0.2, shuffle=False)

    print("Done preprocessing dataset after %d seconds" % (time() - t))

    model = keras.models.Sequential()
    model.add(w2v.get_keras_embedding(False))
    model.add(keras.layers.LSTM(100, dropout=0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics='accuracy')
    early_stop_cb = keras.callbacks.EarlyStopping('val_accuracy', patience=4)
    model.fit(train_X,
              train_y,
              batch_size=64,
              epochs=5,
              callbacks=[early_stop_cb],
              validation_data=(valid_X, valid_y))

    t = time()
    #cv_results = cross_validate(model, train_features, train_data.target, cv=folds, n_jobs=3)
    print("Done training after %d seconds" % (time() - t))

    results = {}

    # results['sentiment_val_acc'] = np.mean(scores)
    # const = norm.ppf(1 - alpha / 2)
    # results['sentiment_int'] = const * np.sqrt(results['sentiment_val_acc'] * (1 - results['sentiment_val_acc']) / folds)
    #
    # model.fit(train_features, train_data.target)
    results['sentiment_test_acc'] = model.evaluate(test_features, test_data.target)[1]
    print(results['sentiment_test_acc'])

    return pd.DataFrame([results])


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError
    path = sys.argv[1]
    print(path)
    t = time()
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)
    print("word2vec loaded after %d seconds" % (time() - t))
    sentiment_res = extrinsic_test__sentiment_analysis(w2v, 0.05, 150)
    w2v_dict = {word: w2v[word] / np.linalg.norm(w2v[word]) for word in w2v.index2word}

    # results = evaluate_on_all(w2v_dict)
    # results = results.join(sentiment_res)
    results = sentiment_res
    print(results)
    results.to_csv("%s.csv" % path)

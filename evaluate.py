import os

from EvaluateW2V.datasets.utils import _fetch_file, _get_dataset_dir
from EvaluateW2V.evaluate import evaluate_on_all
from gensim.models.keyedvectors import KeyedVectors
import sys
from time import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_files
import tarfile
from scipy.stats import norm
import pandas as pd

from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords


tokenizer = WordPunctTokenizer()
stop = set(stopwords.words('english'))


def string_to_nbow(text, w2v):
    tokens = tokenizer.tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop]
    avg = np.mean([w2v[word] for word in tokens if word in w2v], axis=0)

    return avg


def extrinsic_test__sentiment_analysis(w2v, alpha, folds = 5):

    dataset_dir = _get_dataset_dir("sentiment")

    if not os.path.isdir("%s/aclImdb" % dataset_dir):
        path = _fetch_file(url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                           data_dir="sentiment",
                           uncompress=True,
                           move='aclImdb',
                           verbose=0)

        tar = tarfile.open(path, "r:gz")
        tar.extract("aclImdb/train/neg")
        tar.extract("aclImdb/train/pos")
        tar.extract("aclImdb/test/neg")
        tar.extract("aclImdb/test/pos")
        tar.close()

    t = time()

    train_data = load_files("%s/aclImdb/train" % dataset_dir, encoding='utf-8', shuffle=True, random_state=42)
    test_data = load_files("%s/aclImdb/test" % dataset_dir, encoding='utf-8')

    print("Done loading dataset after %d seconds" % (time() - t))

    t = time()

    train_features = [string_to_nbow(text, w2v) for text in train_data.data]
    test_features = [string_to_nbow(text, w2v) for text in test_data.data]

    print("Done preprocessing dataset after %d seconds" % (time() - t))

    model = LogisticRegression(random_state=42)

    t = time()
    cv_results = cross_validate(model, train_features, train_data.target, cv=folds, n_jobs=3)
    print("Done training after %d seconds" % (time() - t))

    results = {}

    scores = cv_results['test_score']
    results['sentiment_val_acc'] = np.mean(scores)
    const = norm.ppf(1 - alpha / 2)
    results['sentiment_int'] = const * np.sqrt(results['sentiment_val_acc'] * (1 - results['sentiment_val_acc']) / folds)

    model.fit(train_features, train_data.target)
    results['sentiment_test_acc'] = model.score(test_features, test_data.target)

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

    results = evaluate_on_all(w2v_dict)
    results = results.join(sentiment_res)
    print(results)
    results.to_csv("%s.csv" % path)

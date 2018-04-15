from EvaluateW2V.evaluate import evaluate_on_all
from gensim.models.keyedvectors import KeyedVectors
import sys
from time import time


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError
    path = sys.argv[1]
    print(path)
    t = time()
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)
    print("word2vec loaded after %d seconds" % (time() - t))
    w2v_dict = {word: w2v[word] for word in w2v.index2word}

    results = evaluate_on_all(w2v_dict)
    print(results)
    results.to_csv("%s.csv" % path)


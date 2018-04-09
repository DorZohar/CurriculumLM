import multiprocessing
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from EvaluateW2V.evaluate import evaluate_on_all
from gensim.models.keyedvectors import KeyedVectors


def evaluate_word2vec(path):
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)
    return evaluate_on_all(w2v.__dict__)


def train_word2vec(inp):

    max_length = 0
    with open(inp, 'r') as f:
        for line in f.readlines():
            max_length = max(max_length, len(line))

    params = {
        'size': 300,
        'window': 10,
        'min_count': 10,
        'workers': max(1, multiprocessing.cpu_count() - 1),
        'sample': 1E-5,
    }

    word2vec = Word2Vec(LineSentence(inp, max_sentence_length=max_length),
                        **params)

    return word2vec


def train_curriculum_word2vec():
    pass


if __name__ == '__main__':
    #print(evaluate_word2vec('C:\\Wiki\\GoogleNews-vectors-negative300.bin'))

    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    input_path, output_path = sys.argv[1:3]
    w2v = train_word2vec(input_path)

    w2v.save(output_path)

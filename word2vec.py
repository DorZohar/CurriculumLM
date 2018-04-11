import multiprocessing
import sys

import itertools
from time import time

from gensim import utils
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import general
from EvaluateW2V.evaluate import evaluate_on_all
from gensim.models.keyedvectors import KeyedVectors
from config import conf

def cluster_to_string(cluster, i):
    return "#CLUSTER_%s#" % cluster[:i]


def string_to_cluster(string):
    return string.strip("#").split('_')[1]


def evaluate_word2vec(path):
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)
    w2v_dict = {word: w2v[word] for word in w2v.index2word}

    return evaluate_on_all(w2v_dict)


def train_word2vec(input_file):

    max_length = 0
    with open(input_file, 'r') as f:
        for line in f.readlines():
            max_length = max(max_length, len(line))

    params = {
        'size': 300,
        'window': 10,
        'min_count': 10,
        'workers': max(1, multiprocessing.cpu_count() - 1),
        'sample': 1E-5,
        'iter': 100,
    }

    word2vec = Word2Vec(LineSentence(input_file, max_sentence_length=max_length),
                        **params)

    return word2vec


class CurriculumIter():
    """
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
    """

    def __init__(self, source, brown_dict, iteration, max_sentence_length, limit=None):
        """
        `source` can be either a string or a file object. Clip the file to the first
        `limit` lines (or no clipped if limit is None, the default).

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.brown_dict = brown_dict
        self.iteration = iteration

    def __iter__(self):
        # If it didn't work like a file, use it as a string filename
        with utils.smart_open(self.source) as fin:
            for line in itertools.islice(fin, self.limit):
                line = utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield [cluster_to_string(self.brown_dict[word], self.iteration) if word in self.brown_dict else word
                           for word in line[i: i + self.max_sentence_length]]
                    i += self.max_sentence_length


def expand_word2vec_matrix(word2vec, old_word2vec, word2cluster, old_len):

    for idx, word in enumerate(word2vec.wv.index2word):
        if word.startswith("#CLUSTER_") or word in word2cluster:
            cluster = word2cluster[word] if word in word2cluster else string_to_cluster(word)
            old_cluster = cluster_to_string(cluster, old_len)
            assert old_cluster in old_word2vec
            word2vec.wv.syn0[idx] = old_word2vec[old_cluster]
        else:
            assert word in old_word2vec
            word2vec.wv.syn0[idx] = old_word2vec[word]

    return word2vec


def train_curriculum_word2vec(input_file, clusters_file, conf):

    # max_length = 0
    # with open(input_file, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         max_length = max(max_length, len(line))
    #
    # print(max_length)

    max_length = 608356

    old_word2vec = None
    old_len = None
    word2cluster = general.read_brown_clusters(clusters_file)

    params = {
        'size': 300,
        'window': 10,
        'min_count': 10,
        'workers': max(1, multiprocessing.cpu_count() - 1),
        'sample': 1E-5,
        'iter': 1,
    }

    for i in range(2):
        start = time()
        # Create w2v model
        print("Iteration %d" % i)
        word2vec = Word2Vec(**params)
        iterator = CurriculumIter(input_file, word2cluster, i + 1, max_length)
        word2vec.build_vocab(iterator)
        if i > 0:
            word2vec = expand_word2vec_matrix(word2vec,
                                              old_word2vec,
                                              word2cluster,
                                              old_len)

        word2vec.train(iterator,
                       total_examples=word2vec.corpus_count,
                       epochs=word2vec.iter,
                       start_alpha=word2vec.alpha,
                       end_alpha=word2vec.min_alpha
                       )
        old_word2vec = word2vec
        old_len = i + 1
        print("Iteration %d finished after %s seconds" % (i, time() - start))

    word2vec = Word2Vec(**params)
    word2vec.build_vocab(LineSentence(input_file, max_sentence_length=max_length))
    word2vec = expand_word2vec_matrix(word2vec,
                                      old_word2vec,
                                      word2cluster,
                                      old_len)

    params['iter'] = 5
    word2vec.train(LineSentence(input_file, max_sentence_length=max_length),
                   total_examples=word2vec.corpus_count,
                   epochs=word2vec.iter,
                   start_alpha=word2vec.alpha,
                   end_alpha=word2vec.min_alpha)

    return word2vec




if __name__ == '__main__':

    #print(evaluate_word2vec('C:\\Wiki\\GoogleNews-vectors-negative300.bin'))

    # if len(sys.argv) == 1:
    #     cfg_path = 'config.py'
    # else:
    #     cfg_path = sys.argv[1]

    #conf = general.load_config(cfg_path)

    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    input_path, output_path = sys.argv[1:3]

    w2v = train_curriculum_word2vec(input_path, conf['brown__clusters_file'], conf)
    w2v.wv.save_word2vec_format(output_path, binary=True)

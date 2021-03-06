import multiprocessing
import sys
import pickle as pkl
import itertools
from time import time

from collections import defaultdict
from gensim import utils
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import general
from EvaluateW2V.evaluate import evaluate_on_all
from gensim.models.keyedvectors import KeyedVectors


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
        'iter': 20,
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


def build_curriculum_vocab(vocab, clusters_dict, prefix):

    total_words = 0
    new_vocab = defaultdict(int)

    for word, count in vocab.items():
        if word not in clusters_dict:
            total_words += 1
            new_vocab[word] = vocab[word]
        else:
            cluster_str = cluster_to_string(clusters_dict[word], prefix)
            if cluster_str in new_vocab:
                new_vocab[cluster_str] += vocab[word]
            else:
                total_words += 1
                new_vocab[cluster_str] = vocab[word]

    return new_vocab


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

    total_epochs = (conf['curriculum_end'] - conf['curriculum_start']) // conf['curriculum_step'] * conf['curriculum_epochs'] + conf['w2v_epochs']
    alpha_per_epoch = (conf['start_alpha'] - conf['end_alpha']) / total_epochs
    cur_alpha = conf['start_alpha']

    params = {
        'size': 300,
        'window': 10,
        'min_count': 10,
        'workers': max(1, multiprocessing.cpu_count() - 1),
        'sample': 1E-5,
        'iter': conf['w2v_epochs'],
    }

    final_word2vec = Word2Vec(**params)
    total_words, corpus_count = final_word2vec.vocabulary.scan_vocab(LineSentence(input_file, max_sentence_length=max_length))
    final_word2vec.corpus_count = corpus_count

    params['iter'] = conf['curriculum_epochs']

    for i in range(conf['curriculum_start'], conf['curriculum_end'], conf['curriculum_step']):
        start = time()
        # Create w2v model
        print("Iteration %d" % i)
        word2vec = Word2Vec(**params)
        iterator = CurriculumIter(input_file, word2cluster, i + 1, max_length)

        word2vec.vocabulary.raw_vocab = build_curriculum_vocab(final_word2vec.vocabulary.raw_vocab, word2cluster, i + 1)
        word2vec.corpus_count = final_word2vec.corpus_count
        word2vec.vocabulary.prepare_vocab(word2vec.hs,
                                          word2vec.negative,
                                          word2vec.wv)
        word2vec.trainables.prepare_weights(word2vec.hs,
                                            word2vec.negative,
                                            word2vec.wv,
                                            vocabulary=word2vec.vocabulary)

        if old_word2vec is not None:
            word2vec = expand_word2vec_matrix(word2vec,
                                              old_word2vec,
                                              word2cluster,
                                              old_len)

        word2vec.train(iterator,
                       total_examples=word2vec.corpus_count,
                       epochs=word2vec.iter,
                       start_alpha=cur_alpha,
                       end_alpha=(cur_alpha - alpha_per_epoch * conf['curriculum_epochs'])
                       )
        cur_alpha -= alpha_per_epoch * conf['curriculum_epochs']
        old_word2vec = word2vec
        old_len = i + 1
        print("Iteration %d finished after %.2f seconds" % (i, time() - start))

    t = time()
    final_word2vec.vocabulary.prepare_vocab(final_word2vec.hs,
                                            final_word2vec.negative,
                                            final_word2vec.wv)
    final_word2vec.trainables.prepare_weights(final_word2vec.hs,
                                              final_word2vec.negative,
                                              final_word2vec.wv,
                                              vocabulary=final_word2vec.vocabulary)
    final_word2vec = expand_word2vec_matrix(final_word2vec,
                                            old_word2vec,
                                            word2cluster,
                                            old_len)

    final_word2vec.train(LineSentence(input_file, max_sentence_length=max_length),
                         total_examples=final_word2vec.corpus_count,
                         epochs=final_word2vec.iter,
                         start_alpha=cur_alpha,
                         end_alpha=conf['end_alpha'])

    print("Training final model finished after %.2f seconds" % (time() - t))

    return final_word2vec


def w2v_curriculum_filename(curriculum, conf):

    if curriculum == '1':
        return "curriculum_w2v_%d_%d_%d_%depochs.bin" % (conf['curriculum_start'],
                                                         conf['curriculum_end'],
                                                         conf['curriculum_step'],
                                                         conf['w2v_epochs'])

    return "baseline_w2v_%depochs.bin" % (conf['w2v_epochs'])


if __name__ == '__main__':

    #print(evaluate_word2vec('C:\\Wiki\\GoogleNews-vectors-negative300.bin'))

    # if len(sys.argv) == 1:
    #     cfg_path = 'config.py'
    # else:
    #     cfg_path = sys.argv[1]

    #conf = general.load_config(cfg_path)

    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    input_path, curriculum = sys.argv[1:3]

    conf = dict()
    conf['curriculum_epochs'] = 1
    conf['curriculum_start'] = 6
    conf['curriculum_end'] = 15
    conf['curriculum_step'] = 2
    conf['w2v_epochs'] = 20
    conf['start_alpha'] = 0.025
    conf['end_alpha'] = 0.0001

    output_path = w2v_curriculum_filename(curriculum, conf)

    if curriculum == '1':
        w2v = train_curriculum_word2vec(input_path, 'brown_clusters_wiki.txt', conf)
    else:
        print("Regular word2vec")
        w2v = train_word2vec(input_path)
    w2v.wv.save_word2vec_format(output_path, binary=True)


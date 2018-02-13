import threading
import numpy as np
import keras


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def brown_generator(file_path, batch_size, max_len, is_test=False):

    batch_sentences = []
    i = 0
    while True:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip('\n').split(' ')
                line = [int(word_id) for word_id in line]
                pos = 0
                while pos < len(line):
                    sentence = line[pos:pos+max_len]
                    batch_sentences.append(sentence)
                    i += 1
                    pos += max_len
                    if i >= batch_size:

                        sequences = keras.preprocessing.sequence.pad_sequences(batch_sentences,
                                                                              maxlen=max_len,
                                                                              padding='pre',
                                                                              truncating='post',
                                                                              )

                        targets = np.expand_dims(sequences, axis=-1)
                        weights = np.not_equal(sequences, 0).astype(np.float32)

                        if is_test:
                            yield {'Input': sequences}, weights
                        else:
                            yield {'Input': sequences}, targets, weights
                        i = 0
                        batch_sentences = []


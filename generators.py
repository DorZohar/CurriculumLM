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
def brown_generator(file_path, batch_size, max_len, input_word_dict, input_classes, output_word_dict, output_classes, is_test=False):

    batch_sentences = []
    batch_targets = []
    i = 0
    while True:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip('\n').split(' ')
                input_line = [input_word_dict[word] if word in input_word_dict else input_classes for word in line]
                output_line = [output_word_dict[word] if word in output_word_dict else output_classes for word in line]
                batch_sentences.append(input_line)
                batch_targets.append(output_line[1:] + [0])
                i += 1
                if i >= batch_size:

                    sequences = keras.preprocessing.sequence.pad_sequences(batch_sentences,
                                                                          maxlen=max_len,
                                                                          padding='pre',
                                                                          truncating='post',
                                                                          )

                    batch_targets = keras.preprocessing.sequence.pad_sequences(batch_targets,
                                                                               maxlen=max_len,
                                                                               padding='pre',
                                                                               truncating='post',
                                                                               )

                    batch_targets = np.expand_dims(batch_targets, axis=-1)
                    weights = np.not_equal(sequences, 0).astype(np.float32)
                    weights[:, -1] = 0

                    if is_test:
                        yield {'Input': sequences}, weights
                    else:
                        yield {'Input': sequences}, batch_targets, weights
                    i = 0
                    batch_sentences = []
                    batch_targets = []


import keras
import time
import sys

import general
import numpy as np
import pickle as pkl
import keras.backend as K
from gensim.models.keyedvectors import KeyedVectors
from generators import brown_generator, dataset_generator
import os
import tensorflow as tf


class CustomSoftmax(keras.layers.Layer):
    def __init__(self, embeddingLayer, units, **kwargs):
        self.W = K.transpose(embeddingLayer.weights[0])
        self.b = None
        self.units = units
        self.activation = keras.activations.get('softmax')
        super(CustomSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.b = self.add_weight(name='b',
                                 shape=(self.units, ),
                                 initializer='uniform',
                                 trainable=True)
        super(CustomSoftmax, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        output = K.dot(x, self.W)
        output = K.bias_add(output, self.b)

        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


def create_embedding_matrix(word_dict, word2vec):

    classes = len(word_dict) + 2
    vec_size = word2vec.vector_size
    matrix = np.zeros((classes, vec_size))

    for word, idx in word_dict.items():
        if word in word2vec:
            matrix[idx] = word2vec[word]

    return matrix


def model_struct(input_classes, output_classes, conf):
    input_layer = keras.layers.Input((None,), name='Input')
    embedding_layer = keras.layers.Embedding(input_dim=input_classes,
                                             output_dim=conf['lstm__embedding_size'],
                                             name='Embedding')

    next_layer = embedding_layer(input_layer)

    if conf['task'] != 'LM':
        uppercase_input_layer = keras.layers.Input((None, 1), name='UppercaseInput')
        next_layer = keras.layers.Concatenate()([next_layer, uppercase_input_layer])
        input_layer = [input_layer, uppercase_input_layer]

    LSTM_layer = keras.layers.LSTM(conf['lstm__hidden_size'],
                                   name='LSTM',
                                   activation=conf['lstm__activation'],
                                   recurrent_dropout=conf['lstm__rec_dropout'],
                                   dropout=conf['lstm__input_dropout'],
                                   return_sequences=True)

    # if conf['task'] != 'LM':
    #     LSTM_layer = keras.layers.Bidirectional(LSTM_layer, name='LSTM')

    next_layer = LSTM_layer(next_layer)

    if conf['lstm__weight_tying']:
        output_layer = keras.layers.TimeDistributed(CustomSoftmax(embedding_layer, units=output_classes),
                                                    name='Softmax')(next_layer)
    else:
        output_layer = keras.layers.TimeDistributed(keras.layers.Dense(output_classes,
                                                                       activation='softmax'),
                                                    name='Softmax')(next_layer)

    return input_layer, output_layer


def create_language_model(conf, input_classes, output_classes, embedding_mat=None, softmax_mat=None, softmax_bias=None, lstm_weights=None):

    assert (softmax_mat is None) == (softmax_bias is None)

    if softmax_mat is not None:
        assert output_classes == softmax_mat.shape[1] == softmax_bias.shape[0]

    if embedding_mat is not None:
        assert input_classes == embedding_mat.shape[0]

    if conf['lstm__limit_gpu']:
        with tf.device('/gpu:%d' % conf['lstm__gpu_id']):
            input_layer, output_layer = model_struct(input_classes, output_classes, conf)
    else:
        input_layer, output_layer = model_struct(input_classes, output_classes, conf)

    model = keras.models.Model(input_layer, output_layer)

    model.compile(optimizer=keras.optimizers.SGD(conf['lstm__learn_rate']),
                  loss='sparse_categorical_crossentropy',
                  sample_weight_mode='temporal',
                  weighted_metrics=['accuracy'])

    if embedding_mat is not None:
        model.get_layer('Embedding').set_weights([embedding_mat])
    if softmax_bias is not None and not conf['lstm__weight_tying']:
        model.get_layer('Softmax').set_weights([softmax_mat, softmax_bias])
    if lstm_weights is not None:
        model.get_layer('LSTM').set_weights(lstm_weights)

    return model


def train_language_model(model, conf, input_word2id, input_classes, output_word2id, output_classes, base_path, is_curriculum=False):

    if conf['task'] == 'LM':
        train_gen = brown_generator(conf['brown__train_file'],
                                    conf['batch_size'],
                                    conf['max_len'],
                                    conf['train_steps'],
                                    input_word2id,
                                    input_classes,
                                    output_word2id,
                                    output_classes)

        valid_gen = brown_generator(conf['brown__valid_file'],
                                    conf['batch_size'],
                                    conf['max_len'],
                                    conf['valid_steps'],
                                    input_word2id,
                                    input_classes,
                                    output_word2id,
                                    output_classes)
    else:
        train_gen = dataset_generator(general.get_dataset_path(conf['task'], 'train'),
                                      conf['batch_size'],
                                      conf['max_len'],
                                      conf['train_steps'],
                                      input_word2id,
                                      input_classes,
                                      output_word2id)

        valid_gen = dataset_generator(general.get_dataset_path(conf['task'], 'valid'),
                                      conf['batch_size'],
                                      conf['max_len'],
                                      conf['valid_steps'],
                                      input_word2id,
                                      input_classes,
                                      output_word2id)

    callbacks = []

    monitor = 'val_loss' if conf['task'] == 'LM' else 'val_weighted_acc'

    if not is_curriculum:
        epochs = conf['epochs']
        earlyStop = keras.callbacks.EarlyStopping(patience=20, monitor=monitor)
        reduceLr = keras.callbacks.ReduceLROnPlateau(factor=0.2,
                                                     monitor=monitor,
                                                     patience=10,
                                                     min_lr=conf['lstm__learn_rate'] * 0.001,
                                                     verbose=conf['verbose'])
        callbacks.append(reduceLr)
    else:
        epochs = conf['mini_epochs']
        earlyStop = keras.callbacks.EarlyStopping(patience=0, monitor=monitor, verbose=conf['verbose'])

    callbacks.append(earlyStop)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    path = '%s%s' % (base_path, conf['model_paths'])

    checkpoint = keras.callbacks.ModelCheckpoint(path,
                                                 monitor=monitor,
                                                 save_best_only=True,
                                                 mode='min' if conf['task'] == 'LM' else 'max',
                                                 verbose=conf['verbose'])

    callbacks.append(checkpoint)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=conf['train_steps'] / conf['batch_size'],
                                  epochs=epochs,
                                  validation_data=valid_gen,
                                  validation_steps=conf['valid_steps'] / conf['batch_size'],
                                  workers=conf['workers'],
                                  callbacks=callbacks,
                                  verbose=conf['verbose'])

    with open('%shistory.txt' % base_path, 'w') as f:
        f.write('%s' % history.history)


def test_language_model(model, conf, input_word2id, input_classes, output_word2id, output_classes):
    if conf['task'] == 'LM':
        test_gen = brown_generator(conf['brown__test_file'],
                                   conf['batch_size'],
                                   conf['max_len'],
                                   conf['test_steps'],
                                   input_word2id,
                                   input_classes,
                                   output_word2id,
                                   output_classes)
    else:
        test_gen = dataset_generator(general.get_dataset_path(conf['task'], 'test'),
                                     conf['batch_size'],
                                     conf['max_len'],
                                     conf['test_steps'],
                                     input_word2id,
                                     input_classes,
                                     output_word2id)

    scores = model.evaluate_generator(test_gen,
                                      steps=conf['test_steps'] / conf['batch_size'])

    return scores


def generate_sample(model, starting_word, word_dict, len=10):

    assert starting_word in word_dict

    reverse_dict = {word_id: word for word, word_id in word_dict.items()}
    sentence = [word_dict[starting_word]]

    for i in range(len):
        np_sentence = np.reshape(sentence, (1, -1))
        probs = model.predict(np_sentence, verbose=0)
        next_word = np.argmax(probs[-1])
        sentence.append(next_word)

    sentence = [reverse_dict[word] for word in sentence]

    print("%s\n" % ' '.join(sentence))


def baseline_model(conf, base_model=None):

    np.random.seed(42)

    if os.name == 'nt':
        base_path = 'Models\\%s\\' % time.strftime('%Y_%m_%d-%H_%M')
    else:
        base_path = 'Models/%s/' % time.strftime('%Y_%m_%d-%H_%M')

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    general.create_config('%sconfig.py' % base_path, conf)

    if conf['task'] != 'LM':
        task_dict = general.extract_classes_dict(conf['task'])

    word2vec = KeyedVectors.load_word2vec_format(conf['w2v_path'], binary=True)
    word_dict = pkl.load(open(conf['brown__dict_file'], 'rb'))

    if base_model is None:
        embedding_mat = create_embedding_matrix(word_dict, word2vec)
        softmax_mat = None
        softmax_bias = None
        lstm_weights = None
    else:
        embedding_mat = base_model.get_layer('Embedding').get_weights()[0]
        # if not conf['lstm__weight_tying']:
        #     softmax_mat, softmax_bias = base_model.get_layer('Softmax').get_weights()
        # else:
        softmax_mat, softmax_bias = None, None
        lstm_weights = base_model.get_layer('LSTM').get_weights()
        lstm_weights[0] = np.pad(lstm_weights[0], [[0, 1], [0, 0]], mode='constant')

    classes = len(word_dict) + 1

    model = create_language_model(conf,
                                  classes + 1,
                                  classes + 1 if conf['task'] == 'LM' else len(task_dict),
                                  embedding_mat,
                                  softmax_mat,
                                  softmax_bias,
                                  lstm_weights)
    train_language_model(model,
                         conf,
                         word_dict,
                         classes,
                         word_dict if conf['task'] == 'LM' else task_dict,
                         classes if conf['task'] == 'LM' else len(task_dict),
                         base_path)
    print(test_language_model(model,
                              conf,
                              word_dict,
                              len(word_dict) + 1,
                              word_dict if conf['task'] == 'LM' else task_dict,
                              classes if conf['task'] == 'LM' else len(task_dict)))

    return model


def curriculum_model(conf):

    np.random.seed(42)

    assert conf['task'] == 'LM' or not conf['curriculum__output']

    if os.name == 'nt':
        base_path = 'Models\\%s\\' % time.strftime('%Y_%m_%d-%H_%M')
    else:
        base_path = 'Models/%s/' % time.strftime('%Y_%m_%d-%H_%M')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

        general.create_config('%sconfig.py' % base_path, conf)

    if conf['task'] != 'LM':
        task_dict = general.extract_classes_dict(conf['task'])

    word_dict = pkl.load(open(conf['brown__dict_file'], 'rb'))
    word2cluster = general.read_brown_clusters(conf['brown__clusters_file'])
    embedding_mat = None
    softmax_mat = None
    softmax_bias = None
    lstm_weights = None

    word2id, classes = general.create_cluster_dict(word2cluster, conf['curriculum__start'] + 1)

    if not conf['curriculum__input']:
        word2vec = KeyedVectors.load_word2vec_format(conf['w2v_path'], binary=True)
        embedding_mat = create_embedding_matrix(word_dict, word2vec)

    for i in range(conf['curriculum__start'], conf['curriculum__stop'], conf['curriculum__jumps']):

        if i > conf['curriculum__start']:
            old_word2id = word2id
            word2id, classes = general.create_cluster_dict(word2cluster, i + 1)

            embedding_mat, softmax_mat, softmax_bias = general.expand_all_matrices(embedding_mat,
                                                                                   softmax_mat,
                                                                                   softmax_bias,
                                                                                   old_word2id,
                                                                                   word2id,
                                                                                   classes,
                                                                                   conf)

        if conf['curriculum__input']:
            input_word2id = word2id
            input_classes = classes
        else:
            input_word2id = word_dict
            input_classes = len(word_dict) + 1

        if conf['curriculum__output']:
            output_word2id = word2id
            output_classes = classes
        else:
            if conf['task'] == 'LM':
                output_word2id = word_dict
                output_classes = len(word_dict) + 1
            else:
                output_word2id = task_dict
                output_classes = len(task_dict)

        print("Iteration %d, Input classes: %d, Output classes: %d" % (i+1, input_classes, output_classes))

        model = create_language_model(conf,
                                      input_classes + 1,
                                      output_classes + 1 if conf['task'] == 'LM' else output_classes,
                                      embedding_mat,
                                      softmax_mat,
                                      softmax_bias,
                                      lstm_weights)

        model_path = '%s\\%.2d\\' % (base_path, i) if os.name == 'nt' else '%s/%.2d/' % (base_path, i)

        train_language_model(model,
                             conf,
                             input_word2id,
                             input_classes,
                             output_word2id,
                             output_classes,
                             model_path,
                             is_curriculum=True)
        print(test_language_model(model, conf, input_word2id, input_classes, output_word2id, output_classes))

        embedding_mat = model.get_layer('Embedding').get_weights()[0]
        if not conf['lstm__weight_tying']:
            softmax_mat, softmax_bias = model.get_layer('Softmax').get_weights()
        lstm_weights = model.get_layer('LSTM').get_weights()

    classes = len(word_dict) + 1

    embedding_mat, softmax_mat, softmax_bias = general.expand_all_matrices(embedding_mat,
                                                                           softmax_mat,
                                                                           softmax_bias,
                                                                           word2id,
                                                                           word_dict,
                                                                           classes,
                                                                           conf)

    model = create_language_model(conf,
                                  classes + 1,
                                  classes + 1 if conf['task'] == 'LM' else len(task_dict),
                                  embedding_mat,
                                  softmax_mat,
                                  softmax_bias,
                                  lstm_weights)
    train_language_model(model,
                         conf,
                         word_dict,
                         classes,
                         word_dict if conf['task'] == 'LM' else task_dict,
                         classes if conf['task'] == 'LM' else len(task_dict),
                         base_path)
    print(test_language_model(model,
                              conf,
                              word_dict,
                              classes,
                              word_dict if conf['task'] == 'LM' else task_dict,
                              classes if conf['task'] == 'LM' else len(task_dict)))

    return model

#
# def continue_after_curriculum(path, iteration, conf):
#
#     if os.name == 'nt':
#         base_path = 'Models\\%s\\' % time.strftime('%Y_%m_%d-%H_%M')
#     else:
#         base_path = 'Models/%s/' % time.strftime('%Y_%m_%d-%H_%M')
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)
#
#     general.create_config('%sconfig.py' % base_path, conf)
#
#     model = keras.models.load_model(path)
#     word_dict = pkl.load(open(conf['brown__dict_file'], 'rb'))
#     word2cluster = general.read_brown_clusters(conf['brown__clusters_file'])
#     word2id, classes = general.create_cluster_dict(word2cluster, iteration+1)
#
#     valid_gen = brown_generator(conf['brown__valid_file'],
#                                 conf['batch_size'],
#                                 conf['max_len'],
#                                 word2id,
#                                 classes,
#                                 word2id,
#                                 classes)
#
#     print(model.evaluate_generator(valid_gen, steps=conf['valid_steps'] / conf['batch_size']))
#
#     print(test_language_model(model, conf, word2id, classes, word2id, classes))
#
#     embedding_mat = model.get_layer('Embedding').get_weights()[0]
#     softmax_mat, softmax_bias = model.get_layer('Softmax').get_weights()
#     lstm_weights = model.get_layer('LSTM').get_weights()
#
#     classes = len(word_dict) + 1
#
#     embedding_mat, softmax_mat, softmax_bias = general.expand_all_matrices(embedding_mat,
#                                                                            softmax_mat,
#                                                                            softmax_bias,
#                                                                            word2id,
#                                                                            word_dict,
#                                                                            classes)
#
#     model = create_language_model(conf, classes + 1, embedding_mat, softmax_mat, softmax_bias, lstm_weights)
#     train_language_model(model, conf, word_dict, classes, base_path)
#     print(test_language_model(model, conf, word_dict, classes))


if __name__ == '__main__':

    if len(sys.argv) == 1:
        cfg_path = 'config.py'
    else:
        cfg_path = sys.argv[1]

    conf = general.load_config(cfg_path)

    if not conf['curriculum__input'] and not conf['curriculum__output']:
        baseline_model(conf)
    else:
        curriculum_model(conf)

import keras
import time

import shutil

import config
import general
import numpy as np
import pickle as pkl
import keras.backend as K
from gensim.models.keyedvectors import KeyedVectors
from generators import brown_generator
import os


def perplexity(y_true, y_pred):
    return K.pow(K.constant(2.0), K.mean(K.sparse_categorical_crossentropy(y_true, y_pred)))


def create_embedding_matrix(word_dict, word2vec):

    classes = len(word_dict) + 2
    vec_size = word2vec.vector_size
    matrix = np.zeros((classes, vec_size))

    for word, idx in word_dict.items():
        if word in word2vec:
            matrix[idx] = word2vec[word]

    return matrix


def create_language_model(conf, classes, embedding_mat=None, softmax_mat=None, softmax_bias=None, lstm_weights=None):

    assert (softmax_mat is None) == (softmax_bias is None)

    if softmax_mat is not None:
        assert classes == softmax_mat.shape[1] == softmax_bias.shape[0]

    input_layer = keras.layers.Input((None, ), name='Input')
    embedding_layer = keras.layers.Embedding(input_dim=classes,
                                             output_dim=conf['lstm__embedding_size'],
                                             name='Embedding')(input_layer)

    LSTM_layer = keras.layers.LSTM(conf['lstm__hidden_size'],
                                   name='LSTM',
                                   activation=conf['lstm__activation'],
                                   recurrent_dropout=conf['lstm__rec_dropout'],
                                   dropout=conf['lstm__input_dropout'],
                                   return_sequences=True)(embedding_layer)

    output_layer = keras.layers.TimeDistributed(keras.layers.Dense(classes,
                                                                   activation='softmax'),
                                                name='Softmax')(LSTM_layer)

    model = keras.models.Model(input_layer, output_layer)

    model.compile(optimizer=keras.optimizers.RMSprop(lr=conf['lstm__learn_rate']),
                  loss='sparse_categorical_crossentropy',
                  sample_weight_mode='temporal',
                  weighted_metrics=['accuracy', perplexity])

    if embedding_mat is not None:
        model.get_layer('Embedding').set_weights([embedding_mat])
    if softmax_bias is not None:
        model.get_layer('Softmax').set_weights([softmax_mat, softmax_bias])
    if lstm_weights is not None:
        model.get_layer('LSTM').set_weights(lstm_weights)

    return model


def train_language_model(model, conf, word2id, classes, base_path):

    train_gen = brown_generator(conf['brown__train_file'],
                                conf['batch_size'],
                                conf['max_len'],
                                word2id,
                                classes)

    valid_gen = brown_generator(conf['brown__valid_file'],
                                conf['batch_size'],
                                conf['max_len'],
                                word2id,
                                classes)

    callbacks = []

    if iter is None:
        epochs = conf['epochs']
        #earlyStop = keras.callbacks.EarlyStopping(patience=2)
        reduceLr = keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                     patience=3,
                                                     min_lr=conf['lstm__learn_rate'] * 0.25)
        callbacks.append(reduceLr)
    else:
        epochs = conf['mini_epochs']
        earlyStop = keras.callbacks.EarlyStopping(patience=0)
        callbacks.append(earlyStop)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    path = '%s\\%s' % (base_path, conf['model_paths'])

    checkpoint = keras.callbacks.ModelCheckpoint(path,
                                                 save_best_only=True,
                                                 mode='min',
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

    with open('%s\\history.txt' % base_path, 'w') as f:
        f.write('%s' % history.history)


def test_language_model(model, conf, word2id, classes):
    test_gen = brown_generator(conf['brown__test_file'],
                               conf['batch_size'],
                               conf['max_len'],
                               word2id,
                               classes)

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


def baseline_model():

    np.random.seed(42)

    base_path = 'Models\\%s\\' % time.strftime('%Y_%m_%d-%H_%M')

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    shutil.copy('config.py', '%s//config.py' % base_path)

    conf = config.conf
    word2vec = KeyedVectors.load_word2vec_format(conf['w2v_path'], binary=True)
    word_dict = pkl.load(open(conf['brown__dict_file'], 'rb'))

    embedding_mat = create_embedding_matrix(word_dict, word2vec)
    classes = len(word_dict) + 1

    model = create_language_model(conf, classes + 1, embedding_mat)
    train_language_model(model, conf, word_dict, classes, base_path)
    print(test_language_model(model, conf, word_dict, len(word_dict) + 1))


def curriculum_model():

    np.random.seed(42)

    base_path = 'Models\\%s\\' % time.strftime('%Y_%m_%d-%H_%M')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    shutil.copy('config.py', '%s//config.py' % base_path)

    conf = config.conf
    word_dict = pkl.load(open(conf['brown__dict_file'], 'rb'))
    word2cluster = general.read_brown_clusters(conf['brown__clusters_file'])
    embedding_mat = None
    softmax_mat = None
    softmax_bias = None
    lstm_weights = None

    word2id, classes = general.create_cluster_dict(word2cluster, 1)

    for i in range(15):

        if i > 0:
            old_word2id = word2id
            word2id, classes = general.create_cluster_dict(word2cluster, i + 1)

            embedding_mat, softmax_mat, softmax_bias = general.expand_all_matrices(embedding_mat,
                                                                                   softmax_mat,
                                                                                   softmax_bias,
                                                                                   old_word2id,
                                                                                   word2id,
                                                                                   classes)

        print("Iteration %d, Classes: %d" % (i+1, classes))

        model = create_language_model(conf, classes + 1, embedding_mat, softmax_mat, softmax_bias, lstm_weights)
        train_language_model(model, conf, word2id, classes, '%s\\%.2d\\' % (base_path, i))
        print(test_language_model(model, conf, word2id, classes))

        embedding_mat = model.get_layer('Embedding').get_weights()[0]
        softmax_mat, softmax_bias = model.get_layer('Softmax').get_weights()
        lstm_weights = model.get_layer('LSTM').get_weights()

    classes = len(word_dict) + 1

    embedding_mat, softmax_mat, softmax_bias = general.expand_all_matrices(embedding_mat,
                                                                           softmax_mat,
                                                                           softmax_bias,
                                                                           word2id,
                                                                           word_dict,
                                                                           classes + 1)

    model = create_language_model(conf, classes + 1, embedding_mat, softmax_mat, softmax_bias, lstm_weights)
    train_language_model(model, conf, word_dict, classes, base_path)
    print(test_language_model(model, conf, word_dict, classes))


if __name__ == '__main__':
    #baseline_model()
    curriculum_model()


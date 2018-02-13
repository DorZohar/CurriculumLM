import keras
import config
import numpy as np
import pickle as pkl
import keras.backend as K
from gensim.models.keyedvectors import KeyedVectors
from generators import brown_generator


def perplexity(y_true, y_pred):
    return K.pow(K.constant(2.0), K.mean(K.sparse_categorical_crossentropy(y_true, y_pred)))


def create_embedding_matrix(word_dict, word2vec):

    classes = len(word_dict) + 1
    vec_size = word2vec.vector_size
    matrix = np.zeros((classes, vec_size))

    for word, idx in word_dict.items():
        if word in word2vec:
            matrix[idx] = word2vec[word]

    return matrix


def create_language_model(conf, word_dict, word2vec):

    classes = len(word_dict) + 1

    input_layer = keras.layers.Input((None, ), name='Input')
    embedding_mat = create_embedding_matrix(word_dict, word2vec)
    embedding_layer = keras.layers.Embedding(input_dim=embedding_mat.shape[0],
                                             output_dim=embedding_mat.shape[1],
                                             weights=[embedding_mat],
                                             mask_zero=True)(input_layer)
    LSTM_layer = keras.layers.LSTM(conf['lstm__hidden_size'],
                                   activation=conf['lstm__activation'],
                                   recurrent_dropout=conf['lstm__rec_dropout'],
                                   dropout=conf['lstm__input_dropout'],
                                   return_sequences=True)(embedding_layer)

    output_layer = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation='softmax'))(LSTM_layer)

    model = keras.models.Model(input_layer, output_layer)

    model.compile(optimizer='RMSprop',
                  loss='sparse_categorical_crossentropy',
                  sample_weight_mode='temporal',
                  weighted_metrics=['accuracy', perplexity])

    return model


def train_language_model(model, train_gen, valid_gen, conf):

    checkpoint = keras.callbacks.ModelCheckpoint(conf['model_paths'],
                                                 save_best_only=True,
                                                 mode='min',
                                                 verbose=conf['verbose'])

    model.fit_generator(train_gen,
                        steps_per_epoch=conf['train_steps'] / conf['batch_size'],
                        epochs=conf['epochs'],
                        validation_data=valid_gen,
                        validation_steps=conf['valid_steps'] / conf['batch_size'],
                        workers=conf['workers'],
                        callbacks=[checkpoint],
                        verbose=conf['verbose'])


def test_language_model(model, test_gen, conf):
    scores = model.evalute_generator(test_gen,
                                     steps_per_epoch=conf['test_steps'] / conf['batch_size'],
                                     verbose=conf['verbose'])

    return scores


def main():
    conf = config.conf
    word2vec = KeyedVectors.load_word2vec_format(conf['w2v_path'], binary=True)
    word_dict = pkl.load(open(conf['brown__dict_file'], 'rb'))

    train_gen = brown_generator(conf['brown__train_file'],
                                conf['batch_size'],
                                conf['max_len'],
                                False)

    valid_gen = brown_generator(conf['brown__valid_file'],
                                conf['batch_size'],
                                conf['max_len'],
                                False)

    test_gen = brown_generator(conf['brown__test_file'],
                               conf['batch_size'],
                               conf['max_len'],
                               True)

    model = create_language_model(conf, word_dict, word2vec)
    train_language_model(model, train_gen, valid_gen, conf)
    print(test_language_model(model, test_gen, conf))


if __name__ == '__main__':
    main()


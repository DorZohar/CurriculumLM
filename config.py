

conf = {
    'brown__rare_word_count': 4,
    'brown__dict_file': 'brown_corpus.pkl',
    'brown__train_file': 'brown_train.txt',
    'brown__valid_file': 'brown_valid.txt',
    'brown__test_file': 'brown_test.txt',
    'brown__train_size': 800000,
    'brown__valid_size': 200000,

    'lstm__hidden_size': 300,
    'lstm__rec_dropout': 0.2,
    'lstm__input_dropout': 0.2,
    'lstm__activation': 'tanh',

    'w2v_path': "C:\\Wiki\\wiki.word2vec.bin",
    'model_paths': 'Models\\model_{epoch:02d}_{val_loss:.2f}.hdf5',

    'batch_size': 500,
    'max_len': 50,
    'epochs': 10,
    'train_steps': 800000,
    'valid_steps': 200000,
    'test_steps': 218532,
    'workers': 4,

    'verbose': 1,

}
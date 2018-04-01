

conf = {
    'brown__rare_word_count': 100,
    'brown__dict_file': 'brown_corpus.pkl',
    'brown__train_file': 'brown_train.txt',
    'brown__valid_file': 'brown_valid.txt',
    'brown__test_file': 'brown_test.txt',
    'brown__clusters_file': 'brown_clusters.txt',
    'brown__file': 'brown.txt',
    'brown__train_size': 0.7,
    'brown__valid_size': 0.15,

    'lstm__embedding_size': 300,
    'lstm__hidden_size': 300,
    'lstm__rec_dropout': 0.2,
    'lstm__input_dropout': 0.2,
    'lstm__activation': 'tanh',
    'lstm__learn_rate': 0.001,
    'lstm__momentum': 0.9,
    'lstm__sgd_optimizer': False,
    'lstm__limit_gpu': False,
    'lstm__gpu_id': 1,
    'lstm__weight_tying': False,

    'curriculum__input': False,
    'curriculum__output': False,
    'curriculum__start': 0,
    'curriculum__stop': 15,
    'curriculum__jumps': 2,

    'w2v_path': "C:\\Wiki\\wiki.word2vec.bin",
    'model_paths': 'model_{epoch:02d}_{val_loss:.2f}.hdf5',

    'batch_size': 50,
    'max_len': 40,
    'epochs': 1, # 50
    'mini_epochs': 10,
    'train_steps': 450,
    'valid_steps': 500,
    'test_steps': 1000,
    'workers': 4,

    'verbose': 1,

    # Can be LM, POS, NER, CHNK
    'task': 'POS',

}

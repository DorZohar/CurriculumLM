from nltk.corpus import brown
import config
from collections import Counter
import pickle as pkl


END_WORD = 'END'


def print_set_to_file(words, word_dict, path):

    unk_symbol = str(len(word_dict))
    end_symbol = str(word_dict[END_WORD])
    word_ids = [str(word_dict[word]) if word in word_dict else unk_symbol for word in words]

    str_word_ids = ' '.join(word_ids)
    str_lines = str_word_ids.replace(" %s " % end_symbol, ' %s\n' % end_symbol)

    with open(path, 'w') as f:
        f.write(str_lines)


def preprocess_brown(conf):

    words = []
    for sent in brown.sents():
        words += sent + [END_WORD]

    print("Number of words: %d" % len(words))

    cnt = Counter(words)
    print("Number of unique words: %d" % len(cnt))

    word_dict = {word: idx for idx, (word, count) in enumerate(cnt.most_common(), 1)
                 if count >= conf['brown__rare_word_count']}

    print("Number of common unique words (Freq >= %d): %d" % (conf['brown__rare_word_count'], len(word_dict)))

    train = words[:conf['brown__train_size']]
    valid = words[conf['brown__train_size']:(conf['brown__train_size'] + conf['brown__valid_size'])]
    test = words[(conf['brown__train_size'] + conf['brown__valid_size']):]

    print("Train size: %d, Validation size: %d, Test size: %d" % (len(train), len(valid), len(test)))

    print_set_to_file(train, word_dict, conf['brown__train_file'])
    print_set_to_file(valid, word_dict, conf['brown__valid_file'])
    print_set_to_file(test, word_dict, conf['brown__test_file'])
    print_set_to_file(words, word_dict, conf['brown__file'])

    pkl.dump(word_dict, open(conf['brown__dict_file'], 'wb'))


if __name__ == '__main__':
    conf = config.conf
    preprocess_brown(conf)

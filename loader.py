# encoding:utf-8
from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import codecs
import re
import jieba


def read_file(filename):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    with codecs.open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        seglist = jieba.lcut(blk)
                        word.extend([w for w in seglist if w not in stopwords])
                contents.append(word)
            except:
                pass
    return labels, contents


def build_vocab(filenames, vocab_dir, vocab_size=8000):
    all_data = []
    for filename in filenames:
        _, data_train = read_file(filename)
        for content in data_train:
            all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)

    with codecs.open(vocab_dir, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    words = codecs.open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    labels = ['neg',"pos"]
    cat_to_id = dict(zip(labels, range(len(labels))))
    return labels, cat_to_id


def process_file(filename, word_to_id, cat_to_id, max_length=200):
    labels, contents = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id)
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def export_word2vec_vectors(vocab, word2vec_dir, trimmed_filename):
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_training_word2vec_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]


def get_sequence_length(x_batch):
    sequence_lengths = []
    for x in x_batch:
        actual_length = np.sum(np.sign(x))
        sequence_lengths.append(actual_length)
    return sequence_lengths
#encoding:utf-8
from sentiment_model import *
import  tensorflow as tf
import tensorflow.contrib.keras as kr
import os
import numpy as np
import jieba
import re
import heapq
import codecs

def predict(sentences):
    config = TextConfig()
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)
    model = TextRNN(config)
    save_dir = './checkpoints/sentimentclassify'
    save_path = os.path.join(save_dir, 'best_validation')

    _,word_to_id=read_vocab(config.vocab_filename)
    input_x= process_file(sentences,word_to_id,max_length=config.seq_length)
    labels = {0:'neg',
              1:'pos',
              }
    feed_dict = {
        model.input_x: input_x,
        model.keep_prob: 1,
        model.sequence_lengths: get_sequence_length(input_x)
    }
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    y_prob=session.run(tf.nn.softmax(model.logits), feed_dict=feed_dict)
    y_prob=y_prob.tolist()
    cat=[]
    for prob in y_prob:
        top2= list(map(prob.index, heapq.nlargest(1, prob)))
        cat.append(labels[top2[0]])
    tf.reset_default_graph()
    return  cat

def sentence_cut(sentences):

    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")
    with codecs.open('./data/stopwords.txt','r',encoding='utf-8') as f:
            stopwords=[line.strip() for line in f.readlines()]
    contents=[]
    for sentence in sentences:
        words=[]
        blocks = re_han.split(sentence)
        for blk in blocks:
            if re_han.match(blk):
                seglist = jieba.lcut(blk)
                words.extend([w for w in seglist if w not in stopwords])
        contents.append(words)
    return  contents


def process_file(sentences,word_to_id,max_length=600):
    data_id=[]
    seglist=sentence_cut(sentences)
    for i in range(len(seglist)):
        data_id.append([word_to_id[x] for x in seglist[i] if x in word_to_id])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length)
    return x_pad


def read_vocab(vocab_dir):
    words = codecs.open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def get_training_word2vec_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]

def get_sequence_length(x_batch):

    sequence_lengths=[]
    for x in x_batch:
        actual_length = np.sum(np.sign(x))
        sequence_lengths.append(actual_length)
    return sequence_lengths





if __name__ == '__main__':
    print('predict random five samples in test data.... ')
    sentences =["淘宝就是坑!",
                "天猫质量不错，功能也算较齐全。铃也还可以。价格也能接受。性价比比较高",
                "拼多多都是假货!!!"
                ]
    cat=predict(sentences)
    for i,sentence in enumerate(sentences,0):
        print ('----------------------the sentiment classify-------------------------')
        print("{0}的情感偏向于{1}".format(sentence,cat[i]))


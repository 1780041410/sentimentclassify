# -*- coding: utf-8 -*-
import tensorflow as tf

class TextConfig(object):

    embedding_dim = 100
    vocab_size =8000
    pre_trianing=None

    seq_length=200
    num_classes=2

    num_layers= 1
    hidden_dim = 128
    attention_size = 100


    keep_prob=0.5
    learning_rate= 1e-3
    lr_decay= 0.9
    grad_clip= 5.0

    num_epochs=10
    batch_size=64
    print_per_batch =100

    train_filename='./data/train.txt'
    test_filename='./data/test.txt'
    vocab_filename = './data/vocab.txt'  # vocabulary
    vector_word_filename='./data/vector_word.txt'
    vector_word_npz='./data/vector_word.npz'


class TextRNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.rnn()

    def rnn(self):

        def basic_rnn_cell_LSTM(rnn_size):

            return tf.contrib.rnn.LSTMCell(rnn_size,state_is_tuple=True)
        def basic_rnn_cell_GRU(rnn_size):
            return tf.contrib.rnn.GRUCell(rnn_size)

        with tf.name_scope('LSTM'):
            lstm_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell_LSTM(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            lstm_rnn_cell = tf.contrib.rnn.DropoutWrapper(lstm_rnn_cell, output_keep_prob=self.keep_prob)

        with tf.name_scope('GRU'):
            gru_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell_GRU(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            gru_rnn_cell = tf.contrib.rnn.DropoutWrapper(gru_rnn_cell, output_keep_prob=self.keep_prob)


        with tf.name_scope('embedding'):
            # self.embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0), trainable=False,name='W')

            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_dim],
                                         initializer=tf.constant_initializer(self.config.pre_trianing))
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)


        with tf.name_scope('LSTM_GRU'):
            rnn_output_lstm, _ = tf.nn.dynamic_rnn(lstm_rnn_cell, inputs=embedding_inputs, sequence_length=self.sequence_lengths, dtype=tf.float32)
            rnn_output_gru, _ = tf.nn.dynamic_rnn(gru_rnn_cell, inputs=embedding_inputs, sequence_length=self.sequence_lengths,
                                                   dtype=tf.float32)

        rnn_output = tf.concat([rnn_output_lstm,rnn_output_gru], 2)
        with tf.name_scope('attention'):
            input_shape = rnn_output.shape
            sequence_size = input_shape[1].value
            hidden_size = input_shape[2].value
            attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.config.attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.config.attention_size]), name='attention_b')
            attention_u = tf.Variable(tf.truncated_normal([self.config.attention_size], stddev=0.1), name='attention_u')
            z_list = []
            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(rnn_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
                z_list.append(z_t)
            # Transform to batch_size * sequence_size
            attention_z = tf.concat(z_list, axis=1)
            self.alpha = tf.nn.softmax(attention_z)
            attention_output = tf.reduce_sum(rnn_output * tf.reshape(self.alpha, [-1, sequence_size, 1]), 1)

        with tf.name_scope('dropout'):

            self.final_output = tf.nn.dropout(attention_output, self.keep_prob)

        with tf.name_scope('output'):
            fc_w = tf.Variable(tf.truncated_normal([hidden_size, self.config.num_classes], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)


        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)


        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))





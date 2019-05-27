from __future__ import print_function
from sentiment_model import *
from loader import *
from sklearn import metrics
import sys
import os
import time
from datetime import timedelta

def feed_data(x_batch, y_batch, keep_prob):
    sequence_lengths = get_sequence_length(x_batch)
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob:keep_prob,
        model.sequence_lengths: sequence_lengths
    }
    return feed_dict

def train():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard/sentimentclassify'
    save_dir = './checkpoints/sentimentclassify'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    print("Loading training and validation data...")
    start_time = time.time()
    x_train, y_train = process_file(config.train_filename, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(config.test_filename, word_to_id, cat_to_id, config.seq_length)
    print("Time cost: %.3f seconds...\n" % (time.time() - start_time))

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    best_val_accuracy = 0
    last_improved = 0  # record global_step at best_val_accuracy
    require_improvement = 1000  # break training if not having improvement over 1000 iter
    flag=False

    for epoch in range(config.num_epochs):
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        start = time.time()
        print('Epoch:', epoch + 1)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.keep_prob)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                    merged_summary, model.loss,
                                                                                    model.acc], feed_dict=feed_dict)
            if global_step % config.print_per_batch == 0:
                end = time.time()
                feed_dict = feed_data(x_val, y_val, 1.0)
                val_summaries, val_loss, val_accuracy = session.run([merged_summary, model.loss, model.acc],
                                                                 feed_dict=feed_dict)
                writer.add_summary(val_summaries, global_step)
                print()
                if val_accuracy > best_val_accuracy:
                    saver.save(session, save_path)
                    best_val_accuracy = val_accuracy
                    last_improved=global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                        global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                        (end - start) / config.print_per_batch,improved_str))
                start = time.time()

            if global_step - last_improved > require_improvement:
                print("No optimization over 1000 steps, stop training")
                flag = True
                break
        if flag:
            break
        config.learning_rate *= config.lr_decay
if __name__ == '__main__':
    print('Configuring lstm+GRU+Attention model...')
    config = TextConfig()
    filenames = [config.train_filename]
    if not os.path.exists(config.vocab_filename):
        build_vocab(filenames, config.vocab_filename, config.vocab_size)

    categories,cat_to_id = read_category()
    words,word_to_id = read_vocab(config.vocab_filename)
    config.vocab_size = len(words)

    if not os.path.exists(config.vector_word_npz):
        export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)

    model = TextRNN(config)
    train()

#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import batch_iter, process_topics

base_dir = 'data/blogs'
train_dir = os.path.join(base_dir, 'train.data')
test_dir = os.path.join(base_dir, 'blogs.test2.txt')
val_dir = os.path.join(base_dir, 'val.data')
vocab_dir = os.path.join(base_dir, 'vocab.txt')
topic_dir = os.path.join(base_dir, 'topics.txt')

save_dir = 'checkpoints/test'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(topic_pos_batch, topic_neg_batch, blog_batch, keep_prob):
    feed_dict = {
        model.topic_pos: topic_pos_batch,
        model.topic_neg: topic_neg_batch,
        model.blog: blog_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess):
    """评估在某一数据上的准确率和损失"""
    data_len = len(topics_pad)
    batch_eval = batch_iter(val_dir, vocab_dir, topics_pad, config.neg_num, config.seq_length, config.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for topic_pos_batch, topic_neg_batch, blog_batch in batch_eval:
        batch_len = len(topic_pos_batch)
        feed_dict = feed_data(topic_pos_batch, topic_neg_batch, blog_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    '''
    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    topic_train, blog_train = process_file(train_dir, vocab_dir, config.seq_length)
    topic_val, blog_val = process_file(val_dir, vocab_dir, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    '''

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # saver.restore(sess=session, save_path=save_path)
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_dir, vocab_dir, topics_pad, config.neg_num, config.seq_length, config.batch_size)
        for topic_pos_batch, topic_neg_batch, blog_batch in batch_train:
            feed_dict = feed_data(topic_pos_batch, topic_neg_batch, blog_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                a = session.run(model.test, feed_dict=feed_dict)
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1


if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TCNNConfig()
    model = TextCNN(config)
    topics_pad = process_topics(topic_dir, vocab_dir)

    train()
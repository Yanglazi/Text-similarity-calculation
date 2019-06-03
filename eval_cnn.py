#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
import six

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import sentences2id, open_file, tokenization

base_dir = 'data/blogs'
train_dir = os.path.join(base_dir, 'blogs.train.txt')
test_dir = os.path.join(base_dir, 'blogs.test2.txt')
val_dir = os.path.join(base_dir, 'blogs.val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

if six.PY2:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if six.PY2:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if six.PY2:
        return content.decode('utf-8')
    else:
        return content


def read_file(filename):
    """读取文件数据"""
    items = []
    with open_file(filename) as f:
        for line in f:
            item = line.strip()
            if item:
                items.append(item)
    return items


def process_file(filename, vocab_file, max_length=600):
    """将文件转换为id表示"""
    items = read_file(filename)

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    item_pad = sentences2id(items, tokenizer, max_length)
    return item_pad


def batch_iter(items, batch_size=64):
    """生成批次数据"""
    data_len = len(items)
    num_batch = int((data_len - 1) / batch_size)

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield items[start_id:end_id]


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def blog_to_features(filename, output_path='output/blogs_feature.npy'):
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
    print('Topic feature generating...')

    items = process_file(filename, vocab_dir, config.topic_length)
    batch_data = batch_iter(items, config.batch_size)

    blog_features = []
    for blog_batch in batch_data:
        blog_feature = session.run(model.blog_feature,
                                   feed_dict={model.blog: blog_batch, model.keep_prob: 1.0})
        blog_features.append(blog_feature)
    blog_features = np.concatenate(blog_features, axis=0)
    np.save(output_path, blog_features)
    print('done!')


def topic_to_features(filename, output_path='output/topics_feature.npy'):
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
    print('Topic feature generating...')

    items = process_file(filename, vocab_dir, config.topic_length)
    batch_data = batch_iter(items, 128)

    topic_features = []
    for topic_batch in batch_data:
        topic_feature = session.run(model.topic_feature,
                                    feed_dict={model.topic_pos: topic_batch, model.keep_prob: 1.0})
        topic_features.append(topic_feature)
    topic_features = np.concatenate(topic_features, axis=0)
    np.save(output_path, topic_features)
    print('done!')


def cosin(topics_feature, blog_feature):
    topics_norm = np.sqrt(np.sum(np.square(topics_feature), axis=1)).reshape(-1, 1)
    blog_feature = blog_feature.reshape(1, -1)
    blog_norm = np.sqrt(np.sum(np.square(blog_feature)))
    cosin = np.sum(blog_feature * topics_feature, axis=1) / (topics_norm * blog_norm).flatten()
    return cosin


def gen_hard_examples(input_file=train_dir, output_file='model_train.data'):
    topics_feature = np.load('data/topics_feature.npy')
    blogs_feature = np.load('data/blogs_feature.npy')

    blog_file = open_file(input_file, 'r')
    real_topics = []
    for line in blog_file:
        topic = line.split('\t')[0]
        real_topics.append(topic)
    print(len(real_topics))
    blog_file.close()

    print('Hard examples generating...')
    f = open_file(output_file, 'w')
    for i, blog_feature in enumerate(blogs_feature):
        dis = cosin(topics_feature, blog_feature)
        dis_idx = np.argsort(-dis)[:128]
        hard_topics = [topics[i] for i in dis_idx]

        f.write('%s\t%s\n' % (real_topics[i], '\t'.join(hard_topics)))
    f.close()
    print("done!")


def demo():
    topics_feature = np.load('output/topics_feature.npy')
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_dir, do_lower_case=True)

    while True:
        if sys.version_info[0] > 2:
            blog_text = input("Input a blog:\n")
        else:
            blog_text = raw_input("Input a blog:\n")

        blog_pad = sentences2id([blog_text], tokenizer, config.seq_length)

        blog_feature = session.run(model.blog_feature,
                                   feed_dict={model.blog: blog_pad, model.keep_prob: 1.0})
        dis = cosin(topics_feature, blog_feature)
        dis_idx = np.argsort(-dis)[:20]

        hard_topics = [topics[i] for i in dis_idx]
        for topic in hard_topics:
            print(topic)


def test(filename):
    topics_feature = np.load('output/topics_feature.npy')

    blog_file = open_file(filename, 'r')
    real_topics, blogs = [], []
    for line in blog_file:
        try:
            fields = line.strip().split('\t')
            topic = fields[0]
            blog = fields[1]
            real_topics.append(topic)
            blogs.append(blog)
        except:
            pass
    blog_file.close()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
    print('Test...')

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_dir, do_lower_case=True)

    blogs_pad = sentences2id(blogs, tokenizer, config.seq_length)
    batch_data = batch_iter(blogs_pad, 1)

    start_time = time.time()
    top1_count = 0
    top5_count5 = 0
    for i, blog_batch in enumerate(batch_data):
        blog_feature = session.run(model.blog_feature, feed_dict={model.blog: blog_batch, model.keep_prob: 1.0})

        dis = cosin(topics_feature, blog_feature)
        dis_idx = np.argsort(-dis)[:5]

        predict_topics = [topics[i] for i in dis_idx]
        if real_topics[i] in predict_topics:
            top5_count5 += 1
        if real_topics[i] == predict_topics[0]:
            top1_count += 1

    acc1 = float(top1_count) / len(real_topics)
    acc5 = float(top5_count5) / len(real_topics)
    print("top1-acc:%f\ttop5-acc:%f" % (acc1, acc5))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TCNNConfig()
    model = TextCNN(config)

    topics_dir = 'output/topics.txt'
    topics = read_file(topics_dir)
    #topic_to_features(topics_dir, 'output/topics_feature.npy')

    demo()
    #test('output/test4')
    #test('data/blogs/blogs.val.txt')

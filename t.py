# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
import data.tokenization as tokenization
import six
import random

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


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if six.PY3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    topics, blogs = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                topic, blog = line.split('\t')
                if topic and blog:
                    topics.append(topic.strip())
                    blogs.append(blog.strip())
            except:
                pass
    return topics, blogs


def sentences2id(sentences, vocab_file, max_length=600):
    sentences_id = []

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    for i in range(len(sentences)):
        tokens = tokenizer.tokenize(sentences[i])
        token_id = tokenizer.convert_tokens_to_ids(tokens)
        sentences_id.append(token_id)

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    topic_pad = kr.preprocessing.sequence.pad_sequences(sentences_id, max_length, padding='post')
    return topic_pad

def process_topic(filename,vocab_file,max_length=32):
    label_topics = []
    f = open_file(filename, 'r')
    for line in f:
        label_topics.append(line.strip())
    f.close()

    label_topics_pad = sentences2id(label_topics, vocab_file, max_length)
    return label_topics_pad


def process_file(filename, vocab_file, max_length=600):
    """将文件转换为id表示"""
    topics, blogs = read_file(filename)

    topics_id, blogs_id = [], []

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    for i in range(len(topics)):
        tokens_topic = tokenizer.tokenize(topics[i])
        tokens_blog = tokenizer.tokenize(blogs[i])
        topic_id = tokenizer.convert_tokens_to_ids(tokens_topic)
        blog_id = tokenizer.convert_tokens_to_ids(tokens_blog)
        topics_id.append(topic_id)
        blogs_id.append(blog_id)

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    topic_pad = kr.preprocessing.sequence.pad_sequences(topics_id, 32, padding='post')
    blog_pad = kr.preprocessing.sequence.pad_sequences(blogs_id, max_length, padding='post')

    return topic_pad, blog_pad


def batch_iter(topic, blog, label_topics_pad, neg_num, batch_size=64):
    """生成批次数据"""
    data_len = len(topic)
    num_batch = int((data_len - 1) / batch_size)

    indices = np.random.permutation(np.arange(data_len))
    topic_pos_shuffle = topic[indices]
    blog_shuffle = blog[indices]

    topic_len = len(label_topics_pad)

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)

        indices = np.random.choice(range(topic_len), neg_num * batch_size)
        topic_neg_shuffle = label_topics_pad[indices]

        yield topic_pos_shuffle[start_id:end_id], topic_neg_shuffle, blog_shuffle[start_id:end_id]

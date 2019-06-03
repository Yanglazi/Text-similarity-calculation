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


def sentences2id(sentences, tokenizer, max_length=600):
    sentences_id = []

    for i in range(len(sentences)):
        tokens = tokenizer.tokenize(sentences[i])
        token_id = tokenizer.convert_tokens_to_ids(tokens)
        sentences_id.append(token_id)

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    topic_pad = kr.preprocessing.sequence.pad_sequences(sentences_id, max_length, padding='post')
    return topic_pad


def process_topics(filename, vocab_file):
    topics = []
    f = open_file(filename, 'r')
    for line in f:
        topics.append(line.strip())
    f.close()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    topics = sentences2id(topics, tokenizer, 32)
    return topics


def batch_process(batch, tokenizer, max_length):
    sentences_id = []
    topics_pos, blogs = [], []
    for item in batch:
        # if item[0] and item[1]:
        topics_pos.append(native_content(item[0]))
        blogs.append(native_content(item[1]))
    topics_pos = sentences2id(topics_pos, tokenizer, 32)
    blogs = sentences2id(blogs, tokenizer, max_length)
    return topics_pos, blogs


def batch_iter(filename, vocab_file, topics, neg_num, max_length=600, batch_size=64):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    topic_len = len(topics)
    with open_file(filename) as f:
        batch = []
        for line in f:
            item = line.strip().split('\t')
            if len(item) != 2: continue
            batch.append(item)
            if len(batch) == batch_size:
                topics_pos, blogs = batch_process(batch, tokenizer, max_length)

                indices = np.random.choice(range(topic_len), neg_num * batch_size)
                topics_neg = topics[indices]
                batch = []

                yield topics_pos, topics_neg, blogs


if __name__ == '__main__':
    train_dir = 'blogs/blogs.train.txt'
    vocab_dir = 'blogs/vocab.txt'
    topics_pad = process_topics('blogs/blogs.train.txt', 'blogs/vocab.txt')
    batch_train = batch_iter(train_dir, vocab_dir, topics_pad, 3, 256, 8)
    count = 0
    for a, b, c in batch_train:
        count += 1
        print(count)

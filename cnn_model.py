# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 192  # 词向量维度
    seq_length = 256  # 序列长度
    topic_length = 32
    num_filters = 128  # 卷积核数目
    kernel_sizes = 3, 4, 5  # 卷积核尺寸
    vocab_size = 21128  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    neg_num = 256
    batch_size = 64
    batch_neg = neg_num * batch_size  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 5000  # 每多少轮输出一次结果
    save_per_batch = 10000  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.topic_pos = tf.placeholder(tf.int32, [None, self.config.topic_length], name='topic_pos')
        self.topic_neg = tf.placeholder(tf.int32, [None, self.config.topic_length], name='topic_neg')
        self.blog = tf.placeholder(tf.int32, [None, self.config.seq_length], name='blog')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.model()

    def cnn(self, input):
        """CNN模型"""
        # 词向量映射
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, input)

        pool_outputs = []
        for i, kernel_size in enumerate(self.config.kernel_sizes):
            with tf.variable_scope("conv%d" % i, reuse=tf.AUTO_REUSE):
                # CNN layer
                conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, kernel_size, name='conv')
                # global max pooling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                pool_outputs.append(gmp)
        pool_output = tf.concat(pool_outputs, axis=1)
        act = tf.nn.relu(pool_output)

        with tf.variable_scope("fc", reuse=tf.AUTO_REUSE):
            fc = tf.layers.dense(act, self.config.hidden_dim, name='fc1')
            dropout = tf.nn.dropout(fc, self.keep_prob)
        return dropout

    def model(self, ):
        topic_pos = self.cnn(self.topic_pos)
        topic_neg = self.cnn(self.topic_neg)
        blog = self.cnn(self.blog)

        self.topic_feature = topic_pos
        self.blog_feature = blog

        with tf.name_scope('merge_negative_topic'):
            # 合并负样本，tile可选择是否扩展负样本。
            topics = tf.concat([topic_pos, topic_neg], axis=0)

        with tf.name_scope('cosine_similarity'):
            # Cosine similarity
            # query_norm = sqrt(sum(each x^2))
            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(blog), 1, True)), [self.config.neg_num + 1, 1])
            # doc_norm = sqrt(sum(each x^2))
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(topics), 1, True))

            prod = tf.reduce_sum(tf.multiply(tf.tile(blog, [self.config.neg_num + 1, 1]), topics), 1, True)
            norm_prod = tf.multiply(query_norm, doc_norm)

            # cos_sim_raw = query * doc / (||query|| * ||doc||)
            cos_sim_raw = tf.truediv(prod, norm_prod)
            # gamma = 20
            self.test = tf.transpose(cos_sim_raw)
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [self.config.neg_num + 1, -1])) * 20
            self.test = cos_sim

        with tf.name_scope("loss"):
            # 全连接层，后面接dropout以及relu激活
            prob = tf.nn.softmax(cos_sim)
            hit_prob = tf.slice(prob, [0, 0], [-1, 1])
            self.loss = -tf.reduce_sum(tf.log(hit_prob))

        with tf.name_scope("optimize"):
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            self.acc = tf.reduce_mean(tf.floor(hit_prob + 0.5))

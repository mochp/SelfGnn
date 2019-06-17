# -*- coding: utf-8 -*-
# @Time : 2019/6/17 9:40
# @Author : mochp

import tensorflow as tf
import utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GNN(object):
    def __init__(self, answers, feature, config, batch_size, is_training=True):
        # var
        self.num_answer = answers.shape[1]
        self.num_feature = feature.shape[1]
        self.dim_answer = config["answers"]["dim"]
        self.dim_ques = config["question_var"]
        self.expend = config["feature"]["dim"]
        self.hidden_size = config["hidden_size"]
        self.is_training = is_training
        self.batch_size = batch_size

        self.logits = {}
        self.loss = {}
        self.total_loss = 0

        # computer var
        self.total_dim = self.dim_ques * self.dim_answer

        # create var
        self.ques_var = tf.Variable(tf.random_normal(
            [self.num_answer, self.dim_ques], stddev=0.35), dtype=tf.float32)
        self.w = tf.Variable(tf.random_normal(
            [self.total_dim, self.total_dim], stddev=0.35), dtype=tf.float32)

        # create placeholders
        self.answer_ph = tf.placeholder(
            tf.float32, shape=[None, self.num_answer])
        self.feature_ph = tf.placeholder(
            tf.float32, shape=[None, self.num_feature])
        self.feature_tile_ph = tf.placeholder(
            tf.float32, shape=[
                None, self.num_feature, self.total_dim])


        # create graph
        self.create_model_graph()

    def create_feed_dict(self, answer, feature, feature_tile):
        feed_dict = {
            self.answer_ph: answer,
            self.feature_ph: feature,
            self.feature_tile_ph: feature_tile
        }
        return feed_dict

    def test_dict(self):
        answers, feature, config = utils.create_test()
        data = utils.DataStream(answers, feature, config, 4)
        answer, feature, feature_tile = data.next_feed_dict(100)
        feed_dict = {
            self.answer_ph: answer,
            self.feature_ph: feature,
            self.feature_tile_ph: feature_tile
        }
        return feed_dict

    @staticmethod
    def compute_gradients(tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [
            grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]

    def create_model_graph(self):
        # create graph
        # 0.拉伸W
        ques1 = tf.expand_dims(self.ques_var, axis=0)
        ques2 = tf.tile(ques1, [self.batch_size, 1, 1])
        self.tile_ques = tf.tile(ques2, [1, 1, self.dim_answer])

        # 1.拉伸answer
        ans1 = tf.cast(self.answer_ph, dtype=tf.int32)
        ans2 = tf.one_hot(ans1, self.dim_answer)

        # 2.扩增answer 100dim
        ans3 = tf.tile(tf.expand_dims(ans2, -1), [1, 1, 1, self.dim_ques])
        ans4 = tf.reshape(ans3, [-1, self.num_answer, self.total_dim])
        self.tile_ans = tf.multiply(ans4, self.tile_ques)

        self.ans_fea_ph = tf.concat((self.answer_ph, self.feature_ph), axis=1)
        # 2.合并feature
        self.ans_fea_tile = tf.concat(
            (self.tile_ans, self.feature_tile_ph), axis=1)

        # 3.attention
        self.tile_w = tf.tile(
            tf.expand_dims(
                self.w, axis=0), [
                self.batch_size, 1, 1])
        # self.tile_w = tf.tile(w2, [self.dim_answer, 1, 1])

        ans_fea_opposite = tf.transpose(self.ans_fea_tile, [0, 2, 1])
        attention1 = tf.matmul(self.ans_fea_tile, self.tile_w)
        self.attention = tf.nn.softmax(tf.matmul(attention1, ans_fea_opposite))
        self.ans_fea_attention = tf.matmul(self.attention, self.ans_fea_tile)

        # 映射向量
        flatten = (self.num_answer + self.num_feature) * self.total_dim
        reduce = tf.reshape(
            self.ans_fea_attention, [
                self.batch_size, 1, flatten])
        reduce1 = tf.layers.dense(inputs=reduce, units=self.hidden_size)
        hidden = tf.reshape(reduce1, [self.batch_size, self.hidden_size])

        # logit-answer
        for i in range(self.num_answer + self.num_feature):
            unit = self.dim_answer if i < self.num_answer else self.expend[i - self.num_answer]
            self.logits[i] = tf.layers.dense(inputs=hidden, units=unit)
            sli1 = tf.slice(self.ans_fea_ph, [0, 0], [-1, 1])
            sli2 = tf.one_hot(tf.cast(sli1, dtype=tf.int32), unit)
            sli3 = tf.reshape(sli2, [-1, unit])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits[i], labels=sli3)
            self.loss[i] = tf.reduce_mean(cross_entropy)
            self.total_loss += self.loss[i]

        # 非训练过程跳过
        if not self.is_training:
            return

        # 8.梯度下降
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        trainable_variables = tf.trainable_variables()
        grads = self.compute_gradients(self.total_loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10)
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables), global_step=global_step)


if __name__ == '__main__':
    # 测试数据
    a, b, c = utils.create_test()
    gnn = GNN(a, b, c, 4, is_training=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    _, logits, loss = sess.run(
        [gnn.train_op, gnn.logits, gnn.loss], feed_dict=gnn.test_dict())
    print(logits)
    print(loss)

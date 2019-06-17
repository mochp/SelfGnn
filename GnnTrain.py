# -*- coding: utf-8 -*-
import utils
import time
import tensorflow as tf
from GnnModel import GNN


restore = False
best_path = "model/gnn.best.model"

answers, feature, config = utils.create_test()
train_graph = GNN(answers, feature, config, 4, is_training=True)

vars_ = {}
for var in tf.global_variables():
    vars_[var.name.split(":")[0]] = var
saver = tf.train.Saver(vars_)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
if restore:
    print("Restoring model from " + best_path)
    saver.restore(sess, best_path)
    print("DONE!")


for epoch in range(epochs):
    print('Train in epoch %d' % epoch)
    start_time = time.time()
    data = utils.DataStream(answers, feature, config, 4)
    answer, feature, feature_tile = data.next_feed_dict(100)
    feed_dict = train_graph.create_feed_dict(answer, feature, feature_tile)
    _, logits, loss = sess.run(
        [train_graph.train_op, train_graph.logits, train_graph.loss], feed_dict=feed_dict)

    print(
        'epoch:{},user:{},mask_num:{},acc:{}'.format(
            epoch, user, mask_num, acc))

    print("use time", time.time() - start_time)
    saver.save(sess, best_path)

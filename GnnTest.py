# -*- coding: utf-8 -*-
import os
import utils
import tensorflow as tf
from GnnModel import GNN
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

questions = utils.get_question("data/BIG5/b.pkl")

best_path = "model/gnn.best.model"
num_dim_ans = 5

test_graph = GNN(questions,num_dim_ans,is_training=True)

vars_ = {}
for var in tf.global_variables():
    vars_[var.name.split(":")[0]] = var
saver = tf.train.Saver(vars_)
 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Restoring model from " + best_path)
saver.restore(sess, best_path)
print("DONE!")

#attention = sess.run(train_graph.attention)
#training

dic = {}
dic["total"] = 0
for user in range(15776,19718):
    dic["total"]+=1
    loss = []
    answer,mask = utils.TestDataStream("data/BIG5/data.csv").user_feed_dict(user)
    feed_dict = test_graph.create_feed_dict(answer,mask)
    accuracy,correct_list = sess.run([test_graph.accuracy,test_graph.correct_list], feed_dict=feed_dict)
    loss.append(accuracy)
    
    for i in range(50):
        if i in dic:
            dic[i]+=correct_list[51*i]
        else:
            dic[i] = 0
            dic[i]+=correct_list[51*i]

    print('user:{},accuracy:{}'.format(user,accuracy))


result = []

for i in range(50):
    result.append(dic[i]/dic["total"])
    
print(sum(result)/len(result))

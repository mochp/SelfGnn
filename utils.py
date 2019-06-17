# -*- coding: utf-8 -*-
# @Time : 2019/6/17 9:40
# @Author : mochp

import numpy as np
import pandas as pd
import yaml


# with open('config.yml','rb') as file_config:
#    config = yaml.load(file_config)
#
#answers = pd.read_excel(config["answers"]["path"])
#feature = pd.read_excel(config["feature"]["path"])
#answers = np.array(answers)
#feature = np.array(feature)


class DataStream(object):
    def __init__(self, answers, feature, config, batch_size):
        assert len(answers) == len(feature), "error dim"
        self.answers = answers
        self.feature = feature
        self.batch_size = batch_size
        self.feature_dim = config["feature"]["dim"]
        self.answer_dim = config["answers"]["dim"]

        self.max_dim = 20 * self.answer_dim
        self.expend = [int(self.max_dim / x) for x in self.feature_dim]

    def user_answer(self, user_id):
        return [x - 1 for x in self.answers[user_id]]

    def user_feature(self, user_id):
        return [x - 1 for x in self.feature[user_id]]

    def user_feature_expand(self, user_id):
        feature = [x - 1 for x in self.feature[user_id]]
        result = []
        for i, fea in enumerate(feature):
            t1 = np.eye(self.feature_dim[i])[fea]
            t2 = np.tile(np.expand_dims(t1, axis=1), self.expend[i])
            t3 = np.resize(t2, -1)
            if len(t3) < self.max_dim:
                t3 = np.concatenate(
                    (t3, np.zeros(self.max_dim - len(t3))), axis=0)
            result.append(t3)
        return np.array(result)

    # feed方式
    def next_feed_dict(self, user_id):
        batch_answer = []
        batch_feature = []
        batch_feature_expand = []
        for _ in range(self.batch_size):
            user_id += 1
            batch_answer.append(self.user_answer(user_id))
            batch_feature.append(self.user_feature(user_id))
            batch_feature_expand.append(self.user_feature_expand(user_id))
        return np.array(batch_answer), np.array(
            batch_feature), np.array(batch_feature_expand)


def create_test():
    with open('config.yml', 'rb') as file_config:
        config = yaml.load(file_config)

    answers = np.load("answers.npy")
    feature = np.load("feature.npy")
    return answers, feature, config

#data = DataStream(answers, feature,config)
#B,C,D = data.next_feed_dict(100)


#answers = pd.read_excel(config["answers"]["path"])
#feature = pd.read_excel(config["feature"]["path"])
#answers = np.array(answers)
#feature = np.array(feature)
# np.save("answers.npy",answers)
# np.save("feature.npy",feature)

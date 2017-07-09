# -*- coding: utf-8 -*
# __author__ = 'Li Shang'

import pickle
from nlp.sdp.Transition import Transition


class Weights(object):

    def __init__(self):
        self._values = dict()
        # 总权重
        self._total_weights = dict()
        # 上次更新权重的时间戳
        self._time_stamp = dict()
        # 当前时间戳
        self.time = 0

        self._back_up = dict()

    def update(self, key, delta):
        if key not in self._values:
            self._values[key] = 0.0
            self._total_weights[key] = 0.0
            self._time_stamp[key] = self.time
        else:
            self._total_weights[key] += ((self.time - self._time_stamp[key])*self._values[key])
            self._time_stamp[key] = self.time

        self._values[key] += delta

    def get_value(self, key):
        if key not in self._values:
            return 0.0
        return self._values[key]

    def average(self):
        for k, v in self._total_weights.items():
            self._total_weights[k] += ((self.time - self._time_stamp[k])*self._values[k])
            self._time_stamp[k] = self.time

        self._back_up = dict(self._values)
        for k, v in self._total_weights.items():
            self._values[k] = self._total_weights[k] / float(self.time)

    def unaverage(self):
        self._values = dict(self._back_up)
        self._back_up.clear()

    def save(self, path):
        return pickle.dump(self._values, open(path, 'wb'))

    def load(self, path):
        self._values = pickle.load(open(path, 'rb'))


class AveragedPerceptron(object):

    def __init__(self):
        self._weights = Weights()
        # self._arc_types = arc_types
        #
        # self._tags = ['Shift', 'Pop', 'Mem', 'Recall']
        # for arc in self._arc_types:
        #     self._tags.append('LeftArc-' + arc)
        #     self._tags.append('RightArc-' + arc)

    def update_weights(self, features, true_tag, guess_tag):
        for feature in features:
            self._weights.update(feature + '-' + true_tag, 1.0)
            self._weights.update(feature + '-' + guess_tag, -1.0)

    def decode(self, allowed, features):
        max_score = float('-inf')
        guess_tag = '#'

        for tag in allowed:
            score = 0
            for feature in features:
                score += self._weights.get_value(feature + '-' + tag)
            if score > max_score:
                max_score = score
                guess_tag = tag

        return guess_tag

    def average(self):
        self._weights.average()

    def unaverage(self):
        self._weights.unaverage()

    def train(self, a_sentence):
        self._weights.time += 1

        trans_model = Transition(a_sentence, is_train=True)
        for features, allowed, true_tag in trans_model.gen_features():
            guess_tag = self.decode(allowed, features)
            # update weights
            if guess_tag != true_tag:
                self.update_weights(features, true_tag, guess_tag)
        # print(trans_model.stack1, trans_model.dependency)

    def predict(self, a_sentence):
        trans_model = Transition(a_sentence, is_train=False)
        for features, allowed, _ in trans_model.gen_features():
            guess_tag = self.decode(allowed, features)
            trans_model.do_a_transition(guess_tag)

        root = 0
        if len(trans_model.stack1) > 0:
            root = trans_model.stack1[0]
        # trans_model.dependency[(0, root)] = 'root'
        return trans_model.dependency, root

    def save(self, path):
        self._weights.save(path)

    def load(self, path):
        self._weights.load(path)

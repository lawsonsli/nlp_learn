# -*- coding: utf-8 -*
# __author__ = 'Li Shang'

import pickle


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

        # self.trans_limit = [[0, 0, float('-inf'), float('-inf'), 0, float('-inf'), float('-inf'), 0, float('-inf'), float('-inf')],
        #  [0, 0, 0, 0, 0, float('-inf'), float('-inf'), 0, float('-inf'), float('-inf')],
        #  [float('-inf'), float('-inf'), 0, 0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        #  [0, 0, 0, 0, 0, float('-inf'), float('-inf'), 0, float('-inf'), float('-inf')],
        #  [0, 0, float('-inf'), float('-inf'), 0, 0, 0, 0, float('-inf'), float('-inf')],
        #  [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), 0, 0, float('-inf'), float('-inf'), float('-inf')],
        #  [0, 0, float('-inf'), float('-inf'), 0, 0, 0, 0, float('-inf'), float('-inf')],
        #  [0, 0, float('-inf'), float('-inf'), 0, float('-inf'), float('-inf'), 0, 0, 0],
        #  [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), 0, 0],
        #  [0, 0, float('-inf'), float('-inf'), 0, float('-inf'), float('-inf'), 0, 0, 0]]

    @staticmethod
    def gen_features(words):
        for i in range(len(words)):
            left2 = words[i - 2] if i - 2 >= 0 else '#'
            left1 = words[i - 1] if i - 1 >= 0 else '#'
            mid = words[i]
            right1 = words[i + 1] if i + 1 < len(words) else '#'
            right2 = words[i + 2] if i + 2 < len(words) else '#'
            features = ['1' + mid, '2' + left1, '3' + right1,
                        '4' + left2 + left1, '5' + left1 + mid, '6' + mid + right1, '7' + right1 + right2]
            yield features

    def update_weights(self, words, true_tags, guess_tags):
        assert len(words) == len(true_tags)
        assert len(guess_tags) == len(true_tags)

        num = len(words)

        for i, features in enumerate(self.gen_features(words)):
            if true_tags[i] != guess_tags[i]:
                for feature in features:
                    self._weights.update(feature + '-' + str(true_tags[i]), 1.0)
                    self._weights.update(feature + '-' + str(guess_tags[i]), -1.0)

        for i in range(num-1):
            self._weights.update(str(true_tags[i]) + '->' + str(true_tags[i + 1]), 1.0)
            self._weights.update(str(guess_tags[i]) + '->' + str(guess_tags[i + 1]), -1.0)

    def viterbi_decode(self, words):
        transitions = [[self._weights.get_value(str(i) + '->' + str(j)) for j in range(10)]
                       for i in range(10)]
        emissions = [[sum(self._weights.get_value(feature + '-' + str(tag)) for feature in features)
                      for tag in range(10)] for features in self.gen_features(words)]
        # 限制转移矩阵
        # for i in range(10):
        #     for j in range(10):
        #         transitions[i][j] += self.trans_limit[i][j]
        for j in [2, 3, 5, 6, 8, 9]:
            transitions[0][j] = float('-inf')

        for j in [0, 1, 4, 5, 6, 7, 8, 9]:
            transitions[2][j] = float('-inf')
        for j in [0, 1, 2, 3, 4, 7, 8, 9]:
            transitions[5][j] = float('-inf')
        for j in [0, 1, 2, 3, 4, 5, 6, 7]:
            transitions[8][j] = float('-inf')

        for j in [5, 6, 8, 9]:
            transitions[1][j] = float('-inf')
            transitions[3][j] = float('-inf')
        for j in [2, 3, 8, 9]:
            transitions[4][j] = float('-inf')
            transitions[6][j] = float('-inf')
        for j in [2, 3, 5, 6]:
            transitions[7][j] = float('-inf')
            transitions[9][j] = float('-inf')

        # 限制首次发射概率
        for j in [2, 3, 5, 6, 8, 9]:
            emissions[0][j] = float('-inf')

        alphas = [[0 for j in range(10)] for i in range(len(words))]
        pointers = [[0 for j in range(10)] for i in range(len(words))]
        for j in range(10):
            alphas[0][j] = emissions[0][j]

        for i in range(1, len(words)):
            for j in range(10):
                score = float('-inf')
                for k in range(10):
                    s = alphas[i - 1][k] + transitions[k][j] + emissions[i][j]
                    if s > score:
                        score = s
                        pointers[i][j] = k
                alphas[i][j] = score

        guess_tags = [0 for j in range(len(words))]
        guess_tags[-1] = alphas[-1].index(max(alphas[-1]))
        for j in range(len(words)-2, -1, -1):
            guess_tags[j] = pointers[j+1][guess_tags[j+1]]

        return guess_tags

    def average(self):
        self._weights.average()

    def unaverage(self):
        self._weights.unaverage()

    def train(self, words, tags):
        assert len(words) == len(tags)

        self._weights.time += 1
        # 预测该句的标签
        guess_tags = self.viterbi_decode(words)

        # 更新权值
        if guess_tags != tags:
            self.update_weights(words, tags, guess_tags)

    def save(self, path):
        self._weights.save(path)

    def load(self, path):
        self._weights.load(path)

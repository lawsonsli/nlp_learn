# -*- coding: utf-8 -*
# __author__ = 'Li Shang'


import random
import time
from nlp.sdp.Perceptron import AveragedPerceptron
import nlp.sdp.config as config


def get_arc_types(all_sentences):
    arc_types = set()
    for sentence in all_sentences:
        assert sentence[0] == '#'
        lines = sentence.split('\n')[1:]

        for a_line in lines:
            cols = a_line.split('\t')
            arc_types |= set(cols[6:])
    return arc_types


def format_print(a_sentence, dependency, root):
    lines = a_sentence.split('\n')
    length = len(lines)
    # add root and pred
    for index in range(1, length):
        if index == root:
            lines[index] += '\t+'
        else:
            lines[index] += '\t-'
    pred = set(map(lambda d: d[0], dependency))
    for index in range(1, length):
        if index in pred:
            lines[index] += '\t+'
        else:
            lines[index] += '\t-'

    # add dependency
    for out in pred:
        for index in range(1, length):
            if (out, index) in dependency:
                lines[index] += '\t' + dependency[(out, index)]
            else:
                lines[index] += '\t_'

    return '\n'.join(lines)

if __name__ == '__main__':
    start_time = time.time()

    with open('../../data/dm.sdp', mode='r', encoding='utf-8') as file:
        text = file.read()
        train_sentences = text.strip('\n').split('\n\n')

    print(len(train_sentences))
    types = get_arc_types(train_sentences)
    config.get_all_transitions(arc_types=types)

    model = AveragedPerceptron()
    # train
    c = 0
    for i in range(1):
        random.shuffle(train_sentences)
        for a_sentence in train_sentences:
            c += 1
            print(c, a_sentence.split('\n')[0])
            model.train(a_sentence)
    model.average()
    model.save('../../data/model.pickle')
    end_time = time.time()
    print('trained, cost', end_time - start_time)

    model.load('../../data/model.pickle')
    # cross validation
    # 'sh run.sh Scorer ../data/dm.sdp_gold ../data/dm.sdp_guess representation=DM')

    # predict answer
    with open('../../data/new_emnlpii.test.sdp_deal_noAns', mode='r', encoding='utf-8') as file:
        text = file.read()
        test_sentences = text.strip('\n').split('\n\n')
    new_sentences = []
    for a_sentence in test_sentences:
        print(a_sentence.split('\n')[0])
        dp, root = model.predict(a_sentence)
        new_sentences.append(format_print(a_sentence, dp, root))
    with open('../../data/new_emnlpii.test.sdp_deal_guess', mode='w', encoding='utf-8') as file:
        file.write('\n\n'.join(new_sentences))
    print('write file done.')

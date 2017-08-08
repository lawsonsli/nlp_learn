# -*- coding: utf-8 -*
# __author__ = 'Li Shang'

from nlp.ner.Perceptron import AveragedPerceptron
from nlp.ner.evaluate import evaluate
import random


class NERtagger(object):

    def __init__(self):
        self.N_ITER = 10
        self.model = AveragedPerceptron()
        self.tags = ('O', 'B-PER', 'I-PER', 'E-PER', 'B-LOC', 'I-LOC', 'E-LOC', 'B-ORG', 'I-ORG', 'E-ORG')
        self.tags_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'E-PER': 3,
                          'B-LOC': 4, 'I-LOC': 5, 'E-LOC': 6, 'B-ORG': 7, 'I-ORG': 8, 'E-ORG': 9}

    @staticmethod
    def read_file(file_name):
        with open(file_name, mode='r', encoding='utf-8') as file:
            text = file.read()
            text = text.strip(' \n')
            return text.split('\n\n')

    def train(self, sentences):

        random.shuffle(sentences)
        for s, a_sentence in enumerate(sentences):
            words_and_tags = a_sentence.split('\n')
            words = [wt.split(' ')[0] for wt in words_and_tags]
            tags = [wt.split(' ')[1] for wt in words_and_tags]

            # 将其映射为0 1 2 3 4 5 6 7 8 9
            for i in range(len(tags)):
                if tags[i][0] == 'I':
                    if i == len(tags)-1 or tags[i+1] != tags[i]:
                        tags[i] = 'E-' + tags[i][-3:]
                tags[i] = int(self.tags_dict[tags[i]])

            self.model.train(words, tags)
            if s % 5000 == 0:
                print('       -----> ' + str(s // 5000) + '/5')

    def tag_test(self, sentences):

        # write golden file
        golden_file = open('../../data/tst_golden', mode='w', encoding='utf-8', newline='\n')
        golden_file.write('\n\n'.join(sentences) + '\n\n')
        golden_file.close()

        predict_sentences = ''
        for a_sentence in sentences:
            words_and_tags = a_sentence.split('\n')
            words = [wt.split(' ')[0] for wt in words_and_tags]
            # words = a_sentence.split('\n')
            words_num = len(words)

            labels = self.model.viterbi_decode(words)

            for i in range(words_num):
                labels[i] = self.tags[labels[i]]
                if labels[i][0] == 'E':
                    labels[i] = 'I-' + labels[i][-3:]

            predict_words = words
            for i in range(words_num):
                predict_words[i] = predict_words[i].split(' ')[0] + ' ' + labels[i]

            predict_sentences += ('\n'.join(predict_words) + '\n\n')

        predict_file = open('../../data/tst_predict', mode='w', encoding='utf-8', newline='\n')
        predict_file.write(predict_sentences)
        predict_file.close()

if __name__ == '__main__':
    tagger = NERtagger()
    all_sentences = tagger.read_file('../../data/ner_trn')

    for t in range(tagger.N_ITER):
        print('ITER ' + str(t+1) + ':')
        tagger.train(all_sentences)

    tagger.model.average()
    tagger.model.save('../../AP_weights_all_train')
    tagger.model.load('../../AP_weights_all_train')
    tagger.tag_test(tagger.read_file('../../data/ner_trn'))
    evaluate('../../data/tst_predict', '../../data/tst_golden')



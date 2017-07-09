# -*- coding: utf-8 -*
# __author__ = 'Li Shang'

import nlp.sdp.config as config


class Transition(object):

    def __init__(self, a_sentence, is_train):

        self.is_train = is_train
        self.vocab, self.dependency = self._build_dependency(a_sentence)
        self.stack1 = list()
        self.stack2 = list()
        self.buffer = list(range(1, len(self.vocab)))
        self.moves = list()

        self.last_is_arc = False

    def _build_dependency(self, a_sentence):
        assert a_sentence[0] == '#'

        vocab = list()
        dependency = dict()
        lines = a_sentence.split('\n')[1:]

        # length = len(lines) + 1
        # self.lc = ['#' for i in range(length)]
        # self.rc = ['#' for i in range(length)]
        # self.lp = ['#' for i in range(length)]
        # self.rp = ['#' for i in range(length)]
        # self.lc_l = ['#' for i in range(length)]
        # self.lc_a = [0 for i in range(length)]
        # self.rc_l = ['#' for i in range(length)]
        # self.rc_a = [0 for i in range(length)]
        # self.lp_l = ['#' for i in range(length)]
        # self.lp_a = [0 for i in range(length)]
        # self.rp_l = ['#' for i in range(length)]
        # self.rp_a = [0 for i in range(length)]

        vocab.append(('#', '#', '#'))
        if not self.is_train:
            for a_line in lines:
                cols = a_line.split('\t')
                vocab.append((cols[1], cols[2], cols[3]))
            return vocab, dependency

        dp_graph = list(map(lambda v: v.split('\t')[6:], lines))
        dp_graph = list(map(list, zip(*dp_graph)))  # transpose

        for index, a_line in enumerate(lines):
            cols = a_line.split('\t')
            vocab.append((cols[1], cols[2], cols[3]))

            if cols[4] == '+':
                dependency[(0, index + 1)] = 'root'
            if cols[5] == '+':
                for i, d in enumerate(dp_graph[0]):
                    if not d == '_':
                        dependency[(index + 1, i + 1)] = d
                dp_graph.pop(0)

        return vocab, dependency

    def _get_gold_transition(self):
        def get_arc(i, j):
            if (i, j) in self.dependency:
                return self.dependency[(i, j)]
            else:
                return None

        def node_has_arc_since(i, start=1):
            if (0, i) in self.dependency:
                return True
            if start == len(self.vocab):
                return False
            for j in range(start, len(self.vocab)):
                if (i, j) in self.dependency or (j, i) in self.dependency:
                    return True
            return False

        def stack_has_arc(s, j):
            i = len(s) - 1
            while i >= 0:
                if get_arc(s[i], j) is not None or get_arc(j, s[i]) is not None:
                    return i
                i -= 1
            return None

        if len(self.stack1) == 0 and len(self.stack2) == 0:
            return 'Shift'
        # if not self.last_is_arc:
        if len(self.stack1) > 0 and len(self.buffer) > 0:
            arc = get_arc(self.stack1[-1], self.buffer[0])
            if arc is not None:
                return 'RightArc-' + arc
            arc = get_arc(self.buffer[0], self.stack1[-1])
            if arc is not None:
                return 'LeftArc-' + arc

        if len(self.stack1) > 0:
            if not node_has_arc_since(self.stack1[-1], 0):# self.buffer[0]+1):
                return 'Pop'

        if len(self.buffer) > 0:
            index = stack_has_arc(self.stack1, self.buffer[0])
            if index is not None and index + 1 < len(self.stack1):
                return 'Mem'
            index = stack_has_arc(self.stack2, self.buffer[0])
            if index is not None: # and index < len(self.stack2):
                return 'Recall'

        if len(self.buffer) > 0:
            return 'Shift'
        if len(self.stack2) > 0:
            return 'Recall'

    def _get_allowed_transitions(self):
        if len(self.moves) == 0:
            return ['Shift']

        allowed = []
        if len(self.stack1) > 0:
            allowed += ['Pop']
            if not self.moves[-1] == 'Recall':
                allowed += ['Mem']
        if len(self.buffer) > 0:
            allowed += ['Shift']
        if len(self.stack2) > 0 and not self.moves[-1] == 'Mem':
            allowed += ['Recall']

        if len(self.stack1) > 0 and len(self.buffer) > 0:
            if not self.moves[-1].startswith('LeftArc-') and not self.moves[-1].startswith('RightArc-'):
                allowed += config.all_transitions
        return allowed

    def do_a_transition(self, a_transition):
        self.last_is_arc = False
        self.moves.append(a_transition)

        if a_transition == 'Shift':
            assert len(self.buffer) > 0
            self.stack1.append(self.buffer[0])
            self.buffer.pop(0)
        elif a_transition == 'Pop':
            assert len(self.stack1) > 0
            self.stack1.pop(-1)
        elif a_transition == 'Mem':
            assert len(self.stack1) > 0
            self.stack2.append(self.stack1[-1])
            self.stack1.pop(-1)
        elif a_transition == 'Recall':
            assert len(self.stack2) > 0
            self.stack1.append(self.stack2[-1])
            self.stack2.pop(-1)
        elif a_transition.startswith('LeftArc-'):
            self.last_is_arc = True

            # out_node = self.buffer[0]
            # in_node = self.stack1[-1]
            # self.lc_a[out_node] += 1
            # self.rp_a[in_node] += 1
            # if self.lc[out_node] == '#' or in_node < self.lc[out_node]:
            #     self.lc[out_node] = in_node
            #     self.lc_l[out_node] = a_transition
            # if self.rp[in_node] == '#' or out_node > self.rp[in_node]:
            #     self.rp[in_node] = out_node
            #     self.rp_l[in_node] = a_transition

            if not self.is_train:
                self.dependency[(self.buffer[0], self.stack1[-1])] = a_transition[8:]
            else:
                self.dependency.pop((self.buffer[0], self.stack1[-1]))
        elif a_transition.startswith('RightArc-'):
            self.last_is_arc = True

            # out_node = self.stack1[-1]
            # in_node = self.buffer[0]
            # self.rc_a[out_node] += 1
            # self.lp_a[in_node] += 1
            # if self.rc[out_node] == '#' or in_node > self.rc[out_node]:
            #     self.rc[out_node] = in_node
            #     self.rc_l[out_node] = a_transition
            # if self.lp[in_node] == '#' or out_node < self.lp[in_node]:
            #     self.lp[in_node] = out_node
            #     self.lp_l[in_node] = a_transition

            if not self.is_train:
                self.dependency[(self.stack1[-1], self.buffer[0])] = a_transition[9:]
            else:
                self.dependency.pop((self.stack1[-1], self.buffer[0]))

    def gen_features(self):
        while len(self.buffer) > 0 or len(self.stack2) > 0:
            s1_1_pos = self.vocab[self.stack1[-1]][2] if len(self.stack1) > 0 else '#'
            s1_2_pos = self.vocab[self.stack1[-2]][2] if len(self.stack1) > 1 else '#'
            s2_1_pos = self.vocab[self.stack2[-1]][2] if len(self.stack2) > 0 else '#'
            s2_2_pos = self.vocab[self.stack2[-2]][2] if len(self.stack2) > 1 else '#'
            b_1_pos = self.vocab[self.buffer[0]][2] if len(self.buffer) > 0 else '#'

            s1_1_lemma = self.vocab[self.stack1[-1]][1] if len(self.stack1) > 0 else '#'
            s1_2_lemma = self.vocab[self.stack1[-2]][1] if len(self.stack1) > 1 else '#'
            s2_1_lemma = self.vocab[self.stack2[-1]][1] if len(self.stack2) > 0 else '#'
            s2_2_lemma = self.vocab[self.stack2[-2]][1] if len(self.stack2) > 1 else '#'
            b_1_lemma = self.vocab[self.buffer[0]][1] if len(self.buffer) > 0 else '#'

            last_2_move = self.moves[-2] if len(self.moves) > 1 else '#'
            last_1_move = self.moves[-1] if len(self.moves) > 0 else '#'

            s1_1_context_n1_lemma = self.vocab[self.stack1[-1] - 1][1] \
                if len(self.stack1) > 0 and self.stack1[-1] > 1 else '#'
            s1_1_context_p1_lemma = self.vocab[self.stack1[-1] + 1][1] \
                if len(self.stack1) > 0 and self.stack1[-1] + 1 < len(self.vocab) else '#'
            s1_1_context_n2_lemma = self.vocab[self.stack1[-1] - 2][1] \
                if len(self.stack1) > 0 and self.stack1[-1] > 2 else '#'
            s1_1_context_p2_lemma = self.vocab[self.stack1[-1] + 2][1] \
                if len(self.stack1) > 0 and self.stack1[-1] + 2 < len(self.vocab) else '#'

            # s1_2_context_n1_lemma = self.vocab[self.stack1[-2] - 1][1] \
            #     if len(self.stack1) > 1 and self.stack1[-2] > 1 else '#'
            # s1_2_context_p1_lemma = self.vocab[self.stack1[-2] + 1][1] \
            #     if len(self.stack1) > 1 and self.stack1[-2] + 1 < len(self.vocab) else '#'

            s2_1_context_n1_lemma = self.vocab[self.stack2[-1] - 1][1] \
                if len(self.stack2) > 0 and self.stack2[-1] > 1 else '#'
            s2_1_context_p1_lemma = self.vocab[self.stack2[-1] + 1][1] \
                if len(self.stack2) > 0 and self.stack2[-1] + 1 < len(self.vocab) else '#'
            s2_1_context_n2_lemma = self.vocab[self.stack2[-1] - 2][1] \
                if len(self.stack2) > 0 and self.stack2[-1] > 2 else '#'
            s2_1_context_p2_lemma = self.vocab[self.stack2[-1] + 2][1] \
                if len(self.stack2) > 0 and self.stack2[-1] + 2 < len(self.vocab) else '#'

            # s2_2_context_n1_lemma = self.vocab[self.stack2[-2] - 1][1] \
            #     if len(self.stack2) > 1 and self.stack2[-2] > 1 else '#'
            # s2_2_context_p1_lemma = self.vocab[self.stack2[-2] + 1][1] \
            #     if len(self.stack2) > 1 and self.stack2[-2] + 1 < len(self.vocab) else '#'

            b_1_context_n1_lemma = self.vocab[self.buffer[0] - 1][1] \
                if len(self.buffer) > 0 and self.buffer[0] > 1 else '#'
            b_1_context_p1_lemma = self.vocab[self.buffer[0] + 1][1] \
                if len(self.buffer) > 0 and self.buffer[0] + 1 < len(self.vocab) else '#'
            b_1_context_n2_lemma = self.vocab[self.buffer[0] - 2][1] \
                if len(self.buffer) > 0 and self.buffer[0] > 2 else '#'
            b_1_context_p2_lemma = self.vocab[self.buffer[0] + 2][1] \
                if len(self.buffer) > 0 and self.buffer[0] + 2 < len(self.vocab) else '#'

            s1_1_context_n1_pos = self.vocab[self.stack1[-1] - 1][2] \
                if len(self.stack1) > 0 and self.stack1[-1] > 1 else '#'
            s1_1_context_p1_pos = self.vocab[self.stack1[-1] + 1][2] \
                if len(self.stack1) > 0 and self.stack1[-1] + 1 < len(self.vocab) else '#'
            s1_1_context_n2_pos = self.vocab[self.stack1[-1] - 2][2] \
                if len(self.stack1) > 0 and self.stack1[-1] > 2 else '#'
            s1_1_context_p2_pos = self.vocab[self.stack1[-1] + 2][2] \
                if len(self.stack1) > 0 and self.stack1[-1] + 2 < len(self.vocab) else '#'

            # s1_2_context_n1_pos = self.vocab[self.stack1[-2] - 1][2] \
            #     if len(self.stack1) > 1 and self.stack1[-2] > 1 else '#'
            # s1_2_context_p1_pos = self.vocab[self.stack1[-2] + 1][2] \
            #     if len(self.stack1) > 1 and self.stack1[-2] + 1 < len(self.vocab) else '#'

            s2_1_context_n1_pos = self.vocab[self.stack2[-1] - 1][2] \
                if len(self.stack2) > 0 and self.stack2[-1] > 1 else '#'
            s2_1_context_p1_pos = self.vocab[self.stack2[-1] + 1][2] \
                if len(self.stack2) > 0 and self.stack2[-1] + 1 < len(self.vocab) else '#'
            s2_1_context_n2_pos = self.vocab[self.stack2[-1] - 2][2] \
                if len(self.stack2) > 0 and self.stack2[-1] > 2 else '#'
            s2_1_context_p2_pos = self.vocab[self.stack2[-1] + 2][2] \
                if len(self.stack2) > 0 and self.stack2[-1] + 2 < len(self.vocab) else '#'

            # s2_2_context_n1_pos = self.vocab[self.stack2[-2] - 1][2] \
            #     if len(self.stack2) > 1 and self.stack2[-2] > 1 else '#'
            # s2_2_context_p1_pos = self.vocab[self.stack2[-2] + 1][2] \
            #     if len(self.stack2) > 1 and self.stack2[-2] + 1 < len(self.vocab) else '#'

            b_1_context_n1_pos = self.vocab[self.buffer[0] - 1][2] \
                if len(self.buffer) > 0 and self.buffer[0] > 1 else '#'
            b_1_context_p1_pos = self.vocab[self.buffer[0] + 1][2] \
                if len(self.buffer) > 0 and self.buffer[0] + 1 < len(self.vocab) else '#'
            b_1_context_n2_pos = self.vocab[self.buffer[0] - 2][2] \
                if len(self.buffer) > 0 and self.buffer[0] > 2 else '#'
            b_1_context_p2_pos = self.vocab[self.buffer[0] + 2][2] \
                if len(self.buffer) > 0 and self.buffer[0] + 2 < len(self.vocab) else '#'

            # s1_1_lc_l = self.lc_l[self.stack1[-1]] if len(self.stack1) > 0 else '#'
            # s1_1_rc_l = self.rc_l[self.stack1[-1]] if len(self.stack1) > 0 else '#'
            # s1_1_lp_l = self.lp_l[self.stack1[-1]] if len(self.stack1) > 0 else '#'
            # s1_1_rp_l = self.rp_l[self.stack1[-1]] if len(self.stack1) > 0 else '#'
            # s1_1_lc_a = self.lc_a[self.stack1[-1]] if len(self.stack1) > 0 else '#'
            # s1_1_rc_a = self.rc_a[self.stack1[-1]] if len(self.stack1) > 0 else '#'
            # s1_1_lp_a = self.lp_a[self.stack1[-1]] if len(self.stack1) > 0 else '#'
            # s1_1_rp_a = self.rp_a[self.stack1[-1]] if len(self.stack1) > 0 else '#'
            #
            # s2_1_lc_l = self.lc_l[self.stack2[-1]] if len(self.stack2) > 0 else '#'
            # s2_1_rc_l = self.rc_l[self.stack2[-1]] if len(self.stack2) > 0 else '#'
            # s2_1_lp_l = self.lp_l[self.stack2[-1]] if len(self.stack2) > 0 else '#'
            # s2_1_rp_l = self.rp_l[self.stack2[-1]] if len(self.stack2) > 0 else '#'
            # s2_1_lc_a = self.lc_a[self.stack2[-1]] if len(self.stack2) > 0 else '#'
            # s2_1_rc_a = self.rc_a[self.stack2[-1]] if len(self.stack2) > 0 else '#'
            # s2_1_lp_a = self.lp_a[self.stack2[-1]] if len(self.stack2) > 0 else '#'
            # s2_1_rp_a = self.rp_a[self.stack2[-1]] if len(self.stack2) > 0 else '#'
            #
            # b_1_lc_l = self.lc_l[self.buffer[-1]] if len(self.buffer) > 0 else '#'
            # b_1_rc_l = self.rc_l[self.buffer[-1]] if len(self.buffer) > 0 else '#'
            # b_1_lp_l = self.lp_l[self.buffer[-1]] if len(self.buffer) > 0 else '#'
            # b_1_rp_l = self.rp_l[self.buffer[-1]] if len(self.buffer) > 0 else '#'
            # b_1_lc_a = self.lc_a[self.buffer[-1]] if len(self.buffer) > 0 else '#'
            # b_1_rc_a = self.rc_a[self.buffer[-1]] if len(self.buffer) > 0 else '#'
            # b_1_lp_a = self.lp_a[self.buffer[-1]] if len(self.buffer) > 0 else '#'
            # b_1_rp_a = self.rp_a[self.buffer[-1]] if len(self.buffer) > 0 else '#'

            index = 0
            features = list()

            def _add_features(feat):
                nonlocal index
                nonlocal features
                index += 1
                features.append(str(index) + str(feat))

            # Unigram
            _add_features(s1_1_pos)
            _add_features(s1_1_lemma)
            _add_features(s1_2_pos)
            _add_features(s1_2_lemma)
            _add_features(s2_1_pos)
            _add_features(s2_1_lemma)
            _add_features(s2_2_pos)
            _add_features(s2_2_lemma)
            _add_features(b_1_pos)
            _add_features(b_1_lemma)

            # _add_features(s1_1_lc_l)
            # _add_features(s1_1_rc_l)
            # _add_features(s1_1_lp_l)
            # _add_features(s1_1_rp_l)
            # _add_features(s1_1_lc_a)
            # _add_features(s1_1_rc_a)
            # _add_features(s1_1_lp_a)
            # _add_features(s1_1_rp_a)
            #
            # _add_features(s2_1_lc_l)
            # _add_features(s2_1_rc_l)
            # _add_features(s2_1_lp_l)
            # _add_features(s2_1_rp_l)
            # _add_features(s2_1_lc_a)
            # _add_features(s2_1_rc_a)
            # _add_features(s2_1_lp_a)
            # _add_features(s2_1_rp_a)
            #
            # _add_features(b_1_lc_l)
            # _add_features(b_1_rc_l)
            # _add_features(b_1_lp_l)
            # _add_features(b_1_rp_l)
            # _add_features(b_1_lc_a)
            # _add_features(b_1_rc_a)
            # _add_features(b_1_lp_a)
            # _add_features(b_1_rp_a)

            _add_features(s1_1_context_n2_pos)
            _add_features(s1_1_context_n1_pos)
            _add_features(s1_1_context_p1_pos)
            _add_features(s1_1_context_p2_pos)
            _add_features(s1_1_context_n2_lemma)
            _add_features(s1_1_context_n1_lemma)
            _add_features(s1_1_context_p1_lemma)
            _add_features(s1_1_context_p2_lemma)

            # _add_features(s1_2_context_n1_pos)
            # _add_features(s1_2_context_p1_pos)
            # _add_features(s1_2_context_n1_lemma)
            # _add_features(s1_2_context_p1_lemma)

            _add_features(s2_1_context_n2_pos)
            _add_features(s2_1_context_n1_pos)
            _add_features(s2_1_context_p1_pos)
            _add_features(s2_1_context_p2_pos)
            _add_features(s2_1_context_n2_lemma)
            _add_features(s2_1_context_n1_lemma)
            _add_features(s2_1_context_p1_lemma)
            _add_features(s2_1_context_p2_lemma)

            # _add_features(s2_2_context_n1_pos)
            # _add_features(s2_2_context_p1_pos)
            # _add_features(s2_2_context_n1_lemma)
            # _add_features(s2_2_context_p1_lemma)

            _add_features(b_1_context_n2_pos)
            _add_features(b_1_context_n1_pos)
            _add_features(b_1_context_p1_pos)
            _add_features(b_1_context_p2_pos)
            _add_features(b_1_context_n2_lemma)
            _add_features(b_1_context_n1_lemma)
            _add_features(b_1_context_p1_lemma)
            _add_features(b_1_context_p2_lemma)

            _add_features(last_2_move)
            _add_features(last_1_move)
            # Bigram
            _add_features(s1_1_context_n2_pos + ' ' + s1_1_context_n1_pos)
            _add_features(s1_1_context_n1_pos + ' ' + s1_1_pos)
            _add_features(s1_1_pos + ' ' + s1_1_context_p1_pos)
            _add_features(s1_1_context_p1_pos + ' ' + s1_1_context_p2_pos)
            _add_features(s1_1_context_n2_lemma + ' ' + s1_1_context_n1_lemma)
            _add_features(s1_1_context_n1_lemma + ' ' + s1_1_lemma)
            _add_features(s1_1_lemma + ' ' + s1_1_context_p1_lemma)
            _add_features(s1_1_context_p1_lemma + ' ' + s1_1_context_p2_lemma)

            _add_features(s2_1_context_n2_pos + ' ' + s2_1_context_n1_pos)
            _add_features(s2_1_context_n1_pos + ' ' + s2_1_pos)
            _add_features(s2_1_pos + ' ' + s2_1_context_p1_pos)
            _add_features(s2_1_context_p1_pos + ' ' + s2_1_context_p2_pos)
            _add_features(s2_1_context_n2_lemma + ' ' + s2_1_context_n1_lemma)
            _add_features(s2_1_context_n1_lemma + ' ' + s2_1_lemma)
            _add_features(s2_1_lemma + ' ' + s2_1_context_p1_lemma)
            _add_features(s2_1_context_p1_lemma + ' ' + s2_1_context_p2_lemma)

            _add_features(b_1_context_n2_pos + ' ' + b_1_context_n1_pos)
            _add_features(b_1_context_n1_pos + ' ' + b_1_pos)
            _add_features(b_1_pos + ' ' + b_1_context_p1_pos)
            _add_features(b_1_context_p1_pos + ' ' + b_1_context_p2_pos)
            _add_features(b_1_context_n2_lemma + ' ' + b_1_context_n1_lemma)
            _add_features(b_1_context_n1_lemma + ' ' + b_1_lemma)
            _add_features(b_1_lemma + ' ' + b_1_context_p1_lemma)
            _add_features(b_1_context_p1_lemma + ' ' + b_1_context_p2_lemma)

            _add_features(s1_1_pos + ' ' + b_1_pos)
            _add_features(s1_1_lemma + ' ' + b_1_lemma)
            _add_features(s2_1_pos + ' ' + b_1_pos)
            _add_features(s2_1_lemma + ' ' + b_1_lemma)

            # Trigram
            # _add_features(s1_1_context_n1_pos + ' ' + s1_1_pos + ' ' + s1_1_context_p1_pos)
            # _add_features(s1_1_context_n1_lemma + ' ' + s1_1_lemma + ' ' + s1_1_context_p1_lemma)
            #
            # _add_features(b_1_context_n1_pos + ' ' + b_1_pos + ' ' + b_1_context_p1_pos)
            # _add_features(b_1_context_n1_lemma + ' ' + b_1_lemma + ' ' + b_1_context_p1_lemma)

            gold = '#'
            allowed = self._get_allowed_transitions()
            if self.is_train:
                gold = self._get_gold_transition()

            yield features, allowed, gold

            if self.is_train:
                self.do_a_transition(gold)


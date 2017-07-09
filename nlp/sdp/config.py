
all_transitions = []
# all_transitions_without_mem = ['Shift', 'Pop', 'Recall']
# all_transitions_without_recall = ['Shift', 'Pop', 'Mem']


def get_all_transitions(arc_types):
    for arc in arc_types:
        all_transitions.append('LeftArc-' + arc)
        all_transitions.append('RightArc-' + arc)
        # all_transitions_without_mem.append('LeftArc-' + arc)
        # all_transitions_without_mem.append('RightArc-' + arc)
        # all_transitions_without_recall.append('LeftArc-' + arc)
        # all_transitions_without_recall.append('RightArc-' + arc)


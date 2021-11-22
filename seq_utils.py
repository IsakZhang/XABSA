import numpy as np


SMALL_POSITIVE_CONST = 1e-4


def ot2bio_ate(ate_tag_sequence):
    """
    ot2bio function for ate task
    """
    n_tags = len(ate_tag_sequence)
    new_ate_sequence = []
    prev_ate_tag = '$$$'
    for i in range(n_tags):
        cur_ate_tag = ate_tag_sequence[i]
        if cur_ate_tag == 'O' or cur_ate_tag == 'EQ':
            # note that, EQ tag can be incorrectly predicted in the testing phase
            # when meet EQ, regard it as O
            new_ate_sequence.append('O')
            prev_ate_tag = 'O'
        else:
            # current ate tag is T
            if cur_ate_tag == prev_ate_tag:
                new_ate_sequence.append('I')
            else:
                new_ate_sequence.append('B')
            prev_ate_tag = cur_ate_tag
    assert len(new_ate_sequence) == len(ate_tag_sequence)
    return new_ate_sequence


def ot2bieos_ate(ate_tag_sequence):
    """
    ot2bieos function for ate task
    """
    n_tags = len(ate_tag_sequence)
    new_ate_sequence = []
    prev_ate_tag = '$$$'
    for i in range(n_tags):
        cur_ate_tag = ate_tag_sequence[i]
        if cur_ate_tag == 'O' or cur_ate_tag == 'EQ':
            # note that, EQ tag can be incorrectly predicted in the testing phase
            # when meet EQ, regard it as O
            new_ate_sequence.append('O')
            prev_ate_tag = 'O'
        else:
            # current ate tag is T
            if cur_ate_tag != prev_ate_tag:
                if i == n_tags - 1:
                    new_ate_sequence.append('S')
                else:
                    next_ate_tag = ate_tag_sequence[i + 1]
                    if next_ate_tag == 'O':
                        new_ate_sequence.append('S')
                    else:
                        new_ate_sequence.append('B')
            else:
                # previous ate tag is also T
                if i == n_tags - 1:
                    new_ate_sequence.append('E')
                else:
                    next_ate_tag = ate_tag_sequence[i + 1]
                    if next_ate_tag == 'O':
                        new_ate_sequence.append('E')
                    else:
                        new_ate_sequence.append('I')
            prev_ate_tag = 'T'
    return new_ate_sequence


def ot2bio_absa(absa_tag_sequence):
    """
    ot2bio function for ts tag sequence
    """
    #new_ts_sequence = []
    new_absa_sequence = []
    n_tag = len(absa_tag_sequence)
    prev_pos = '$$$'
    for i in range(n_tag):
        cur_absa_tag = absa_tag_sequence[i]
        if cur_absa_tag == 'O':
            new_absa_sequence.append('O')
            cur_pos = 'O'
        else:
            # current tag is subjective tag, i.e., cur_pos is T
            # print(cur_ts_tag)
            cur_pos, cur_sentiment = cur_absa_tag.split('-')
            if cur_pos == prev_pos:
                # prev_pos is T
                new_absa_sequence.append('I-%s' % cur_sentiment)
            else:
                # prev_pos is O
                new_absa_sequence.append('B-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_absa_sequence


def ot2bieos_absa(absa_tag_sequence):
    """
    ot2bieos function for end-to-end aspect-based sentiment analysis task
    """
    n_tags = len(absa_tag_sequence)
    #new_ts_sequence = []
    new_absa_sequence = []
    prev_pos = '$$$'

    for i in range(n_tags):
        cur_absa_tag = absa_tag_sequence[i]
        if cur_absa_tag == 'O' or cur_absa_tag == 'EQ':
            # when meet the EQ tag, regard it as O
            new_absa_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_absa_tag.split('-')
            # cur_pos is T
            if cur_pos != prev_pos:
                # prev_pos is O and new_cur_pos can only be B or S
                if i == n_tags - 1:
                    new_absa_sequence.append('S-%s' % cur_sentiment)
                else:
                    next_absa_tag = absa_tag_sequence[i + 1]
                    if next_absa_tag == 'O':
                        new_absa_sequence.append('S-%s' % cur_sentiment)
                    else:
                        new_absa_sequence.append('B-%s' % cur_sentiment)
            else:
                # prev_pos is T and new_cur_pos can only be I or E
                if i == n_tags - 1:
                    new_absa_sequence.append('E-%s' % cur_sentiment)
                else:
                    next_absa_tag = absa_tag_sequence[i + 1]
                    if next_absa_tag == 'O':
                        new_absa_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_absa_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_absa_sequence


def ot2bio(tags, task):
    if task == 'ate':
        return ot2bio_ate(tags)
    elif task == 'absa':
        return ot2bio_absa(tags)
    else:
        raise Exception("Unsupported task!")


def ot2bieos(tags, task):
    if task == 'ate':
        return ot2bieos_ate(tags)
    elif task == 'absa':
        return ot2bieos_absa(tags)
    else:
        raise Exception("Unsupported task!")


def bio2ot_ate(ate_tag_sequence):
    """ 
    bio2ot function for ate task 
    """
    n_tags = len(ate_tag_sequence)
    new_ate_sequence = []
    for i in range(n_tags):
        ate_tag = ate_tag_sequence[i]
        if ate_tag == 'O' or ate_tag == 'EQ':
            # note that, EQ tag can be incorrectly predicted in the testing phase
            # when meet EQ, regard it as O
            new_ate_sequence.append('O')
        else:
            new_ate_sequence.append('T')
    assert len(new_ate_sequence) == len(ate_tag_sequence)
    return new_ate_sequence


def bio2ot_absa(absa_tag_sequence):
    """ 
    bio2ot function for absa task 
    """
    new_absa_sequence = []
    n_tags = len(absa_tag_sequence)
    for i in range(n_tags):
        absa_tag = absa_tag_sequence[i]
        #assert absa_tag != 'EQ'
        if absa_tag == 'O' or absa_tag == 'EQ':
            new_absa_sequence.append('O')
        else:
            pos, sentiment = absa_tag.split('-')
            new_absa_sequence.append('T-%s' % sentiment)
    return new_absa_sequence


def tag2ate(tag_sequence):
    """
    :param tag_sequence:
    """
    n_tags = len(tag_sequence)
    ate_sequence = []
    beg, end = -1, -1
    for i in range(n_tags):
        ate_tag = tag_sequence[i]
        if ate_tag == 'S':
            ate_sequence.append((i, i))
        elif ate_tag == 'B':
            beg = i
        elif ate_tag == 'E':
            end = i
            if end > beg > -1:
                ate_sequence.append((beg, end))
            beg, end = -1, -1
    return ate_sequence


def tag2absa(tag_sequence):
    """
    transform absa tag sequence to a list of absa triplet (b, e, sentiment)
    """
    n_tags = len(tag_sequence)
    absa_sequence, sentiments = [], []
    beg, end = -1, -1
    for i in range(n_tags):
        absa_tag = tag_sequence[i]
        # current position and sentiment
        # tag O and tag EQ will not be counted
        eles = absa_tag.split('-')
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        if sentiment != 'O':
            # current word is a subjective word
            sentiments.append(sentiment)
        if pos == 'S':
            # singleton
            # assert len(sentiments) == 1
            absa_sequence.append((i, i, sentiments[-1]))
            sentiments = []
        elif pos == 'B':
            beg = i
            if len(sentiments) > 1:
                # remove the effect of the noisy I-{POS,NEG,NEU}
                sentiments = [sentiments[-1]]
        elif pos == 'E':
            end = i
            # schema1: only the consistent sentiment tags are accepted
            # that is, all of the sentiment tags are the same
            if end > beg > -1 and len(set(sentiments)) == 1:
                absa_sequence.append((beg, end, sentiment))
                sentiments = []
                beg, end = -1, -1
    return absa_sequence


def match_ate(gold_ate_sequence, pred_ate_sequence):
    """
    calculate the proportions of correctly predicted aspect terms
    """
    hit_count, gold_count, pred_count = 0, 0, 0
    gold_count = len(gold_ate_sequence)
    pred_count = len(pred_ate_sequence)
    for t in pred_ate_sequence:
        if t in gold_ate_sequence:
            hit_count += 1
    return hit_count, gold_count, pred_count


def match_absa(gold_absa_sequence, pred_absa_sequence):
    """
    calculate the number of correctly predicted aspect sentiment
    :param gold_absa_sequence: gold standard targeted sentiment sequence
    :param pred_absa_sequence: predicted targeted sentiment sequence
    """
    # positive, negative and neutral
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_absa_sequence:
        #print(t)
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_absa_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_absa_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count


def compute_metrics_absa(pred, gold, label_vocab, tagging_schema):
    """
    compute metric scores for absa task
    """
    assert len(pred) == len(gold)
    num_samples = len(pred)

    # number of true positive, gold standard, predicted aspect sentiment triplet
    n_tp_absa, n_gold_absa, n_pred_absa = np.zeros(3), np.zeros(3), np.zeros(3)
    class_count = np.zeros(3)
    class_precision, class_recall, class_f1 = np.zeros(3), np.zeros(3), np.zeros(3)
    absa_ground_truth, absa_predictions = [], []

    for i in range(num_samples):
        eval_positions = np.where(gold[i] != -100)[0]
        #print("eval_positions:", eval_positions)
        pred_labels = pred[i][eval_positions]
        gold_labels = gold[i][eval_positions]
        pred_tags = [label_vocab[label] for label in pred_labels]
        gold_tags = [label_vocab[label] for label in gold_labels]

        if tagging_schema == 'OT':
            gold_tags = ot2bieos_absa(gold_tags)
            pred_tags = ot2bieos_absa(pred_tags)
        elif tagging_schema == 'BIO':
            gold_tags = ot2bieos_absa(bio2ot_absa(gold_tags))
            pred_tags = ot2bieos_absa(bio2ot_absa(pred_tags))
        else:
            pass # current tagging schema is BIEOS, do nothing
        pred_absa_seq, gold_absa_seq = tag2absa(tag_sequence=pred_tags), tag2absa(tag_sequence=gold_tags)
        hit_count, gold_count, pred_count = match_absa(gold_absa_sequence=gold_absa_seq, pred_absa_sequence=pred_absa_seq)
        absa_ground_truth.append(gold_absa_seq)
        absa_predictions.append(pred_absa_seq)
        # true-positive count
        n_tp_absa += hit_count
        n_gold_absa += gold_count
        n_pred_absa += pred_count

        for (b, e, s) in gold_absa_seq:
            if s == 'POS':
                class_count[0] += 1
            elif s == 'NEG':
                class_count[1] += 1
            else:
                class_count[2] += 1
    print("#POS: {}, #NEG: {}, #NEU: {}".format(class_count[0], class_count[1], class_count[2]))
    for i in range(3):
        num_hit = n_tp_absa[i]
        num_gold = n_gold_absa[i]
        num_pred = n_pred_absa[i]
        class_precision[i] = float(num_hit) / float(num_pred + SMALL_POSITIVE_CONST)
        class_recall[i] = float(num_hit) / float(num_gold + SMALL_POSITIVE_CONST)
        class_f1[i] = 2 * class_precision[i] * class_recall[i] / (class_precision[i] + class_recall[i] + SMALL_POSITIVE_CONST)
    #print("class count:", class_count)
    macro_f1 = class_f1.mean()

    num_hit_total = sum(n_tp_absa)
    num_pred_total = sum(n_pred_absa)
    num_gold_total = sum(n_gold_absa)
    micro_precision = float(num_hit_total) / (num_pred_total + SMALL_POSITIVE_CONST)
    micro_recall = float(num_hit_total) / (num_gold_total + SMALL_POSITIVE_CONST)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + SMALL_POSITIVE_CONST)
    scores = {'macro_f1': macro_f1, 'precision': micro_precision,
              "recall": micro_recall, "micro_f1": micro_f1}
    
    return scores, absa_ground_truth, absa_predictions


def compute_metrics_ate(pred, gold, label_vocab, tagging_schema):
    """
    compute metric scores for ate task
    """
    assert len(pred) == len(gold) 
    num_samples = len(pred)
    
    # number of true postive, gold standard, predicted aspect terms
    n_tp_ate, n_gold_ate, n_pred_ate = 0, 0, 0
    ate_ground_truth, ate_predictions = [], []
    
    for i in range(num_samples):
        # eval_positions = np.where(eval_mask[i] == 1)[0]
        eval_positions = np.where(gold[i] != -100)[0]
        pred_labels = pred[i][eval_positions]
        gold_labels = gold[i][eval_positions]
        pred_tags = [label_vocab[label] for label in pred_labels]
        gold_tags = [label_vocab[label] for label in gold_labels]
        if tagging_schema == 'OT':
            gold_tags = ot2bieos_ate(gold_tags)
            pred_tags = ot2bieos_ate(pred_tags)
        elif tagging_schema == 'BIO':
            gold_tags = ot2bieos_ate(bio2ot_ate(gold_tags))
            pred_tags = ot2bieos_ate(bio2ot_ate(pred_tags))
        else:
            pass # current tagging schema is BIEOS, do nothing
        # golden & predicted aspect terms
        gold_ate_seq, pred_ate_seq = tag2ate(tag_sequence=gold_tags), tag2ate(tag_sequence=pred_tags)
        hit_count, gold_count, pred_count = match_ate(gold_ate_sequence=gold_ate_seq, pred_ate_sequence=pred_ate_seq)
        ate_ground_truth.append(gold_ate_seq)
        ate_predictions.append(pred_ate_seq)
        n_tp_ate += hit_count
        n_gold_ate += gold_count
        n_pred_ate += pred_count
    print("number of gold aspect terms: {}".format(n_gold_ate))
    precision = float(n_tp_ate) / (float(n_pred_ate) + SMALL_POSITIVE_CONST)
    recall = float(n_tp_ate) / (float(n_gold_ate) + SMALL_POSITIVE_CONST)
    f1 = 2 * precision * recall / (precision + recall + SMALL_POSITIVE_CONST)
    scores = {'precision': precision, 'recall': recall, 'f1': f1}
    return scores, ate_ground_truth, ate_predictions


def write_sents_labels(word_seqs, label_seqs, file_name):
    lines = []
    for i in range(len(word_seqs)):
        for j in range(len(word_seqs[i])):
            if word_seqs[i][j] != '\u200b':
                lines.append(f"{word_seqs[i][j]}\t{label_seqs[i][j]}\tNONE-CATE\n")
        lines.append('\n')

    with open(file_name, 'w+', encoding='UTF-8') as fp:
        fp.writelines(lines)

    print(f"Write file to {file_name}")

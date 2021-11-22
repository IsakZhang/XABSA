# This script handles the data reading under different settings

import logging
import os
import random
import time
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, XLMRobertaTokenizerFast

from seq_utils import ot2bio, ot2bieos

random.seed(42)
logger = logging.getLogger(__name__)


def get_tag_list(task, tagging_schema):
    """ Set up the tag list """
    task = task.lower()
    tagging_schema = tagging_schema.lower()
    if task == 'absa':
        if tagging_schema == 'ot':
            return ['O', 'T-POS', 'T-NEG', 'T-NEU']
        elif tagging_schema == 'bio':
            return ['O', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'bieos':
            return ['O', 'B-POS', 'I-POS', 'E-POS', 'S-POS',
                    'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
                    'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
        else:
            raise Exception("Invalid tagging schema: {}".format(tagging_schema))
    elif task == 'ate':
        if tagging_schema == 'ot':
            return ['O', 'T']
        elif tagging_schema == 'bio':
            return ['O', 'B', 'I']
        elif tagging_schema == 'bieos':
            return ['O', 'B', 'I', 'E', 'S']
        else:
            raise Exception("Invalid tagging schema: {}".format(tagging_schema))
    else:
        raise Exception("Invalid task name: {}".format(task))


def get_tag_vocab(task, tagging_schema, label_path):
    """ Get the tab vocab """
    if label_path == '':
        tag_list = get_tag_list(task, tagging_schema)
    # if the label list is provided as a text file
    else:
        tag_list = []
        with open('label_path') as f:
            for line in f:
                if line.strip():
                    tag_list.append(line.strip())
    tag2idx = {tag: idx for idx, tag in enumerate(tag_list)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return tag_list, tag2idx, idx2tag


def read_examples_from_file(file_path, task, tagging_schema, ratio=1.0):
    """
    Read data examples from file, also preprocess the tag seq
    Return List[List[word]], List[List[tag]]
    """
    sents, labels = [], []
    logger.info(f"Read data from file {file_path}")
    with open(file_path, 'r', encoding='UTF-8') as fp:
        words, tags = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                word, tag, _ = line.split('\t')
                words.append(word)
                tags.append(tag)
            else:
                if task.lower() == 'ate':
                    tags = [tag[0] for tag in tags]
                # convert the tagging scheme
                if tagging_schema.lower() == 'bieos':
                    tags = ot2bieos(tags, task)
                elif tagging_schema.lower() == 'bio':
                    tags = ot2bio(tags, task)
                else:
                    pass  # original tags follow the OT tagging schema, do nothing
                sents.append(words)
                labels.append(tags)
                words, tags = [], []  # clear the buffer

    print(f"Total examples = {len(sents)}")
    return sents, labels


def read_examples_from_multiple_file(file_path_list, task, tagging_schema, exp_type, ratio):
    """
    Read and mix examples from multiple files
    """
    all_sents, all_labels = [], []
    for file_path in file_path_list:
        lang = file_path.split('-')[1]
        sents, labels = read_examples_from_file(file_path, task, tagging_schema)
        all_sents.extend(sents)
        all_labels.extend(labels)
    assert len(all_sents) == len(all_labels)
    
    print(f"** Total examples (involving multiple files) = {len(all_sents)}")
    return list(all_sents), list(all_labels)


def fix_space_issue(encodings, tokenizer):
    """
    Fix space issue by XLM-R tokenizer which will tokenize some punc/word 
    to two tokens with the same offset_mapping: (0, 1)
    """
    input_ids = encodings.input_ids
    offset_mappings = encodings.offset_mapping

    num_sents = len(input_ids)
    for i in range(num_sents):
        num_tokens = len(input_ids[i])
        for j in range(num_tokens):
            # we change the offset mapping for those empty space from (0, 1) to (0, 0)
            # those space token will be projected to -100 in the encoded label seq
            # 6 is the idx for the empty space in XLMRobertaTokenizer
            if input_ids[i][j] == 6 and offset_mappings[i][j] == (0, 1):
                offset_mappings[i][j] = (0, 0)
            # there are also cases where tokens should be reserved are assigned 
            # with offset_mapping (0, 0), we fix those cases to normal ones
            if offset_mappings[i][j] == (0, 0) and input_ids[i][j] not in [0, 1, 2, 6]:
                print(tokenizer.convert_ids_to_tokens(input_ids[i][j]), "(will be kept)")
                offset_mappings[i][j] = (0, 1)

    encodings.offset_mapping = offset_mappings
    return encodings


def encode_tags(tags, tag2id, offset_mapping):
    """
    Convert tag seq to ids and add "-100" for ignored tokens
    """
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    i = 0
    for doc_labels, doc_offset in zip(labels, offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        mask = (arr_offset[:,0] == 0) & (arr_offset[:, 1] != 0)
        # if the label list is too long, we have to truncate the last few postions
        # since the max len for the model is only 512 (mbert)
        if len(doc_labels) > 300:
            # print("** Here is a super long sentence!")
            num_positive_positions = sum([1 for pos in mask if pos])
            doc_enc_labels[mask] = doc_labels[:num_positive_positions]
        else:
            doc_enc_labels[mask] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
        i += 1

    return encoded_labels


class XABSADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class XABSAKDDataset(Dataset):
    """ Dataset for distillation, only have soft labels """
    def __init__(self, encodings, teacher_probs):
        self.encodings = encodings
        self.teacher_probs = teacher_probs

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['teacher_probs'] = self.teacher_probs[idx]
        return item

    def __len__(self):
        return len(self.teacher_probs)


def build_or_load_dataset(args, tokenizer, mode='train'):
    """
    Load the corresponding dataset for train/eval
    This is the top wrapper for different tasks, data splits, or exp_type
    """
    # set the tag vocab
    _, tag2idx, _ = get_tag_vocab(task=args.task, tagging_schema=args.tagging_schema, 
                                  label_path=args.label_path)

    # get the correponding data file path according to the exp_type and mode
    exp_type = args.exp_type

    if mode == 'dev':
        # "gold-en-dev.tex" as DEV
        file_name_or_list = f'gold-{args.src_lang}-dev.txt'

    elif mode == 'test':
        # "gold-fr-test" as TEST (to be consistent with previous works)
        file_name_or_list = f"gold-{args.tgt_lang}-test.txt"

    elif mode == 'unlabeled':
        # use unlabeled data (without considering the labels)
        file_name_or_list = f"gold-{args.tgt_lang}-train.txt"

    elif mode == 'unlabeled_mtl':
        file_name_or_list = [f"gold-{l}-train.txt" for l in ['fr', 'es', 'nl', 'ru']]

    elif mode == 'train':
        # Supervised setting (to have a "upperbound")
        # e.g., 'gold-en-train.txt'
        if exp_type == 'supervised':
            assert args.src_lang == args.tgt_lang, "Src and Tgt langs should be the same under supervised setting!"
            file_name_or_list = f"gold-{args.src_lang}-train.txt"

        # Translate-train setting
        # need to have the transalted data such as 'smt-fr-train.txt'
        elif exp_type == 'smt':
            file_name_or_list = f"{exp_type}-{args.tgt_lang}-train.txt"

        elif exp_type.startswith('mtl'):
            file_name_or_list = ['gold-en-train.txt']
            file_name_or_list += [f'smt-{l}-train.txt' for l in ['fr', 'es', 'nl', 'ru']]

        # Proposed ACS method
        # need to have the code switching data such as 'cs-en-fr-train.txt'
        elif exp_type == 'acs':
            file_name_or_list = [f"gold-{args.src_lang}-train.txt", 
                                 f"cs_{args.src_lang}-{args.tgt_lang}-train.txt",
                                 f"cs_{args.tgt_lang}-{args.src_lang}-train.txt",
                                 f"smt-{args.tgt_lang}-train.txt"]           

        elif exp_type == 'acs_mtl':
            lang_list = ['fr', 'es', 'nl', 'ru']
            file_name_or_list = ["gold-en-train.txt"]
            file_name_or_list += [f'smt-{l}-train.txt' for l in lang_list]
            file_name_or_list += [f'cs_en-{l}-train.txt' for l in lang_list]
            file_name_or_list += [f'cs_{l}-en-train.txt' for l in lang_list]

    else:
        raise Exception("Invalid mode `{mode}`, should be one of the [train, dev, test]")

    logger.info(f"We will read file from {file_name_or_list} as {mode.upper()} data")

    # Load data features from cache or build dataset from scratch
    top_data_dir = f"{args.data_dir}/rest"
    if isinstance(file_name_or_list, str):
        file_path = f"{top_data_dir}/{file_name_or_list}"
        cached_features_file = "{0}/cached-{1}-{2}-{3}-{4}".format(
            top_data_dir, args.task, args.tfm_type, exp_type, file_name_or_list[:-4])

    elif isinstance(file_name_or_list, list):
        file_path = [f"{top_data_dir}/{f}" for f in file_name_or_list]
        included_sets = 'train_train' if exp_type.startswith('bilingual') else 'xxx'
        cached_features_file = "{0}/cached-{1}-{2}-{3}-mixed-{4}-{5}".format(
            top_data_dir, args.task, args.tfm_type, exp_type, args.tgt_lang, included_sets)

    if os.path.exists(cached_features_file) and not args.ignore_cached_data:
        logger.info(f"Find cached_features_file: {cached_features_file}")
        encodings, encoded_labels = torch.load(cached_features_file)
    else:
        logger.info(f"Didn't find / Ignore cached_features_file: {cached_features_file}, create and save...")
        # read in the text and tags
        if isinstance(file_name_or_list, str):
            texts, tags = read_examples_from_file(file_path, args.task, args.tagging_schema, ratio=args.data_select)
        elif isinstance(file_name_or_list, list):
            texts, tags = read_examples_from_multiple_file(file_path, args.task, args.tagging_schema, exp_type, ratio=0.5)

        # need to use XXFast to obtain the offset info
        assert isinstance(tokenizer, BertTokenizerFast) or isinstance(tokenizer, XLMRobertaTokenizerFast) 
        encodings = tokenizer(texts, is_split_into_words=True, 
                              return_offsets_mapping=True, padding=True, truncation=True)

        # fix the issues for xlmr tokenizer
        if args.tfm_type == 'xlmr':
            encodings = fix_space_issue(encodings, tokenizer)
        encoded_labels = encode_tags(tags, tag2idx, encodings.offset_mapping)
        encodings.pop("offset_mapping") # we don't want to pass this to the model

        if args.local_rank in [-1, 0]:
            logger.info(f"Saving features into cached file {cached_features_file}")
            # torch.save((encodings, encoded_labels), cached_features_file)

    dataset = XABSADataset(encodings, encoded_labels)
    return dataset


def write_results_to_log(log_file_path, best_test_result, args, dev_results, 
                         test_results, global_steps):
    """
    Record dev and test results to log file
    """
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = "Exp setting: {0} for {1} ({7}) | {6:.4f} | {2} -> {3} in {4} setting.\nModel is saved in {5}".format(
        args.tfm_type, 'XABSA', args.src_lang, args.tgt_lang, args.exp_type, 
        args.saved_model_dir, best_test_result, args.tagging_schema
    )
    train_settings = "Train setting: bs={0}, lr={1}, total_steps={2} (Start eval from {3})".format(
        args.per_gpu_train_batch_size, args.learning_rate, args.max_steps, args.train_begin_saving_step
    )
    results_str = "\n* Results *:  Dev  /  Test  \n"
    metric_names = ['micro_f1', 'precision', 'recall', 'eval_loss']
    for gstep in global_steps:
        results_str += f"Step-{gstep}:\n"
        for name in metric_names:
            name_step = f'{name}_{gstep}'
            results_str += f"{name:<8}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}"
            results_str += ' '*5
        results_str += '\n'

    log_str = f"{local_time}\n{exp_settings}\n{train_settings}\n{results_str}\n\n"
    with open('log.txt', "a+") as f:
        f.write(log_str)
    with open(log_file_path, "a+") as f:
        f.write(log_str)

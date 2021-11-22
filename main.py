import argparse
import os
import logging
import random
import pickle
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from transformers import BertConfig, BertTokenizer, BertTokenizerFast
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

from model import BertABSATagger, XLMRABSATagger
from seq_utils import compute_metrics_absa
from data_utils import XABSAKDDataset
from data_utils import build_or_load_dataset, get_tag_vocab, write_results_to_log 


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizerFast),
    'mbert': (BertConfig, BertABSATagger, BertTokenizerFast),
    'xlmr': (XLMRobertaConfig, XLMRABSATagger, XLMRobertaTokenizerFast)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--tfm_type", default='mbert', type=str, required=True,
                        help="The base transformer, selected from: [bert, mbert, xlmr]")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--exp_type", default='supervised', type=str, required=True,
                        help="Experiment type, selected from: [supervised, zero_shot, ...]")

    # source/target data and languages
    parser.add_argument("--data_dir", default='./data/', type=str, required=True, help="Base data dir")
    parser.add_argument("--src_lang", default='en', type=str, required=True, help="source language")
    parser.add_argument("--tgt_lang", default='fr', type=str, required=True, help="target language")
    parser.add_argument("--data_select", default=1.0, type=float, help="ratio of the selected data to train, 1 is to use all data")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS')
    parser.add_argument('--label_path', type=str, default='')
    parser.add_argument("--ignore_cached_data", action='store_true')
    parser.add_argument("--train_data_sampler", type=str, default='random')
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_distill", action='store_true', help="Whether to run knowledge distillation.")
    parser.add_argument("--trained_teacher_paths", type='str', help="path of the trained model")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")
    parser.add_argument("--eval_begin_end", default="15-19", type=str)

    # train configs
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--freeze_bottom_layer", default=-1, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--train_begin_saving_step", type=int, default=10000, help="Starting point of evaluation.")
    parser.add_argument("--train_begin_saving_epoch", type=int, default=10, help="Starting point of evaluation.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()

    # set up output dir: './outputs/mbert-en-fr-zero_shot/'
    output_dir = f"outputs/{args.tfm_type}-{args.src_lang}-{args.tgt_lang}-{args.exp_type}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_optimizer_grouped_parameters(args, model, no_grad=None):
    no_decay = ["bias", "LayerNorm.weight"]
    if no_grad is not None:
        logger.info(" The frozen parameters are:")
        for n, p in model.named_parameters():
            p.requires_grad = False if any(nd in n for nd in no_grad) else True
            if not p.requires_grad:
                logger.info("   Freeze: %s", n)
        logger.info(" The parameters to be fine-tuned are:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info("   Fine-tune: %s", n)
    else:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert False, "parameters to update with requires_grad=False"

    outputs = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0}
    ]
    return outputs


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    if args.local_rank in [-1, 0]:
       tb_writer = SummaryWriter()

    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.train_data_sampler == 'random':
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after X steps 
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")

        # save a checkpoint when each epoch ends after a specific epoch
        '''
        n_epoch_name = n_epoch
        if n_epoch_name >= args.train_begin_saving_epoch: 
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(n_epoch_name))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
        '''

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, train_loss / global_step


def train_kd(args, train_dataset, model, tokenizer):
    """ Train the model with the soft labels """
    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training with soft labels *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'teacher_probs':  batch['teacher_probs'].to(args.device)}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after 1000 steps 
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")

    return global_step, train_loss / global_step


def evaluate(args, eval_dataset, model, idx2tag, mode, step=None):
    """
    Perform evaluation on a given `eval_datset` 
    """
    eval_output_dir = args.output_dir
    # eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, mode=mode)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss, num_eval_steps = 0.0, 0
    preds, pred_labels, gold_labels = None, None, None
    results = {}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            outputs = model(**inputs)
            # logits: (bsz, seq_len, label_size)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        num_eval_steps += 1

        if preds is None:
            preds = logits.cpu().numpy()
            gold_labels = inputs['labels'].cpu().numpy()
        else:
            preds = np.append(preds, logits.cpu().numpy(), axis=0)
            gold_labels = np.append(gold_labels, inputs['labels'].cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / num_eval_steps
    # argmax operation over the last dimension
    pred_labels = np.argmax(preds, axis=-1)

    result, ground_truth, predictions = compute_metrics_absa(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    
    result['eval_loss'] = eval_loss
    results.update(result)

    if mode == 'test':
        file_to_write= {'results': results, 'labels': ground_truth, 'preds': predictions}
        file_name_to_write = f'{args.saved_model_dir}/{args.src_lang}-{args.tgt_lang}-preds-{step}.pickle'
        pickle.dump(file_to_write, open(file_name_to_write, 'wb'))
        logger.info(f"Write predictions to {file_name_to_write}")
    """
    output_eval_file = os.path.join(eval_output_dir, "%s_results.txt" % mode)
    with open(output_eval_file, "w") as writer:
        #logger.info("***** %s results *****" % mode)
        for key in sorted(result.keys()):
            if 'eval_loss' in key:
                logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        #logger.info("***** %s results *****" % mode)
    """
    return results


def get_teacher_probs(args, dataset, model_class, teacher_model_path):
    teacher_model = model_class.from_pretrained(teacher_model_path)
    teacher_model.to(args.device)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # compute logits for the dataset using the model!
    logger.info(f"***** Compute logits for [{args.tgt_lang}] using the model {teacher_model_path} *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = None
    teacher_model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         None}
            outputs = teacher_model(**inputs)
            logits = outputs[0]

        # nb_eval_steps += 1
        preds = logits.detach() if preds is None else torch.cat((preds, logits.detach()), dim=0) # dataset_len x max_seq_len x label_len

    preds = torch.nn.functional.softmax(preds, dim=-1)
    
    return preds


def get_multi_teacher_probs(args, dataset, model_class):
    teacher_paths = args.trained_teacher_paths 
    
    # obtain all preds
    all_preds = []
    for teacher_path in teacher_paths:
        preds = get_teacher_probs(args, dataset, model_class, teacher_path)
        all_preds.append(preds)
    
    logger.info("Fuse the soft labels from three pre-trained models")
    combined_preds = 1/3 * all_preds[0] + 1/3 * all_preds[1] + 1/3 * all_preds[2]

    return combined_preds

    
def main():
    # --------------------------------
    # Prepare the tags, env etc.
    args = init_args()
    print("\n", "="*30, f"NEW EXP ({args.src_lang} -> {args.tgt_lang} for {args.exp_type}", "="*30, "\n")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ['MASTER_ADDR'] = args.MASTER_ADDR
        os.environ['MASTER_PORT'] = args.MASTER_PORT
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=1)
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # not using 16-bits training
    logger.info(f"Process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}") 
    logger.info(f"Distributed training: {bool(args.local_rank != -1)}, 16-bits training: False")

    # Set seed
    set_seed(args)

    # Set up task and the label
    tag_list, tag2idx, idx2tag = get_tag_vocab('absa', args.tagging_schema, args.label_path)
    num_tags = len(tag_list)
    args.num_labels = num_tags
    logger.info(f"Perform XABSA task with label list being {tag_list} (n_labels={num_tags})")

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # hard-code lr and bs based on parameter search
    if args.tfm_type == 'mbert':
        lr, batch_size = 5e-5, 16  
    elif args.tfm_type == 'xlmr':
        lr, batch_size = 4e-5, 25
    # args.learning_rate = lr
    # args.per_gpu_train_batch_size = batch_size
    logger.info(f"We hard-coded set lr={args.learning_rate} and bs={args.per_gpu_train_batch_size}")
    

    # -----------------------------------------------------------------
    # Training process (train a model using the data and save the model)
    if args.do_train:
        logger.info("\n\n***** Prepare to conduct training  *****\n")

        # Set up model (from pre-trained tfms)
        args.tfm_type = args.tfm_type.lower() 
        logger.info(f"Load pre-trained {args.tfm_type} model from `{args.model_name_or_path}`")

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
        )
        # logger.info(f"config info: \n {config}")
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )
        # config.absa_type = args.absa_type  # type of task-specific layer, 'linear'

        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.to(args.device)

        # Distributed and parallel training
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # load the training dataset
        print()
        logger.info("Prepare training examples...")
        train_dataset = build_or_load_dataset(args, tokenizer, mode='train')
        
        # begin training!
        _, _ = train(args, train_dataset, model, tokenizer)
        
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.mkdir(args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin')) 


    if args.do_distill:
        print()
        logger.info("********** Training with KD **********")
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        if args.exp_type == 'macs_kd':
            dataset = build_or_load_dataset(args, tokenizer, mode='unlabeled_mtl')
        else:
            dataset = build_or_load_dataset(args, tokenizer, mode='unlabeled')
        
        # teacher_model_path = get_teacher_model_path(args)
        if args.exp_type == 'acs_kd_s':
            teacher_model_path = args.trained_teacher_paths
            teacher_probs = get_teacher_probs(args, dataset, model_class, teacher_model_path)
        # use the three combinations as multi-teacher distill
        elif args.exp_type == 'acs_kd_m':
            teacher_probs = get_multi_teacher_probs(args, dataset, model_class)
        # MTL version of distillation
        elif args.exp_type == 'macs_kd':
            teacher_model_path = args.trained_teacher_paths
            teacher_probs = get_teacher_probs(args, dataset, model_class, teacher_model_path)
        
        train_dataset = XABSAKDDataset(dataset.encodings, teacher_probs)
        logger.info(f"Obtained the soft labels")

        # initialize the model with the translated data
        # student_model_path = f"outputs/{args.tfm_type}-{args.src_lang}-{args.tgt_lang}-smt/checkpoint"
        student_model_path = args.student_model_path

        s_config_class, s_model_class, _ = MODEL_CLASSES[args.tfm_type]
        config = s_config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
        )
        #student_model_path = args.model_name_or_path   # not good
        
        student_model = s_model_class.from_pretrained(student_model_path)
        student_model.to(args.device)
        logger.info(f"We initialize the student model with {student_model_path}")
        
        _, _ = train_kd(args, train_dataset, student_model, tokenizer)


    # -----------------------------------------------------------------
    # Evaluation process (whether it is supervised setting or zero-shot)
    if args.do_eval:
        exp_type = args.exp_type
        logger.info("\n\n***** Prepare to conduct evaluation *****\n")
        logger.info(f"We are evaluating for *{args.tgt_lang}* under *{args.exp_type}* setting...")
        
        dev_results, test_results = {}, {}
        best_f1, best_checkpoint, best_global_step = -999999.0, None, None
        all_checkpoints, global_steps = [], []

        # find the dir containing trained model, different dirs under different settings
        # if the model is multilingual, we will only use one target language for the output dir
        if 'mtl' in exp_type:
            one_tgt_lang = 'fr'
            saved_model_dir = f"outputs/{args.tfm_type}-{args.src_lang}-{one_tgt_lang}-{exp_type}"
        
        elif exp_type == 'zero_shot':
            saved_model_dir = f"outputs/{args.tfm_type}-{args.src_lang}-{args.src_lang}-supervised"
            if not os.path.exists(saved_model_dir):
                raise Exception("No trained models can be found!")
        
        else:
            saved_model_dir = args.output_dir
        
        args.saved_model_dir = saved_model_dir
        # print(args.saved_model_dir)

        # retrieve all the saved checkpoints for model selection
        for f in os.listdir(saved_model_dir):
            sub_dir = os.path.join(saved_model_dir, f)
            if os.path.isdir(sub_dir):
                all_checkpoints.append(sub_dir)
        logger.info(f"We will perform validation on the following checkpoints: {all_checkpoints}")
        
        # load the dev and test dataset
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        config = config_class.from_pretrained(all_checkpoints[0])
        tokenizer = tokenizer_class.from_pretrained(all_checkpoints[0])
        logger.info("Load DEV dataset...")
        dev_dataset = build_or_load_dataset(args, tokenizer, mode='dev')
        logger.info("Load TEST dataset...")
        test_dataset = build_or_load_dataset(args, tokenizer, mode='test')

        for checkpoint in all_checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoint) > 1 else ""
            # only perform evaluation at the specific epochs
            eval_begin, eval_end = args.eval_begin_end.split('-')
            if int(eval_begin) <= int(global_step) < int(eval_end):
                global_steps.append(global_step)

                # reload the model and conduct inference
                logger.info(f"\nLoad the trained model from {checkpoint}...")
                model = model_class.from_pretrained(checkpoint, config=config)
                model.to(args.device)

                dev_result = evaluate(args, dev_dataset, model, idx2tag, mode='dev')
                # regard the micro-f1 as the criteria of model selection
                metrics = 'micro_f1'
                if dev_result[metrics] > best_f1:
                    best_f1 = dev_result[metrics]
                    best_checkpoint = checkpoint
                    best_global_step = global_step

                # add the global step to the name of these metrics for recording
                # 'micro_f1' --> 'micro_f1_1000'
                dev_result = dict((k + '_{}'.format(global_step), v) for k, v in dev_result.items())
                dev_results.update(dev_result)

                test_result = evaluate(args, test_dataset, model, idx2tag, mode='test', step=global_step)
                test_result = dict((k + '_{}'.format(global_step), v) for k, v in test_result.items())
                test_results.update(test_result)
    
        # print test results over last few steps
        logger.info(f"\n\nThe best checkpoint is {best_checkpoint}")
        best_step_metric = f"{metrics}_{best_global_step}"
        print(f"F1 scores on test set: {test_results[best_step_metric]:.4f}")

        print("\n* Results *:  Dev  /  Test  \n")
        metric_names = ['micro_f1', 'precision', 'recall', 'eval_loss']
        for gstep in global_steps:
            print(f"Step-{gstep}:")
            for name in metric_names:
                name_step = f'{name}_{gstep}'
                print(f"{name:<10}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}", sep='  ')
            print()

        results_log_dir = './results_log'
        if not os.path.exists(results_log_dir):
            os.mkdir(results_log_dir)
        log_file_path = f"{results_log_dir}/{args.tfm_type}-{args.exp_type}-{args.tgt_lang}.txt"
        write_results_to_log(log_file_path, test_results[best_step_metric], args, dev_results, test_results, global_steps)


if __name__ == '__main__':
    main()

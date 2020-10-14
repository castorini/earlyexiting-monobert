# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import gc
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer

from transformers.modeling_highway_bert import BertForSequenceClassification

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}

def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_collection_dir", required=False,
                        type=str, help="The evaluation collection dir (with many partitions).")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--plot_data_dir", default="./plotting/", type=str, required=False,
                        help="The directory to store data for plotting figures.")
    parser.add_argument("--evaluation_dir", default="./evaluation/", type=str, required=False,
                        help="The directory to store score files for evaluation.")
    parser.add_argument("--todo_partition_list", default="", type=str, required=False,
                        help="Done partition's num, separated by comma.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--quick_eval", action='store_true',
                        help="Set this flag to use accelerated evaluation (no early exiting).")
    parser.add_argument("--eval_highway", action='store_true',
                        help="Set this flag if it's evaluating highway models")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--pc", default=1.0, type=float,
                        help="Positive confidence threshold for early exit.")
    parser.add_argument("--nc", default=1.0, type=float,
                        help="Negative confidence threshold for early exit.")
    parser.add_argument("--limit_layer", default="-1", type=str, required=False,
                        help="The layer for limit training.")
    parser.add_argument("--train_routine",
                        choices=['raw', 'two_stage', 'all'],
                        default='raw', type=str,
                        help="Training routine (a routine can have mutliple stages, each with different strategies.")

    parser.add_argument("--output_score_file", action='store_true',
                        help="Produce score file for downstream evaluation.")
    parser.add_argument("--testset", action='store_true',
                        help="Output results on the test set")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
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

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--log_id", type=str, required=True,
                        help="log file id")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if not args.do_train:
        p_list = []
        for part in args.todo_partition_list.split(','):
            if '-' in part:
                head, tail = part.split('-')
                p_list.extend(list(range(int(head), int(tail)+1)))
            else:
                p_list.append(int(part))
        args.todo_partition_list = p_list
        print(args.todo_partition_list)

    return args
    

args = get_args()

logging.basicConfig(filename="logs/{}.log".format(args.log_id),
                    filemode='w',
                    level=0)
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_wanted_result(result):
    if "spearmanr" in result:
        print_result = result["spearmanr"]
    elif "f1" in result:
        print_result = result["f1"]
    elif "mcc" in result:
        print_result = result["mcc"]
    elif "acc" in result:
        print_result = result["acc"]
    else:
        print(result)
        exit(1)
    return print_result


def train(args, train_dataset, model, tokenizer, train_strategy='raw'):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    print(args.per_gpu_train_batch_size, args.n_gpu, args.train_batch_size)
    # train_sampler = RandomSampler(train_dataset)\
    #     if args.local_rank == -1\
    #     else DistributedSampler(train_dataset, shuffle=False)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    calculate_number_of_parameters = False
    if calculate_number_of_parameters:
        counter = {
            "embedding": 0,
            "layernorm": 0,
            "trm": 0,
            "highway": 0,
            "final": 0,
            "all": 0
        }
        for n, p in model.named_parameters():
            size = p.numel()
            if "highway" in n:
                counter["highway"] += size
            elif "layer" in n:
                counter["trm"] += size
            elif "LayerNorm" in n:
                counter["layernorm"] += size
            elif "embedding" in n:
                counter["embedding"] += size
            else:
                print(n)
                counter["final"] += size
            counter["all"] += size
        print(counter)
        exit(0)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if train_strategy == 'raw':
        # the original bert model
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        ("highway" not in n) and (not any(nd in n for nd in no_decay))],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if
                        ("highway" not in n) and (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0}
        ]
    elif train_strategy == "only_highway":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        ("highway" in n) and (not any(nd in n for nd in no_decay))],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if
                        ("highway" in n) and (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0}
        ]
    elif train_strategy in ['all']:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    else:
        raise NotImplementedError("Wrong training strategy!")

    optimizers = [AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)]
    if len(optimizers) == 1:
        optimizer = optimizers[0]
    else:
        optimizer = optimizers[-1]

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    if args.fp16:
        # haven't fixed for multiple optimizers yet!
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    fout = open(args.output_dir + "/layer_example_counter", 'w')

    print_loss_switch = False  # only True for debugging
    tqdm_disable = print_loss_switch or (args.local_rank not in [-1, 0])

    for epoch_num in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=tqdm_disable)
        n_layers = model.module.num_layers if hasattr(model, 'module') else model.num_layers  # multi-gpu compatible
        layer_example_counter = {i: 0 for i in range(n_layers+1)}
        cumu_loss = 0.0
        epoch_loss = 0.0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            if train_strategy=='limit':
                inputs['train_strategy'] = train_strategy + args.limit_layer
            else:
                inputs['train_strategy'] = train_strategy
            inputs['layer_example_counter'] = layer_example_counter
            inputs['step_num'] = step
            outputs = model(**inputs)
            losses = outputs[0]  # model outputs are always tuple in transformers (see doc)

            for i in range(len(losses)):
                # i is the index of loss terms and also the optimizer to backprop it
                loss = losses[i]
                optimizer = optimizers[i]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=True)
                tr_loss += loss.item()
                if print_loss_switch and step%10==0:
                    print(cumu_loss/10)
                    cumu_loss = 0
                cumu_loss += loss.item()
                epoch_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    if i == len(losses) - 1:
                        scheduler.step()  # Update learning rate schedule
                        global_step += 1
                    model.zero_grad()

                    # this block doesn't work as expected any more
                    # but it only affects tensorboard (which i don't care)
                    if args.local_rank in [-1, 0] \
                            and args.logging_steps > 0 \
                            and global_step % args.logging_steps == 0:
                        # Log metrics
                        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

        if args.local_rank in [-1, 0]:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'epoch-{}'.format(epoch_num))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

        print('Epoch loss: ', epoch_loss)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    fout.close()
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", output_layer=-1, eval_highway=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}
    for eval_partition in sorted(
            os.listdir(args.eval_collection_dir), key=lambda x: int(x[x.find('partition')+9:])
    ):
        if int(eval_partition[9:]) not in args.todo_partition_list:  # the number
            continue
        eval_dataset, eval_qpids = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=True,
            eval_dir_partition=(args.eval_collection_dir, eval_partition),
            testset=args.testset)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size  #* max(1, args.n_gpu)  # multi-gpu eval disabled
        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and False:  # multi-gpu eval disabled
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        exit_layer_counter = {(i + 1): 0 for i in range(model.num_layers)}
        rel_logit_collection = []
        prob_collection = []
        exit_layer_collection = []
        st = time.time()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                               'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                if output_layer >= 0:
                    inputs['output_layer'] = output_layer
                outputs = model(**inputs)
                if eval_highway:
                    for j in range(outputs[1].shape[0]):
                        rel_logit_collection.append(
                            [torch.softmax(x[0], dim=1)[j][1].cpu().item() for x in outputs[2]['highway'][:-1]] + \
                            [torch.softmax(outputs[1], dim=1)[j][1].cpu().item()]
                        )

                    exit_layer_counter[outputs[-1]] += 1
                    exit_layer_collection.append(outputs[-1])
                tmp_eval_loss, logits = outputs[:2]
                tmp_eval_loss = tmp_eval_loss[-1]

                eval_loss += tmp_eval_loss.mean().item()
                prob_collection.append(  # for early exit eval
                    torch.softmax(logits, dim=1)[0][1].cpu().item()
                )
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_time = time.time() - st
        print("Eval time:", eval_time)

        if eval_highway:
            # also record correctness per layer
            save_path = args.plot_data_dir + \
                         args.model_name_or_path[2:]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path + "/correctness_layer{}.npy".format(output_layer),
                    np.array(np.argmax(preds, axis=1) == out_label_ids))
            np.save(save_path + "/prediction_layer{}.npy".format(output_layer),
                    np.array(np.argmax(preds, axis=1)))

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        if eval_highway:
            print("Exit layer counter", exit_layer_counter)
            actual_cost = sum([l * c for l, c in exit_layer_counter.items()])
            full_cost = len(eval_dataloader) * model.num_layers
            print("Expected saving", actual_cost / full_cost)

            if args.output_score_file:
                split = 'eval' if args.testset else 'dev'

                if args.quick_eval:
                    # all layers evaluation
                    for i in range(model.num_layers):
                        layer_dir = os.path.join(args.evaluation_dir, 'layer'+str(i))
                        if not os.path.exists(layer_dir):
                            os.makedirs(layer_dir)
                        submit_fname = os.path.join(
                            layer_dir,
                            split+'.'+eval_partition+'.score'
                        )
                        with open(submit_fname, 'w') as fout:
                            for j in range(len(eval_qpids)):
                                print('{}\t{}\t{}'.format(eval_qpids[j][0],
                                                          eval_qpids[j][1],
                                                          rel_logit_collection[j][i]),
                                      file=fout)

                else:
                    # early exit evaluation
                    if not os.path.exists(args.evaluation_dir):
                        os.makedirs(args.evaluation_dir)
                    submit_fname = os.path.join(
                        args.evaluation_dir,
                        split + '.' + eval_partition + '.score'
                    )
                    with open(submit_fname, 'w') as fout:
                        for j in range(len(eval_qpids)):
                            print('{}\t{}\t{}\t{}'.format(eval_qpids[j][0],
                                                          eval_qpids[j][1],
                                                          prob_collection[j],
                                                          exit_layer_collection[j]),
                                  file=fout)
                    np.save(
                        os.path.join(args.evaluation_dir, split+'.'+eval_partition+'.npy'),
                        np.array(exit_layer_counter))

        del eval_dataset, eval_qpids
        gc.collect()

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False,
                            testset=False, eval_dir_partition=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    split = 'train'
    if evaluate:
        split = 'dev'
    if testset:
        split = 'test'
    if eval_dir_partition is None:
        file_name = processor.get_file_name()[split]
        data_dir = args.data_dir
    else:
        data_dir, file_name = eval_dir_partition
    cached_features_fname = 'cached_{}_{}_{}_{}__{}'.format(
        split,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task),
        file_name
    )
    if eval_dir_partition is None:
        cached_features_file = os.path.join(args.data_dir, cached_features_fname)
    else:
        cached_features_file = os.path.join(args.data_dir, 'partition_cache', cached_features_fname)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features, qpids = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
            # after swap: contradiction, neutral, entailment
        qpids = []
        if not evaluate:
            examples = processor.get_train_examples(args.data_dir)
        else:
            if not testset:
                examples = processor.get_dev_examples(
                    data_dir, fname=file_name)
            else:
                examples = processor.get_test_examples(
                    data_dir, fname=file_name)
            for e in examples:
                qpids.append(e.guid.split('-')[-2:])
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save([features, qpids], cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    gc.collect()
    return (dataset, qpids)

def main(args):

    if args.train_routine == 'limit':
        finished_layers = os.listdir(args.plot_data_dir + args.output_dir)
        for fname in finished_layers:
            layer = fname[len('layer-'):fname.index('.npy')]
            try:
                if layer == args.limit_layer:  # both are type'str'
                    # already done
                    exit(0)
            except ValueError:
                pass

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)

    config.divide = args.train_routine

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    model.core.encoder.set_early_exit_thresholds(args)
    model.core.init_highway_pooler()

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

        if args.train_routine in ["raw"]:
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        elif args.train_routine == "two_stage":
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            result = evaluate(args, model, tokenizer, prefix="")
            print_result = get_wanted_result(result)
            print("result: {}".format(print_result))

            # second stage
            train(args, train_dataset, model, tokenizer, train_strategy="only_highway")

        elif args.train_routine in ['all']:
            global_step, tr_loss = train(args, train_dataset, model, tokenizer,
                                         train_strategy='all')
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        else:
            raise NotImplementedError("Wrong training routine!")

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.core.encoder.set_early_exit_thresholds(args)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix,
                              eval_highway=args.eval_highway,
                              output_layer=int(args.limit_layer))
            print_result = get_wanted_result(result)
            print("result: {}".format(print_result))
            if args.train_routine=='limit':
                save_fname = args.plot_data_dir + \
                             args.output_dir + f"/layer-{args.limit_layer}.npy"
                if not os.path.exists(os.path.dirname(save_fname)):
                    os.makedirs(os.path.dirname(save_fname))
                np.save(save_fname, np.array([print_result]))

            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main(args)

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import csv
import os
import logging
import argparse
import random
import sys

from tqdm import tqdm, trange

from sklearn.metrics import f1_score
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from at import FGM


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_unlabel_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a comma separated value file."""
        # import pandas as pd
        # df = pd.read_csv(input_file)
        # lines = []
        # for (i, row) in df.iterrows():
        #     import ipdb
        #     ipdb.set_trace()
        #     lines.append(row)
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=",")
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MLDProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    def __init__(self, lang, tokenizer):
        self.lang = lang
        self.tokenizer = tokenizer


    def get_train_examples(self, data_dir):
        """See base class."""

        lines = self._read_tsv(os.path.join(data_dir, "english.train.1000"))

        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ('train', i)
            text_a = line[1]
            label = line[0]
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))

        return examples

    def get_unlabel_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, self.lang + ".train.1000"))

        src_examples = []
        trg_examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ('uns', i)
            text_a = line[1]
            label = line[0]
            src_examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
            trg_examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))

        return src_examples, trg_examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        src_lines = self._read_tsv(os.path.join(data_dir, self.lang + ".dev"))
        trg_lines = self._read_tsv(os.path.join(data_dir, "english.dev"))
        src_examples = []
        trg_examples = []
        for (i, line) in enumerate(src_lines):
            guid = "dev-%d" % (i)
            text_a = line[1]
            label = line[0]
            src_examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        for (i, line) in enumerate(trg_lines):
            guid = "dev-%d" % (i)
            text_a = line[1]
            label = line[0]
            trg_examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))

        return src_examples

    def get_test_examples(self, data_dir):
        """See base class."""
        src_lines = self._read_tsv(os.path.join(data_dir, self.lang + ".test"))
        src_examples = []
        trg_examples = []
        for (i, line) in enumerate(src_lines):
            guid = "test-%d" % (i)
            text_a = line[1]
            label = line[0]
            src_examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))

        return src_examples

    def get_labels(self):
        """See base class."""
        return ["CCAT", "ECAT", "MCAT", "GCAT"]

class CLYProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        lines = self._read_tsv(os.path.join(data_dir, "en_yelp_train.tsv"))

        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ('train', i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_unlabel_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "zh_hotel_train.tsv"))

        src_examples = []
        trg_examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ('uns', i)
            text_a = line[1]
            label = line[0]
            src_examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
            trg_examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))

        return src_examples, trg_examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        src_lines = self._read_tsv(os.path.join(data_dir, "en_yelp_dev.tsv"))
        trg_lines = self._read_tsv(os.path.join(data_dir, "zh_hotel_test.tsv"))
        src_examples = []
        trg_examples = []
        for (i, line) in enumerate(src_lines):
            guid = "dev-%d" % (i)
            text_a = line[1]
            label = line[0]
            src_examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        for (i, line) in enumerate(trg_lines):
            guid = "dev-%d" % (i)
            text_a = line[1]
            label = line[0]
            trg_examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))

        return trg_examples

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5"]



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def cons_rate(out, src_out):
    outputs = np.argmax(out, axis=1)
    src_o = np.argmax(src_out, axis=1)
    return np.sum(outputs == src_o)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x



def sort(train_examples, eval_examples, label_list, id2conf, num_k):
    label2id = {}
    id2train = []
    ud_train_examples, ud_unlabel_examples = train_examples, eval_examples
    for i, item in enumerate(id2conf):
        if item[0] not in label2id.keys():
            label2id[item[0]] = {}
        label2id[item[0]][i] = item[1]

    for i in range(len(label_list)):
        if i not in label2id.keys():
            print("no class " + str(i))
            continue
        i2cs = label2id[i]

        sorted_i2cs = sorted(i2cs.items(), reverse=True, key=lambda kv: kv[1])

        for i2c in sorted_i2cs[:num_k]:
            id2train.append(i2c[0])
            ex = eval_examples[i2c[0]]
            ex.label_id = i
            ud_train_examples.append(ex)

    for index in sorted(id2train, reverse=True):
        del ud_unlabel_examples[index]

    return ud_train_examples, ud_unlabel_examples


def train(model, optimizer, train_examples, eval_examples, best_acc, args):
    src_train_features = train_examples
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", args.num_train_steps)
    src_input_ids = torch.tensor([f.input_ids for f in src_train_features], dtype=torch.long)
    src_input_mask = torch.tensor([f.input_mask for f in src_train_features], dtype=torch.long)
    src_segment_ids = torch.tensor([f.segment_ids for f in src_train_features], dtype=torch.long)
    src_label_ids = torch.tensor([f.label_id for f in src_train_features], dtype=torch.long)
    train_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.adv_training:
        fgm = FGM(model)
    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            loss, pool_rep = model(input_ids, segment_ids, input_mask, label_ids)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if args.adv_training:
                fgm.attack()  
                loss_adv, _ = model(input_ids, segment_ids, input_mask, label_ids)
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()  
                fgm.restore() 


            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(args.global_step / args.t_total,
                                                                  args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                args.global_step += 1


        # validation starts
        eval_s_features = eval_examples
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        src_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
        src_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
        src_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
        src_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
        eval_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred_l, true_l = [], []
        reps = []
        labels = []

        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            src_input_ids, src_input_mask, src_segment_ids, src_label_ids = batch

            with torch.no_grad():
                tmp_eval_loss, pooled_ouput = model(src_input_ids, src_segment_ids, src_input_mask, src_label_ids)
                logits, pooled_ouput = model(src_input_ids, src_segment_ids, src_input_mask)

            reps.append(pooled_ouput.cpu().detach().numpy())
            logits = logits.detach().cpu().numpy()
            label_ids = src_label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            labels.append(label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += src_input_ids.size(0)
            nb_eval_steps += 1
            pred_labels = np.argmax(logits, axis=1)
            pred_l.append(pred_labels)
            true_l.append(label_ids)

        pred_l = np.concatenate(pred_l, axis=None)
        true_l = np.concatenate(true_l, axis=None)
        f1 = f1_score(true_l, pred_l, average='macro')
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        logger.info("  acc = %f", eval_accuracy)
        if eval_accuracy > best_acc:
            best_acc = eval_accuracy
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), args.output_model_file)
        model.train()
    return best_acc

def eval(model, train_example, eval_examples, label_list, args):

    eval_s_features = eval_examples
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    src_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
    src_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
    src_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
    src_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
    eval_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    pred_l, true_l = [], []
    reps = []
    labels = []
    cons_r = 0
    # variable for self-training:
    id2maxp = []

    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        src_input_ids, src_input_mask, src_segment_ids, src_label_ids = batch

        with torch.no_grad():
            tmp_eval_loss, pooled_ouput = model(src_input_ids, src_segment_ids, src_input_mask, src_label_ids)
            logits, pooled_ouput = model(src_input_ids, src_segment_ids, src_input_mask)

        reps.append(pooled_ouput.cpu().detach().numpy())
        logits = torch.nn.functional.softmax(logits)
        logits = logits.detach().cpu().numpy()
        label_ids = src_label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)
        labels.append(label_ids)
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += src_input_ids.size(0)
        nb_eval_steps += 1
        pred_labels = np.argmax(logits, axis=1)
        pred_l.append(pred_labels)
        true_l.append(label_ids)

        soft_label = np.argmax(logits, axis=1)
        confi = np.max(logits, axis=1)
        for i in range(len(soft_label)):
            id2maxp.append((soft_label[i], confi[i]))

    if args.self_train:
        ud_train_examples, ud_unlabel_examples = sort(train_example, eval_examples, label_list, id2maxp, args.num_k)
    else:
        ud_train_examples, ud_unlabel_examples = train_example, eval_examples
    pred_l = np.concatenate(pred_l, axis=None)
    true_l = np.concatenate(true_l, axis=None)
    f1 = f1_score(true_l, pred_l, average='micro')
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'global_step': args.global_step,
              'loss': args.tr_loss / args.nb_tr_steps,
              'f1': f1}
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return ud_train_examples, ud_unlabel_examples


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--resume",
                        default=False,
                        action='store_true',
                        help="Whether to resume training.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--adv_training",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using adversarial training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=512,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_k",
                        default=40,
                        type=int,
                        help="Instances Selected.")
    parser.add_argument("--num_self_train",
                        default=6,
                        type=int,
                        help="Instances Selected.")
    parser.add_argument("--lang",
                        default='zh',
                        type=str,
                        help="language for zero-shot")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    processors = {
        'mld': MLDProcessor,
        'cly': CLYProcessor

    }

    num_labels_task = {
        'mld': 4,
        'cly': 5
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.output_dir = '/freespace/local/xd48/bert_output/' + args.output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        print("load model from directory ({})".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    processor = processors[task_name](args.lang, tokenizer)
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()



    args.self_train = True

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.resume:
        print('Resume Training')
        args.old_output_dir = '/freespace/local/xd48/bert_output/' + args.old_output_dir
        saved_output_model_file = os.path.join(args.old_output_dir, "pytorch_model.bin")
        model_state_dict = torch.load(saved_output_model_file)
        model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict,
                                                              num_labels=num_labels)


    args.output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

    self_train_time = args.num_self_train
    args.n_gpu = n_gpu
    args.device = device
    args.num_train_steps = num_train_steps

    eval_examples = processor.get_dev_examples(args.data_dir)
    unlabel_examples = processor.get_unlabel_examples(args.data_dir)

    if args.do_train:
        src_train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
    else:
        src_train_features = None

    ul_s_features = convert_examples_to_features(
        unlabel_examples[0], label_list, args.max_seq_length, tokenizer)

    ud_train_fea, ud_unlabel_fea = src_train_features, ul_s_features

    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    best_acc = 0

    if args.do_train:
        for time in range(self_train_time):
            args.tr_loss = 0
            args.nb_tr_steps = 1
            args.global_step = 0
            model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                    args.local_rank), num_labels=num_labels)

            if args.fp16:
                model.half()
            model.to(device)
            if args.local_rank != -1:
                try:
                    from apex.parallel import DistributedDataParallel as DDP
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

                model = DDP(model)
            elif n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = num_train_steps
            if args.local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()
            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=args.learning_rate,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=args.learning_rate,
                                     warmup=args.warmup_proportion,
                                     t_total=t_total)

            args.t_total = t_total

            best_acc = train(model, optimizer, ud_train_fea, eval_features, best_acc, args)

            if args.self_train and time != self_train_time-1:
                ud_train_fea, ud_unlabel_fea = eval(model, ud_train_fea, ud_unlabel_fea, label_list, args)

    args.self_train = False
    model_state_dict = torch.load(args.output_model_file)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        args.tr_loss = 0
        args.nb_tr_steps = 1
        args.global_step = 0
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer)
        eval(model, src_train_features, test_features, label_list, args)


if __name__ == "__main__":
    main()

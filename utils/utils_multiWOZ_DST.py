import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
import ast
from collections import Counter
from collections import OrderedDict
from embeddings import GloveEmbedding, KazumaCharEmbedding
from tqdm import tqdm
import os
import pickle
from random import shuffle

from .fix_label import *

EXPERIMENT_SERVICES = []
TRAIN_SERVICES = []
DEV_SERVICES = []
except_services = [u'Hotels_3',u'Services_3',u'Alarm_1']

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split():
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
                for v in value.split():
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id, sequicity, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_service = data_info['turn_service']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.gating_label = data_info['gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.generate_y = data_info["generate_y"]
        self.sequicity = sequicity
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        turn_uttr = self.turn_uttr[index]
        turn_service = self.preprocess_domain(self.turn_service[index])
        generate_y = self.generate_y[index]
        generate_y = self.preprocess_slot(generate_y, self.trg_word2id)
        context = self.dialog_history[index]
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]

        item_info = {
            "ID":ID,
            "turn_id":turn_id,
            "turn_belief":turn_belief,
            "gating_label":gating_label,
            "context":context,
            "context_plain":context_plain,
            "turn_uttr_plain":turn_uttr,
            "turn_service":turn_service,
            "generate_y":generate_y,
            }
        return item_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book","").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story

    def preprocess_domain(self, turn_service):
        services = {}
        for idx, service in enumerate(EXPERIMENT_SERVICES):
            services[service] = idx

        return services[turn_service]


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)

        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths) # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i,:end,:] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
    gating_label = torch.tensor(item_info["gating_label"])
    turn_service = torch.tensor(item_info["turn_service"])
    #turn_slot = torch.tensor(item_info["turn_slot"])

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        gating_label = gating_label.cuda()
        turn_service = turn_service.cuda()
        #turn_slot = turn_slot.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_service"] = turn_service
    #item_info["turn_slot"] = turn_slot
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths
    return item_info

def read_langs(file_name, gating_dict, SLOTS, dataset, lang, mem_lang, sequicity, training, max_line = None):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    service_counter = {}
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first
        for dial_dict in dials:
            if (args["all_vocab"] or dataset=="train") and training:
                for ti, turn in enumerate(dial_dict["turns"]):
                    lang.index_words(turn["system_utterance"], 'utter')
                    lang.index_words(turn["utterance"], 'utter')
        # determine training data ratio, default is 100%
        if training and dataset=="train" and args["data_ratio"]!=100:
            random.Random(10).shuffle(dials)
            dials = dials[:int(len(dials)*0.01*args["data_ratio"])]

        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""
            last_belief_dict = {}
            dial_services = dial_dict["services"]
            # Filtering and counting domains
            for service in dial_dict["services"]:
                if service not in EXPERIMENT_SERVICES:
                    continue
                if service not in service_counter.keys():
                    service_counter[service] = 0
                service_counter[service] += 1

            # Unseen domain setting
            if args["only_service"] != "" and args["only_service"] not in dial_dict["services"]:
                continue
            for excepts in except_services :
                if (len(except_services) != 0 and dataset == "test" and excepts not in dial_dict["services"]) or \
                (len(except_services) != 0 and dataset != "test" and [excepts] == dial_dict["services"]):
                    continue
            # Reading data
            for ti, turn in enumerate(dial_dict["turns"]):
                turn_service = turn["frames"][0]["service"]
                turn_id = turn["turn_idx"]
                turn_uttr = turn["system_utterance"] + " ; " + turn["utterance"]
                turn_uttr_strip = turn_uttr.strip()
                dialog_history +=  (turn["system_utterance"] + " ; " + turn["utterance"] + " ; ")
                if turn_id > 2:
                    dhistory = dialog_history.split(';')
                    dialog_history = ""
                    i = len(dhistory) - 7
                    while i < len(dhistory)-1:
                        dialog_history +=  dhistory[i] + ";" + dhistory[i+1] + ";"
                        i += 2

                source_text = dialog_history.strip()
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)
                # Generate domain-dependent slot list
                slot_temp = SLOTS

                if dataset == "train":
                    except_services2 = except_services + [u'Travel_1', u'Weather_1'] #Single-Domainになかったので除去
                    if len(except_services) != 0:
                        for idx, excepts in enumerate(except_services2):
                            slot_temp = [k for k in slot_temp if excepts not in k]
                            turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if excepts not in k])

                    elif args["only_service"] != "":
                        slot_temp = [k for k in SLOTS if args["only_service"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_service"] in k])
                elif dataset == "dev":
                    if len(except_services) != 0:
                        for idx, excepts in enumerate(except_services):
                            slot_temp = [k for k in slot_temp if excepts not in k]
                            turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if excepts not in k])

                    elif args["only_service"] != "":
                        slot_temp = [k for k in SLOTS if args["only_service"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_service"] in k])
                else:
                    if len(except_services) != 0:
                        s = []
                        t = []
                        for excepts in except_services:
                            s += [k for k in SLOTS if excepts in k]
                            t += [(k, v) for k, v in turn_belief_dict.items() if excepts in k]
                        slot_temp = s
                        turn_belief_dict = OrderedDict(t)
                    elif args["only_service"] != "":
                        slot_temp = [k for k in SLOTS if args["only_service"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_service"] in k])

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]
                if (args["all_vocab"] or dataset=="train") and training:
                    mem_lang.index_words(turn_belief_dict, 'belief')

                """cur_serviceのスロットだけ抜き取る"""
                for sidx, service in enumerate(dial_services):
                    if sidx == 0:
                        turn_slot = [k for k in slot_temp if service in k]
                    else:
                        turn_slot += [k for k in slot_temp if service in k]

                class_label, generate_y, slot_mask, gating_label  = [], [], [], []
                start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys():
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])

                data_detail = {
                    "ID":dial_dict["dialogue_id"],
                    "services":dial_dict["services"],
                    "turn_service":turn_service,
                    "turn_id":turn_id,
                    "dialog_history":source_text,
                    "turn_belief":turn_belief_list,
                    "gating_label":gating_label,
                    "turn_uttr":turn_uttr_strip,
                    'generate_y':generate_y
                    }
                data.append(data_detail)

                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())

            cnt_lin += 1
            if(max_line and cnt_lin>=max_line):
                break

    # add t{} to the lang file
    if "t{}".format(max_value_len-1) not in mem_lang.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter')

    print("service_counter", service_counter)
    return data, max_resp_len, slot_temp


def get_seq(pairs, lang, mem_lang, batch_size, type, sequicity):
    if(type and args['fisher_sample']>0):
        shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    data_info = {}
    data_keys = pairs[0].keys()

    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            #print(pair[k])
            #input()
            data_info[k].append(pair[k])

    dataset = Dataset(data_info, lang.word2index, lang.word2index, sequicity, mem_lang.word2index)

    if args["imbalance_sampler"] and type:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  # shuffle=type,
                                                  collate_fn=collate_fn,
                                                  sampler=ImbalancedDatasetSampler(dataset))
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=type,
                                                  collate_fn=collate_fn)
    return data_loader


def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def get_all_slot_information(schemas, schemas2):
    SLOTS = []
    for schema in schemas:
        EXPERIMENT_SERVICES.append(schema['service_name'])
        for slot in schema['slots']:
            SLOTS.append('{}-{}'.format(schema['service_name'],slot['name']))
    for schema in schemas2:
        if schema['service_name'] not in EXPERIMENT_SERVICES:
            EXPERIMENT_SERVICES.append(schema['service_name'])
            for slot in schema['slots']:
                SLOTS.append('{}-{}'.format(schema['service_name'],slot['name']))

    return SLOTS

def get_slot_information(schemas, train = True):
    SLOTS = []
    for schema in schemas:
        if train:
            TRAIN_SERVICES.append(schema['service_name'])
        else:
            DEV_SERVICES.append(schema['service_name'])
        for slot in schema['slots']:
            SLOTS.append('{}-{}'.format(schema['service_name'],slot['name']))
        #ontology_domains = dict([(k, v) for k, v in schema['slots'].items() if k.split("-")[0] in EXPERIMENT_SERVICES])
        #SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def prepare_data_seq(training, task="dst", sequicity=0, batch_size=100):
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
    file_train = 'data/dstc8/train_dials.json'
    file_dev = 'data/dstc8/dev_dials.json'
    file_test = 'data/dstc8/test_dials.json'
    # Create saving folder
    if args['path']:
        folder_name = args['path'].rsplit('/', 2)[0] + '/'
    else:
        #folder_name = 'save/{}-'.format(args["decoder"])+args["addName"]+args['dataset']+str(args['task'])+'/'
        folder_name = 'save/{}-'.format(args["decoder"])+args["addName"]+"dstc8"+str(args['task'])+'/'
    print("folder_name", folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # load domain-slot pairs from ontology
    schema_train = json.load(open("data/dstc8/train/schema.json", 'r'))
    schema_dev = json.load(open("data/dstc8/dev/schema.json", 'r'))
    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json"))
    ALL_SLOTS = get_all_slot_information(schema_train, schema_dev)
    TRAIN_SLOTS = get_slot_information(schema_train)
    DEV_SLOTS = get_slot_information(schema_dev, False)
    gating_dict = {"ptr":0, "dontcare":1, "none":2}
    # Vocabulary
    lang, mem_lang = Lang(), Lang()
    lang.index_words(ALL_SLOTS, 'slot')
    mem_lang.index_words(ALL_SLOTS, 'slot')

    lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
    mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'

    if training:
        pair_train, train_max_len, slot_train = read_langs(file_train, gating_dict, TRAIN_SLOTS, "train", lang, mem_lang, sequicity, training)
        #print(pair_train)
        #print(train_max_len)
        #print(slot_train)
        #input()
        #pair_train, train_max_len, slot_train = read_langs(file_train, gating_dict, ALL_SLOTS, "train", lang, mem_lang, sequicity, training)
        train = get_seq(pair_train, lang, mem_lang, batch_size, True, sequicity)
        nb_train_vocab = lang.n_words
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, DEV_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev   = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test  = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)
        if os.path.exists(folder_name+lang_name) and os.path.exists(folder_name+mem_lang_name):
            print("[Info] Loading saved lang files...")
            with open(folder_name+lang_name, 'rb') as handle:
                lang = pickle.load(handle)
            with open(folder_name+mem_lang_name, 'rb') as handle:
                mem_lang = pickle.load(handle)
        else:
            print("[Info] Dumping lang files...")
            with open(folder_name+lang_name, 'wb') as handle:
                pickle.dump(lang, handle)
            with open(folder_name+mem_lang_name, 'wb') as handle:
                pickle.dump(mem_lang, handle)
        emb_dump_path = 'data/emb{}.json'.format(len(lang.index2word))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)
    else:
        with open(folder_name+lang_name, 'rb') as handle:
            lang = pickle.load(handle)
        with open(folder_name+mem_lang_name, 'rb') as handle:
            mem_lang = pickle.load(handle)

        pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, DEV_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev   = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test  = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)

    test_4d = []
    if len(except_services) != 0:
        pair_test_4d, _, _ = read_langs(file_test, gating_dict, DEV_SLOTS, "dev", lang, mem_lang, sequicity, training)
        #pair_test_4d, _, _ = read_langs(file_test, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        test_4d  = get_seq(pair_test_4d, lang, mem_lang, eval_batch, False, sequicity)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % nb_train_vocab )
    print("Vocab_size Belief %s" % mem_lang.n_words )
    print("Max. length of dialog words for RNN: %s " % max_word)
    print("USE_CUDA={}".format(USE_CUDA))

    SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[0]))))
    print(SLOTS_LIST[0])
    print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
    print(SLOTS_LIST[3])
    LANG = [lang, mem_lang]
    return train, dev, test, test_4d, LANG, SLOTS_LIST, gating_dict, nb_train_vocab



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.turn_service[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

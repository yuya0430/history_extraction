import re
import os
import json
import glove
import urllib.request
import zipfile
import logging
import pickle
import math
import numpy as np
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import glob
from collections import Counter
from slot_similarity import Getsimilar_slot

fin = open('/work/yuya0430/dstc8/utils/mapping.pair','r')
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text

def normalize(text):
    # lower case every word
    text = text.lower() #小文字化

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text) #文字列の先頭か末尾が空白のとき削除


    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = text.replace('_', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text) # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)

    #単語の書き換え
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def slot_cossim(slot1, slot2, vectors):
    slot1_words = slot1.split(' ')
    slot2_words = slot2.split(' ')
    count = 0
    s1_v = np.zeros(300)
    for s1_w in slot1_words:
        if s1_w not in vectors.keys():
            continue
        else:
            s1_v += vectors[s1_w]
            count += 1

    if count == 0:
        count = 1
        #input()
    s1_v /= count

    count = 0
    s2_v = np.zeros(300)
    for s2_w in slot2_words:
        if s2_w not in vectors.keys():
            continue
        else:
            s2_v += vectors[s2_w]
            count += 1
    if count == 0:
        count = 1
        #input()
    s2_v /= count
    if (s1_v == np.zeros(300)).all() or (s2_v == np.zeros(300)).all():
        result = 0.0
    else:
        result = cos_sim(s1_v, s2_v)

    return result

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def slot_cossim(value1_list, value2_list, vectors):
    count = 0
    v1_v = np.zeros(300)
    for value1 in value1_list :
        value1_word = value1.split(' ')
        for v1_w in value1_word:
            if v1_w not in vectors.keys():
                continue
            else:
                v1_v += vectors[v1_w]
                count += 1

    if count == 0:
        count = 1
        #input()
    v1_v /= count

    count = 0
    v2_v = np.zeros(300)
    for value2 in value2_list :
        value2_word = value2.split(' ')
        for v2_w in value2_word:
            if v2_w not in vectors.keys():
                continue
            else:
                v2_v += vectors[v2_w]
                count += 1

    if count == 0:
        count = 1
        #input()
    v2_v /= count
    if (v1_v == np.zeros(300)).all() or (v2_v == np.zeros(300)).all():
        result = 0.0
    else:
        result = cos_sim(v1_v, v2_v)

    return result

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(message)s")
logger = logging.getLogger("glove")

# ファイルからvalueを抽出
if not os.path.exists('value_list.pickle'):
    with open('../data/dstc8/train/schema.json', 'r', encoding='shift_jis') as f:
        data = json.load(f)

    # slotを抽出
        value_list = {}
        for service in data:
            for k in service['slots']:
                value_list[service['service_name'] + '-' + k['name']] = []


    all_file = glob.glob("../data/dstc8/train/dialogues_*.json")
    for file in all_file:
        with open(file, 'r', encoding='shift_jis') as f:
            data = json.load(f)
            for didx, dialog in enumerate(data):
                for tidx, turn in enumerate(dialog['turns']):
                    utterance = turn['utterance']
                    for fidx, frame in enumerate(turn['frames']):
                        service = frame['service']
                        if tidx % 2 == 0:
                            for slot_name, slot_value_list in frame['state']['slot_values'].items():
                                for slot_value in slot_value_list:
                                    service_slot = service + '-' + slot_name
                                    slot_value = normalize(slot_value)
                                    if service_slot in value_list.keys() and slot_value not in value_list[service_slot]:
                                        value_list[service_slot].append(slot_value)

                        else:
                            for aidx, action in enumerate(frame['actions']):
                                #if action['slot'] == '' or action['slot'] == 'count' or action['slot'] == 'intent':
                                #    continue
                                service_slot = service + '-' + action['slot']
                                for slot_value in action['values']:
                                    slot_value = normalize(slot_value)
                                    if service_slot in value_list.keys() and slot_value not in value_list[service_slot]:
                                        value_list[service_slot].append(slot_value)

                        for sidx, slot in enumerate(frame['slots']):
                            start = slot['start']
                            end = slot['exclusive_end']
                            slot_name = slot['slot']
                            slot_value = normalize(utterance[start:end])
                            service_slot = service + '-' + slot_name
                            if service_slot in value_list.keys() and slot_value not in value_list[service_slot]:
                                value_list[service_slot].append(slot_value)
    f = open('value_list.pickle','wb')
    pickle.dump(value_list,f)
    f.close

else:
    f = open('value_list.pickle', 'rb')
    value_list = pickle.load(f)
    f.close

my_vocab = []
for values in value_list.values():
    for value in values:
        vocab = value.split(' ')
        for v in vocab:
            if v not in my_vocab:
                my_vocab.append(v)

if not os.path.exists('value_embedding.pickle'):
    with open('./glove/emb/glove.840B.300d.txt') as emb:
        data1 = emb.read()
    emb.close()
    my_vectors = {}
    check_list = []
    lines1 = data1.split('\n')
    for line in lines1:
        vectors = line.split(' ')
        if vectors[0] in my_vocab:
            check_list.append(vectors[0])
            my_vectors[vectors[0]] = [float(s) for s in vectors[1:]]

    check = [c for c in my_vocab if c not in check_list]

    f = open('value_embedding.pickle','wb')
    pickle.dump(my_vectors,f)
    f.close

else:
    f = open('value_embedding.pickle','rb')
    my_vectors = pickle.load(f)
    f.close

if not os.path.exists('value_matrix.pickle'):
    cos_matrix = np.zeros((215, 215), dtype=np.float64)
    slot_list = list(value_list.keys())

    slot_dict = {}
    for idx, slot in enumerate(slot_list):
        slot_dict[slot] = idx


    for idx1, slot1 in enumerate(slot_list) :
        for idx2, slot2 in enumerate(slot_list[idx1:]) :
            res = slot_cossim(value_list[slot1], value_list[slot2], my_vectors)
            cos_matrix[idx1][idx2+idx1] = res
            cos_matrix[idx2+idx1][idx1] = res

    similarslot_list = Getsimilar_slot()
    plot_data = []

    for idx, similarslot in enumerate(similarslot_list):
        idx1 = slot_dict[similarslot[0][0]]
        idx2 = slot_dict[similarslot[1][0]]
        idx3 = slot_dict[similarslot[2][0]]

        plot_data.append({'slot':similarslot[0][0], 'similar':similarslot[1][0], 'slot_similar':similarslot[1][1], 'value_similar': cos_matrix[idx1][idx2]})
        plot_data.append({'slot':similarslot[0][0], 'similar':similarslot[2][0], 'slot_similar':similarslot[2][1], 'value_similar': cos_matrix[idx1][idx3]})

    f = open('value_matrix.pickle','wb')
    pickle.dump(plot_data,f)
    f.close

else:
    f = open('value_matrix.pickle','rb')
    plot_data = pickle.load(f)
    f.close


f = open('value_list.txt', 'w')
X = []
Y = []
for d in plot_data:
    X.append(d['slot_similar'])
    Y.append(d['value_similar'])
    f.write("slot:{} \n similar:{} \n slot_similar:{} \n value_similar:{} \n\n".format(d['slot'],d['similar'],d['slot_similar'],d['value_similar']))

clf = linear_model.LinearRegression()
X2 = [[x] for x in X]
clf.fit(X2, Y) # 予測モデルを作成

# 散布図
plt.scatter(X2, Y)

# 回帰直線
x = np.linspace(0,1,100)
y = clf.coef_ * x + clf.intercept_
plt.plot(x,y)
#plt.plot(X2, clf.predict(X2))
plt.xlabel('slotの類似度', fontsize=18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.ylabel('value_listの類似度', fontsize=18)
plt.tick_params(labelsize=11)
plt.grid()
plt.savefig('figure.png') # -----(2)

print("回帰係数= ", clf.coef_)
print("切片= ", clf.intercept_)
print("決定係数= ", clf.score(X2, Y))

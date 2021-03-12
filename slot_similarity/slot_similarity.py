import re
import os
import json
import glove
import urllib.request
import zipfile
import logging
import pickle
import numpy as np

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
    #text = re.sub('^\'', '', text)
    #text = re.sub('\'$', '', text)
    #text = re.sub('\'\s', ' ', text)
    #text = re.sub('\s\'', ' ', text)

    #単語の書き換え
    #for fromx, tox in replacements:
    #    text = ' ' + text + ' '
    #    text = text.replace(fromx, tox)[1:-1]

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

def Getsimilar_slot():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(message)s")
    logger = logging.getLogger("glove")

    # ファイルからスキーマを抽出
    with open('../data/train/schema.json', 'r', encoding='shift_jis') as f:
        data = json.load(f)

    # slotとslot説明を抽出
    slot_list = []

    for service in data:
        for k in service['slots']:
            slot_list += [{'name': normalize(k['name']).replace('_',' ') , 'description': normalize(k['description']), \
            'is_categorical': k['is_categorical'], 'possible_values': k['possible_values'], 'service_slot': (service['service_name'] + '-' + k['name'])}]

    my_vocab = []
    for slot in slot_list:
        name_vocab = slot['name'].split(' ')
        des_vocab = slot['description'].split(' ')
        vocab = name_vocab + des_vocab
        for v in vocab:
            if v not in my_vocab:
                my_vocab.append(v)


    if not os.path.exists('slot_embedding.pickle'):
        with open('./glove/emb/glove.840B.300d.txt') as emb:
            data1 = emb.read()
        emb.close()
        my_vectors = {}
        #check_list = []
        lines1 = data1.split('\n')
        for line in lines1:
            vectors = line.split(' ')
            if vectors[0] in my_vocab:
                #check_list.append(vectors[0])
                my_vectors[vectors[0]] = [float(s) for s in vectors[1:]]

        #check = [c for c in my_vocab if c not in check_list]
        #print(check)
        #input()

        f = open('slot_embedding.pickle','wb')
        pickle.dump(my_vectors,f)
        f.close

    else:
        f = open('slot_embedding.pickle','rb')
        my_vectors = pickle.load(f)
        f.close

    cos_matrix_name = np.zeros((215, 215), dtype=np.float64)
    cos_matrix_desc = np.zeros((215, 215), dtype=np.float64)
    for idx1, slot1 in enumerate(slot_list) :
        for idx2, slot2 in enumerate(slot_list[idx1:]) :
            res1 = slot_cossim(slot1['name'],slot2['name'], my_vectors)
            res2 = slot_cossim(slot1['description'],slot2['description'], my_vectors)
            cos_matrix_name[idx1][idx2+idx1] = res1
            cos_matrix_name[idx2+idx1][idx1] = res1
            cos_matrix_desc[idx1][idx2+idx1] = res2
            cos_matrix_desc[idx2+idx1][idx1] = res2

    cos_matrix = (cos_matrix_name + cos_matrix_desc)/2

    c = 0
    n = 3

    similarslot_list = []
    for idx, slot_dict in enumerate(slot_list):
        original_slot = slot_dict['service_slot']

        max_index = cos_matrix[idx].argsort()[::-1]

        similarslot = []
        similarslot.append([original_slot, 1.0, slot_dict['is_categorical'], slot_dict['possible_values']])
        for mi in max_index:
            if len(similarslot) == 3:
                break
            else:
                service, slot = slot_list[mi]['service_slot'].split('-')
                if similarslot[0][0].split('-')[0] != service: #元のスロットと異なるサービスを持つものに制限
                    #if similarslot[0][2]: #カテゴリカルスロットかどうかで分岐
                         #if slot_list[mi]['is_categorical']: #類似スロットもカテゴリカルスロットかどうか
                            #if len(set(similarslot[0][3]) & set(slot_list[mi]['possible_values'])) != 0: #possible_valuesに同じvalueが存在するかどうか
                                #similarslot.append([slot_list[mi]['service_slot'], cos_matrix[idx][mi], slot_list[mi]['is_categorical'], slot_list[mi]['possible_values']])

                    #else:
                    similarslot.append([slot_list[mi]['service_slot'], cos_matrix[idx][mi], slot_list[mi]['is_categorical'], slot_list[mi]['possible_values']])

                    #similarslot_list += [[slot_list[mi]['service_slot'], cos_matrix[idx][mi], slot_list[mi]['possible_values']] \
                    #for simslot in similarslot_list if simslot[0].split('-')[1] != slot]]
        similarslot_list.append(similarslot)
    return similarslot_list

def main():
    similarslot_list = Getsimilar_slot()
    n = 3
    c = 0
    f = open('similar_slot'+str(n)+'.txt', 'w')
    for similarslot in similarslot_list:
        if len(similarslot_list) == n:
            c += 1

        #n=3のとき
        f.write("slot:{} \nno1:{},{} \nno2:{},{} \n\n".format(similarslot[0][0],similarslot[1][0],\
        similarslot[1][1],similarslot[2][0],similarslot[2][1]))

        # n=2のとき
        #f.write("slot: {} \n no1:{}     {} \n\n".format(similarslot_list[0][0],similarslot_list[1][0],similarslot_list[1][1]))

        #n=1のとき
        #f.write("slot: {} \n\n".format(similarslot_list[0][0]))

    #f.write(str(c))

if __name__=='__main__':
    main()

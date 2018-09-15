#coding=utf-8

import numpy as np
import pickle
import json
import re

def load_vec_pkl(fname,vocab,k=300):
    """
    Loads 300x1 word vecs from word2vec
    """
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    w2v = pickle.load(open(fname,'rb'))
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]
    return w2v,k,W

def load_vec_txt(fname,vocab,k=300):
    f = open(fname)
    w2v={}
    W = np.zeros(shape=(vocab.__len__() + 1, k))

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]
    return w2v,k,W

def load_vec_txt_all(fname,vocab,k=300):
    f = open(fname)
    w2v={}
    vocab_w2v = {}

    i = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        if len(coefs) == k:
            w2v[word] = coefs
            vocab_w2v[str(word)] = i
            i += 1

    f.close()

    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    vocab_w2v["UNK"] = i

    W = np.zeros(shape=[i + vocab.__len__() + 1, k])


    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]

    for word in vocab_w2v:
        if not vocab.__contains__(word):
            vocab[word] = vocab.__len__() + 1
            W[vocab[word]] = w2v[word]

    W = W[:vocab.__len__()+1]


    return w2v, k, W, vocab


def load_vec_onehot(vocab_w_inx):
    """
    Loads 300x1 word vecs from word2vec
    """
    k=vocab_w_inx.__len__()

    W = np.zeros(shape=(vocab_w_inx.__len__()+1, k+1))


    for word in vocab_w_inx:
        W[vocab_w_inx[word],vocab_w_inx[word]] = 1.
    # W[1, 1] = 1.
    return k,W

def make_idx_data_index_EE_LSTM(file,max_s,source_vob,target_vob):
    """
    Coding the word sequence and tag sequence based on the digital index which provided by source_vob and target_vob
    :param the tag file: word sequence and tag sequence
    :param the word index map and tag index map: source_vob,target_vob
    :param the maxsent lenth: max_s
    :return: the word_index map, the index_word map, the tag_index map, the index_tag map,
    the max lenth of word sentence
    """

    data_s_all=[]
    data_t_all=[]
    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        t_sent = sent['tags']
        data_t = []
        data_s = []
        if len(s_sent) > max_s:

            # i=max_s-1
            # while i >= 0:
            #     data_s.append(source_vob[s_sent[i]])
            #     i-=1

            i = 0
            while i < max_s:
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
        else:

            # num=max_s-len(s_sent)
            # for inum in range(0,num):
            #     data_s.append(0)
            # i=len(s_sent)-1
            # while i >= 0:
            #     data_s.append(source_vob[s_sent[i]])
            #     i-=1

            i = 0
            while i < len(s_sent):
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)

        if len(t_sent) > max_s:
            for i in range(0,max_s):
                data_t.append(target_vob[t_sent[i]])
        else:
            for word in t_sent:
                data_t.append(target_vob[word])
            while len(data_t) < max_s:
                data_t.append(0)
        data_t_all.append(data_t)
    f.close()
    return [data_s_all,data_t_all]

def make_idx_data_index_EE_LSTM2(file, max_s, tag_vob):

    data_t_all=[]
    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))

        t_sent = sent['tags']
        data_t = []

        if len(t_sent) > max_s:
            for i in range(0, max_s):
                data_t.append(tag_vob[t_sent[i]])
        else:
            for word in t_sent:
                data_t.append(tag_vob[word])
            while len(data_t) < max_s:
                data_t.append(0)
        data_t_all.append(data_t)
    f.close()
    # print(data_t_all)
    return data_t_all

def make_idx_data_index_EE_LSTM3(file, max_s, source_vob):

    data_s_all=[]
    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        data_s = []
        if len(s_sent) > max_s:
            i = 0
            while i < max_s:
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
        else:
            i = 0
            while i < len(s_sent):
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)

    f.close()
    return data_s_all

def get_word_index(train, test):
    """
    Give each word an index
    :param the train file and the test file
    :return: the word_index map, the index_word map, the tag_index map, the index_tag map,
    the max lenth of word sentence
    """
    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    count = 1
    tarcount=1
    # count = 0
    # tarcount = 0
    max_s=0
    max_t=0

    f = open(train,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__() > max_s:
            max_s = sourc.__len__()

        target = sent['tags']

        if target.__len__() > max_t:
            max_t = target.__len__()
        for word in target:
            if not target_vob.__contains__(word):
                target_vob[word] = tarcount
                target_idex_word[tarcount] = word
                tarcount += 1
    f.close()

    f = open(test, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__()>max_s:
            max_s = sourc.__len__()

        target = sent['tags']
        if not source_vob.__contains__(target[0]):
                source_vob[target[0]] = count
                sourc_idex_word[count] = target[0]
                count += 1
        if target.__len__()> max_t:
            max_t = target.__len__()
        for word in target:
            if not target_vob.__contains__(word):
                target_vob[word] = tarcount
                target_idex_word[tarcount] = word
                tarcount += 1
    f.close()
    if not source_vob.__contains__("**END**"):
        source_vob["**END**"] = count
        sourc_idex_word[count] = "**END**"
        count+=1
    if not source_vob.__contains__("UNK"):
        source_vob["UNK"] = count
        sourc_idex_word[count] = "UNK"
        count+=1
    return source_vob,sourc_idex_word,target_vob,target_idex_word,max_s


def get_entitylabeling_index(entlabelingfile):
    """
    Give each entity pair an index
    :param the entlabelingfile file
    :return: the word_index map, the index_word map,
    the max lenth of word sentence
    """
    entlabel_vob = {}
    entlabel_idex_word = {}
    count = 1
    # count = 0
    max_s=0

    f = open(entlabelingfile,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tags']
        for word in sourc:

            if not entlabel_vob.__contains__(word):
                entlabel_vob[word] = count
                entlabel_idex_word[count] = word
                # print(count, ",,,,,,", entlabel_idex_word[count])
                count += 1
        if sourc.__len__()>max_s:
            max_s = sourc.__len__()

    f.close()

    return entlabel_vob,entlabel_idex_word,max_s

def get_Feature_index(labelingfile):
    """
    Give each feature labelling an index
    :param the entlabelingfile file
    :return: the word_index map, the index_word map,
    the max lenth of word sentence
    """
    label_vob = {}
    label_idex_word = {}
    count = 1
    # count = 0
    max_s=0

    f = open(labelingfile,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tags']
        for word in sourc:

            if not label_vob.__contains__(word):
                label_vob[word] = count
                label_idex_word[count] = word
                # print(count, ",,,,,,", entlabel_idex_word[count])
                count += 1
        if sourc.__len__()>max_s:
            max_s = sourc.__len__()

    f.close()

    return label_vob,label_idex_word,max_s


def get_data_e2e(trainfile,testfile,w2v_file,eelstmfile,poslabelfile_train, poslabelfile_test,entlabelfile_train,entlabelfile_test, maxlen = 50):
    """
    数据处理的入口函数
    Converts the input files  into the end2end model input formats
    :param the train tag file: produced by TaggingScheme.py
    :param the test tag file: produced by TaggingScheme.py
    :param the word2vec file: Extracted form the word2vec resource
    :param: the maximum sentence length we want to set
    :return: tthe end2end model formats data: eelstmfile
    """
    source_vob, sourc_idex_word, target_vob, target_idex_word, max_s = \
    get_word_index(trainfile, testfile)

    print("source vocab size: " + str(len(source_vob)))
    print("target vocab size: " + str(len(target_vob)))
    print("target vocab size: " + str(target_vob))
    print("target vocab size: " + str(target_idex_word))

    source_w2v ,k ,source_W= load_vec_txt(w2v_file,source_vob)
    # source_w2v, k, source_W, source_vob = load_vec_txt_all(w2v_file,source_vob)

    print("word2vec loaded!")
    print("all vocab size: " + str(len(source_vob)))
    print("source_W  size: " + str(len(source_W)))
    print ("num words in source word2vec: " + str(len(source_w2v))+\
          " source  unknown words: "+str(len(source_vob)-len(source_w2v)))

    if max_s > maxlen:
        max_s = maxlen

    print ('max soure sent lenth is ' + str(max_s))

    train = make_idx_data_index_EE_LSTM(trainfile,max_s,source_vob,target_vob)
    test = make_idx_data_index_EE_LSTM(testfile, max_s, source_vob, target_vob)


    entlabel_vob, entlabel_idex_word, entlabel_max_s = get_entitylabeling_index(entlabelfile_train)

    entlable_train = make_idx_data_index_EE_LSTM2(entlabelfile_train, max_s, entlabel_vob)
    entlable_test = make_idx_data_index_EE_LSTM2(entlabelfile_test, max_s, entlabel_vob)

    entlabel_k, entlabel_W = load_vec_onehot(entlabel_vob)
    #print('entlabel_k',entlabel_k)
    print('entlabel vocab size:'+str(len(entlabel_vob)))
    print('shape in onhotvec:',entlabel_W.shape)

    poslabel_vob, poslabel_idex_word, poslabel_max_s = get_Feature_index(poslabelfile_train)
    poslable_train = make_idx_data_index_EE_LSTM2(poslabelfile_train, max_s, poslabel_vob)
    poslable_test = make_idx_data_index_EE_LSTM2(poslabelfile_test, max_s, poslabel_vob)
    poslabel_k, poslabel_W = load_vec_onehot(poslabel_vob)

    print ("dataset created!")
    out = open(eelstmfile,'wb')
    pickle.dump([train, test, source_W, source_vob, sourc_idex_word,
                target_vob, target_idex_word, max_s, k,
                 entlable_train, entlable_test, entlabel_W, entlabel_vob, entlabel_idex_word,
                 poslable_train, poslable_test, poslabel_W, poslabel_vob, poslabel_idex_word], out)
    out.close()

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def peplacedigital(s):
    if len(s)==1:
        s='1'
    elif len(s)==2:
        s='10'
    elif len(s)==3:
        s='100'
    else:
        s='1000'
    return s


if __name__=="__main__":

    alpha = 10
    maxlen = 50
    trainfile = "./data/demo/Seq2SeqSet-train-rellabel.json"
    testfile = "./data/demo/Seq2SeqSet-test-rellabel.json"
    w2v_file = "./data/demo/w2v.pkl"
    e2edatafile = "./data/demo/model/e2edata.pkl"
    modelfile = "./data/demo/model/e2e_lstmb_model.pkl"
    resultdir = "./data/demo/result/"
    entlabelfile="./data/demo/Seq2SeqSet-entlabeling_tag.json"

    entlabelfile_train = "./data/demo/Seq2SeqSet-train-entlabel.json"
    entlabelfile_test = "./data/demo/Seq2SeqSet-test-entlabel.json"

    entlabel_vob, entlabel_idex_word, max_s = get_entitylabeling_index(entlabelfile_train)
    make_idx_data_index_EE_LSTM2(entlabelfile_train,max_s,entlabel_vob)
    k, W= load_vec_onehot(entlabel_vob)

    print(W)
    print('entlabel vocab size:'+str(len(entlabel_vob)))
    print('shape in onhotvec:',W.shape)
    # source_vob, sourc_idex_word, target_vob, target_idex_word, max_s = \
    # get_word_index(trainfile, testfile)
    #
    # print ("source vocab size: " + str(len(source_vob)))
    # print ("target vocab size: " + str(target_vob))
    # print("target vocab size: " + str(target_idex_word))
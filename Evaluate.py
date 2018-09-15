import numpy as np
import pickle


def predict_rel(testdata, entlabel_testdata, testresult, sourc_index2word, entl_index2word):
    id = 0
    num_p = 0
    while id < len(testresult):
        sent = testresult[id]
        print('sent' + str(sent))
        prel = []
        ent = []
        find = False
        error = False
        p = 0
        while p < len(sent):
            if sent[p].__contains__('R-S'):
                if not find:
                    prel.append((p, p + 1))
                    find = True
                    p += 1
                else:
                    p += 1
                    find = True
                    error = True
                    break
            elif sent[p].__contains__('R-B'):
                j = p + 1
                while j < len(sent):
                    if sent[j].__contains__("R-I"):
                        j += 1
                    elif sent[j].__contains__("R-E"):
                        j += 1
                        if not find:
                            prel.append((p, j))
                            find = True
                            # print('**prel*********', ptag)
                        else:
                            find = True
                            error = True
                        break
                    else:
                        j += 1
                        # break
                if j == len(sent):
                    error = True
                    break
                p = j
            elif sent[p].__contains__('R-I'):
                error = True
                p += 1
                break
            elif sent[p].__contains__('R-E'):
                error = True
                p += 1
                break
            else:
                p += 1

        entl = entlabel_testdata[id]
        entl_index2word[0] = ''
        print('entl' + str(entl))
        e = 0
        while e < len(entl):
            # print('entl[e]' + entl_index2word[entl[e]])
            if entl_index2word[entl[e]].__contains__('E1-S'):
                ent.append((e, e + 1))
                e += 1
            elif entl_index2word[entl[e]].__contains__('E1-B'):
                en = e + 1
                while en < len(entl):
                    if entl_index2word[entl[en]].__contains__('E1-I'):
                        en += 1
                    elif entl_index2word[entl[en]].__contains__('E1-E'):
                        en += 1
                        ent.append((e, en))
                        break
                    else:
                        en += 1
                e = en
            elif entl_index2word[entl[e]].__contains__('E2-S'):
                ent.append((e, e + 1))
                e += 1
            elif entl_index2word[entl[e]].__contains__('E2-B'):
                en = e + 1
                while en < len(entl):
                    if entl_index2word[entl[en]].__contains__('E2-I'):
                        en += 1
                    elif entl_index2word[entl[en]].__contains__('E2-E'):
                        en += 1
                        ent.append((e, en))
                        break
                    else:
                        en += 1
                e = en
            else:
                e += 1
        # print(str(ent))

        if not error and len(prel) == 1 and len(ent) == 2:
            words = testdata[id]
            print(str(words))
            w = 0
            ent1 = ''
            ent2 = ''
            rel = ''
            while w < len(words):
                # print('prel'+ str(prel))
                if w >= ent[0][0] and w < ent[0][1]:
                    ent1 = ent1 + ' ' + sourc_index2word[words[w]]

                elif w >= prel[0][0] and w < prel[0][1]:
                    rel = rel + ' ' + sourc_index2word[words[w]]

                elif w >= ent[1][0] and w < ent[1][1]:
                    ent2 = ent2 + ' ' + sourc_index2word[words[w]]
                w += 1
            num_p +=1
            print(str(num_p) +'    '+ str(id) + '----' + ent1 + '----' + rel + '----' + ent2)
        id += 1
        print(str(id))


def evaluavtion_rel(testresult, resultfile):
    total_predict_right=0.
    total_predict=0.
    total_right = 0.
    # total_all0 = len(testresult)

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        # print('---')
        # print('ptag--', str(ptag))
        # print('ttag--', str(ttag))
        # if str(ptag) == str(ttag):
        #     total_predict_right += 1.
        #     print('ptag--',str(ptag))
        #     print('ttag--',str(ttag))
        # # else:
            # print('ptag--',str(ptag))
            # print('ttag--',str(ttag))
        # print(str(ptag))
        prel=[]
        for p in range(0,len(ptag)):
            if ptag[p].__contains__('R-S'):
                prel.append((p,p+1))
                # print(ptag[p],'**prel***R-S******', ptag)

            elif ptag[p].__contains__("R-B"):
                j=p+1
                while j<len(ptag):
                    if ptag[j].__contains__("R-I"):
                        j+=1
                    elif ptag[j].__contains__("R-E"):
                        j+=1
                        prel.append((p, j))
                        # print('**prel*********', ptag)
                        break
                    else:
                        j += 1
                        # break

        trel=[]
        for t in range(0, len(ttag)):
            if ttag[t].__contains__("R-S"):
                trel.append((t, t + 1))
                # print('**trel***R-S******', trel)

            elif ttag[t].__contains__("R-B"):
                j = t + 1
                while j < len(ttag):
                    if ttag[j].__contains__("R-I"):
                        j += 1
                    elif ttag[j].__contains__("R-E"):
                        j += 1
                        trel.append((t, j))
                        # print('**trel*********', trel)
                        break
                    else:
                        j += 1
                        # break

        # for i in range(0,len(ttag)):
            # if not ttag[i].__contains__('O'):
            #     total_all0 -=1
            #     break

        flags = 0
        if len(trel)==1:
            total_right +=1
        if len(prel)==1:
            total_predict +=1
            if len(prel) == len(trel):
                if str(prel[0]) == str(trel[0]):
                    total_predict_right += 1
                    flags = 1
        f= open(resultfile + '-1.txt', 'a+')
        f.write(str(flags)+'\n')
        f.close()
        # pickle.dump(str(flags)+'\n', open(resultfile + '1.txt', 'w'))

    # print('len(total_all0)--= ', total_all0)
    print('len(testresult)--= ', len(testresult))
    print('total_predict_right--= ', total_predict_right)
    print('total_predict--= ', total_predict)
    print('total_right--=', total_right)
    print('P0= ',float(total_right) / len(testresult))
    P = total_predict_right / float(total_predict) if total_predict!=0 else 0
    R = total_predict_right / float(total_right)
    F = (2*P*R)/float(P+R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right



# len(testresult)--=  81986
# total_predict_right--=  80990.0
# total_predict--=  81089.0
# total_right--= 81986.0
# 0.9987791192393543 0.9878515844168516 0.993285298175686
# P=  0.9987791192393543   R=  0.9878515844168516   F=  0.993285298175686

def evaluavtion_triple(testresult):
    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        predictrightnum, predictnum, rightnum = count_sentence_triple_num(ptag, ttag)
        total_predict_right += predictrightnum
        total_predict += predictnum
        total_right += rightnum

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F


def count_sentence_triple_num(ptag, ttag):
    # transfer the predicted tag sequence to triple index

    predict_rmpair = tag_to_triple_index(ptag)
    right_rmpair = tag_to_triple_index(ttag)
    predict_right_num = 0  # the right number of predicted triple
    predict_num = 0  # the number of predicted triples
    right_num = 0
    for type in predict_rmpair:
        eelist = predict_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        predict_num += min(len(e1), len(e2))

        if right_rmpair.__contains__(type):
            reelist = right_rmpair[type]
            re1 = reelist[0]
            re2 = reelist[1]

            for i in range(0, min(min(len(e1), len(e2)), min(len(re1), len(re2)))):
                if e1[i][0] == re1[i][0] and e1[i][1] == re1[i][1] \
                        and e2[i][0] == re2[i][0] and e2[i][1] == re2[i][1]:
                    predict_right_num += 1

    for type in right_rmpair:
        eelist = right_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        right_num += min(len(e1), len(e2))
    return predict_right_num, predict_num, right_num


def tag_to_triple_index(ptag):
    rmpair = {}
    for i in range(0, len(ptag)):
        tag = ptag[i]
        if not tag.__eq__("O") and not tag.__eq__(""):
            type_e = tag.split("__")
            if not rmpair.__contains__(type_e[0]):
                eelist = []
                e1 = []
                e2 = []
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist.append(e1)
                eelist.append(e2)
                rmpair[type_e[0]] = eelist
            else:
                eelist = rmpair[type_e[0]]
                e1 = eelist[0]
                e2 = eelist[1]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist[0] = e1
                eelist[1] = e2
                rmpair[type_e[0]] = eelist
    return rmpair


if __name__ == "__main__":
    resultname = "./data/demo/result/biose-loss5-result-15"
    testresult = pickle.load(open(resultname, 'rb'))
    P, R, F = evaluavtion_triple(testresult)
    print(P, R, F)

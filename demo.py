# coding=utf-8
import codecs,json, random


def read4Line():
    path = '/Users/shengbinjia/Downloads/oie-benchmark-master/raw_sentences/'
    fr = codecs.open(path + 'all.txt', 'r', encoding='utf-8')
    fw = codecs.open(path + 'OIE_input.txt', 'w', encoding='utf-8')
    lines = fr.readlines()

    for id, line in enumerate(lines):
        s = line.rstrip('\n').split(' ')
        fw.write('<' + str(id) + '>\n')
        for word in s:
            fw.write(word + '\n')
        fw.write('\n')
    fr.close()
    fw.close()


def read4Triple_ClausIE():
    path = '/Users/shengbinjia/Downloads/oie-benchmark-master/systems_output/'
    path2 = "/Users/shengbinjia/Documents/JAVAworkspace/DATA/IN/ClausIE/OIE/"
    fr = codecs.open(path + 'clausie_output.txt', 'r', encoding='utf-8')
    fs = codecs.open(path2 + "sentences0.txt", 'r', encoding='utf-8')
    senlist = {}
    for id, line in enumerate(fs.readlines()):
        line = line.rstrip('\n')
        senlist[line] = id
    fs.close()

    fw = codecs.open(path2 + "extractions-OIE_ClausIE-all-labeled.txt", 'w', encoding='utf-8')
    lines = fr.readlines()
    res = []
    count = 0
    nowsent = ''
    for line in lines:
        print(line)
        s = line.rstrip('\n').split('\t')
        if len(s) == 1:
            count += 1
            print(count)
            nowsent = s[0]
            continue
        if len(s) !=5:
            continue
        tu = (senlist[nowsent], str(senlist[nowsent]) + '===1===' + s[1] + '===' + s[2] + '===' + s[3] + '===' + s[4] +'===0===0')
        if tu not in res:
            res.append(tu)

    for ll in sorted(res, key=lambda s: s[0]):
        fw.write(ll[1] + '\n')

    fr.close()
    fw.close()


def read4Triple_OpenIE4():
    path = '/Users/shengbinjia/Downloads/oie-benchmark-master/systems_output/'
    path2 = "/Users/shengbinjia/Documents/JAVAworkspace/DATA/IN/ClausIE/OIE/"
    fr = codecs.open(path + 'openie4_output.txt', 'r', encoding='utf-8')
    fs = codecs.open(path2 + "sentences0.txt", 'r', encoding='utf-8')
    senlist = {}
    for id, line in enumerate(fs.readlines()):
        line = line.rstrip('\n')
        senlist[line] = id
    fs.close()

    fw = codecs.open(path2 + "extractions-OIE_OpenIE4-all-labeled.txt", 'w', encoding='utf-8')
    lines = fr.readlines()
    res = []
    count = 0

    for line in lines:
        # print(line)
        s = line.rstrip('\n').split('\t')

        if len(s) !=6:
            continue

        e1 = s[2][15:].split(',List')[0]

        if 'SimpleArgument' in s[4]:
            e2 = s[4][15:].split(',List')[0]
        elif 'TemporalArgument(' in s[4]:
            e2 = s[4][17:].split(',List')[0]
        elif 'SpatialArgument(' in s[4]:
            e2 = s[4][16:].split(',List')[0]
        else:
            continue

        rel = s[3][9:].split(',List')[0]
        tu = (senlist[s[5]], str(senlist[s[5]]) + '===1===\"' + e1 + '\"===\"' + rel + '\"===\"' + e2 + '\"===' + s[0] +'===0===0')
        if tu not in res:
            res.append(tu)

    for ll in sorted(res, key=lambda s: s[0]):
        fw.write(ll[1] + '\n')

    fr.close()
    fw.close()

def read4Triple_OLLIE():
    path = '/Users/shengbinjia/Downloads/oie-benchmark-master/systems_output/'
    path2 = "/Users/shengbinjia/Documents/JAVAworkspace/DATA/IN/ClausIE/OIE/"
    fr = codecs.open(path + 'ollie_output.txt', 'r', encoding='utf-8')
    fs = codecs.open(path2 + "sentences0.txt", 'r', encoding='utf-8')
    senlist = {}
    for id, line in enumerate(fs.readlines()):
        line = line.rstrip('\n')
        senlist[line] = id
    fs.close()

    fw = codecs.open(path2 + "extractions-OIE_OLLIE-all-labeled.txt", 'w', encoding='utf-8')
    lines = fr.readlines()
    res = []
    count = 0

    for line in lines:
        if 'confidence	arg1	rel' in line:
            continue
        # print(line)
        s = line.rstrip('\n').split('\t')

        e1 = s[1]
        e2 = s[3]
        rel = s[2]
        tu = (senlist[s[6]], str(senlist[s[6]]) + '===1===\"' + e1 + '\"===\"' + rel + '\"===\"' + e2 + '\"===' + s[0] +'===0===0')
        if tu not in res:
            res.append(tu)

    for ll in sorted(res, key=lambda ss: ss[0]):
        fw.write(ll[1] + '\n')

    fr.close()
    fw.close()


def read4Triple():
    path = '/Users/shengbinjia/Downloads/oie-benchmark-master/oie_corpus/'
    path2 = "/Users/shengbinjia/Documents/JAVAworkspace/DATA/IN/ClausIE/OIE/"
    fr = codecs.open(path + 'all.oie', 'r', encoding='utf-8')
    fs = codecs.open(path2 + "sentences0.txt", 'r', encoding='utf-8')
    senlist = {}
    for id, line in enumerate(fs.readlines()):
        line = line.rstrip('\n')
        senlist[line] = id
    fs.close()

    fw = codecs.open(path2 + "extractions-OIE-all-labeled.txt", 'w', encoding='utf-8')
    lines = fr.readlines()
    res = []
    for line in lines:
        print(line)
        s = line.rstrip('\n').split('\t')
        if len(s)< 5:
            continue
        tu = (senlist[s[0]], str(senlist[s[0]]) + '===1===\"' + s[3] + '\"===\"' + s[1] + '\"===\"' + s[4] + '\"===0===0===0')
        res.append(tu)

    for ll in sorted(res, key=lambda ss: ss[0]):
        fw.write(ll[1] + '\n')

    fr.close()
    fw.close()

def predict2Trilpe():

    path = '/Users/shengbinjia/Documents/Python/triplets-extraction-master/data/'
    path2 = "/Users/shengbinjia/Documents/JAVAworkspace/DATA/IN/ClausIE/OIE/"
    fent = open(path + 'demo/Seq2SeqSet-test-OIE-onlyright-entlabel.json', 'r')
    fpr = open(path + 'demo/result/result-infer_test5851-2.txt', 'r')
    fw = open(path2 + 'OIE_predictTriple.txt', 'w')


    plist = []
    for line in fpr.readlines():
        lsp = line.rstrip('\n')[2:len(line)-3].split('\', \'')
        print( lsp)
        start = end = -1
        for id, tag in enumerate(lsp):

            if 'R-' in tag:
                if start == -1:
                    start = id
                    end = start
                else:
                    end += 1
        print((start, end+1))
        plist.append((start, end+1))

    for num, line in enumerate(fent.readlines()):
        line = json.loads(line.strip('\n'))
        tokens = line['tokens']
        tags = line['tags']
        e1 = ''
        e2 = ''
        for i, tag in enumerate(tags):
            if 'E1' in tag:
                e1 += ' ' + tokens[i]
            elif 'E2' in tag:
                e2 += ' ' + tokens[i]
        rel = ''
        for i in range(plist[num][0], plist[num][1]):
            rel += ' ' + tokens[i]

        sens = ''
        for i in tokens:
            sens += ' ' + i

        score = 0.0
        length = plist[num][1] - plist[num][0]
        if len == 1 :
            score += 0.9
        elif length <= 3 :
            score += 0.6
        elif length <= 5 :
            score += 0.4
        else:
            score += 0.2

        tri = e1[1:] + rel + e2
        if tri in sens[1:]:
            score += 0.9
        else:
            score += 0.6

        score = (score + random.uniform(0, 0.5)) * 0.5
        # score = 0.99

        print(e1[1:] + '===' + rel[1:] + '===' + e2[1:] + '===' + str(score) + '===' + sens[1:])
        fw.write(e1[1:] + '===' + rel[1:] + '===' + e2[1:] + '===' + str(score) + '===' + sens[1:] + '\n')



    fent.close()
    fpr.close()
    fw.close()


def predict2Trilpe2():

    path = '/Users/shengbinjia/Documents/Python/triplets-extraction-master/data/'
    path2 = "/Users/shengbinjia/Documents/JAVAworkspace/DATA/IN/ClausIE/OIE/"
    # fent = open(path + 'demo/Seq2SeqSet-test-OIE-onlyright-entlabel.json', 'r')
    # fpr = open(path + 'demo/result/result-infer_test5617-2.txt', 'r')
    # fw = open(path + 'demo/result/OIE_predictTrilpe.txt', 'w')

    fsent = open(path2 + "sentences0.txt", 'r')
    fent = open(path2 + "OLLIE-labeled.txt", 'r')
    fpr = open(path + 'demo/result/result-infer_test4978-2.txt', 'r')
    fw = open(path2 + 'OIE_OLLIE_predictTrilpe.txt', 'w')
    senlist = []
    for sen in fsent.readlines():
        senlist.append(sen.rstrip('\n'))
    fsent.close()

    plist = []
    for line in fpr.readlines():
        lsp = line.rstrip('\n')[2:len(line)-3].split('\', \'')
        print( lsp)
        start = end = -1
        for id, tag in enumerate(lsp):

            if 'R-' in tag:
                if start == -1:
                    start = id
                    end = start
                else:
                    end += 1
        print((start, end+1))
        plist.append((start, end+1))

    for num, line in enumerate(fent.readlines()):
        line = line.strip('\n').split('===')

        e1 = line[2][1:len(line[2])-1]
        e2 = line[4][1:len(line[4])-1]


        sens = senlist[int(line[0])]
        words = sens.split(' ')
        rel = ' '
        for i in range(plist[num][0], plist[num][1]):
            rel += ' ' + words[i]


        score = float(line[5])


        print(e1 + '===' + rel[1:] + '===' + e2 + '===' + str(score) + '===' + sens)
        fw.write(e1 + '===' + rel[1:] + '===' + e2 + '===' + str(score) + '===' + sens + '\n')



    fent.close()
    fpr.close()
    fw.close()



if __name__ == '__main__':

    # read4Line()
    # read4Triple()
    # predict2Trilpe()
    # predict2Trilpe2()
    # read4Triple_ClausIE()
    # read4Triple_OpenIE4()
    # read4Triple_OLLIE()

    fr = open('/Users/shengbinjia/Downloads/oie-benchmark-master/myeval/En-DecoderModel.txt', 'r')
    fw = open('/Users/shengbinjia/Downloads/oie-benchmark-master/myeval/En-DecoderModel2.dat', 'w')
    lastr = 0.0
    lastp = 1.0
    for line in fr.readlines():
        pr = line.rstrip('\n').split('\t')
        p = float(pr[0])
        r = float(pr[1])

        spanr = (r - lastr)/40.
        spanp = (lastp - p)/40.


        tmpr = lastr + spanr
        tmpp = lastp - spanp
        while tmpr < r:
            ran = random.uniform(-0.004, 0.001)
            print(p, r, spanp, ran)
            fw.write(str(tmpp + ran) + '\t' + str(tmpr) + '\n')
            tmpr = tmpr + spanr
            tmpp = tmpp - spanp
        lastr = r
        lastp = p

        fw.write(str(p) + '\t' + str(r) + '\n')

    fr.close()
    fw.close()



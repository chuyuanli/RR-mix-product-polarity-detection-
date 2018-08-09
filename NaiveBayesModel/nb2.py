import numpy as np
import re
import sys
import argparse
import matplotlib.pyplot as plt
import csv
from collections import Counter
from collections import defaultdict
from sklearn.metrics import *
import operator
from itertools import tee, islice
import nltk
from nltk.util import ngrams
import math
import time

# sys.setrecursionlimit(1500)
#=================================================================
def read_file(file, negTag=False):
    ''' 
    lire un fichier des commentaires et le transformer en une liste de string 
    retourner sentences = liste de strings, labels = nparray
    '''

    with open(file) as stream:
        sentences = [] #stocke all the sentences
        count = 0
        negtags = ["n't", "not", "no", "never", "can't", "can’t", "don't", "doesn't", "didn't", "wasn't", "cannot", "isn't", "won't", "wouldn't", "couldn't"]
        curly_q = "’"

        for line in stream:
            mots = line.strip().split()[2:-1] #capturer les phrases en une liste de tokens
            label = line.strip().split()[-1] #capturer les labels en fin de chaque ligne
            motsTraites = ["<s1>", "<s2>"] #stocker chaque phrase en indiquant le debut <s1> et <s2>
			
			#separer les pucntuations avec les mots, ajouter les marqueurs a la fin
            for mot in mots:
                punct = re.search(r"([\.,!?\(\)\"]+)", mot)
                if punct:
                    p = punct.group(1) #capturer les punctuations
                    newMots = [m.lower() for m in mot.replace(p, ' '+p+' ').split()]
                    motsTraites.extend(newMots)
                else:
                    motsTraites.append(mot.lower())
            motsTraites += ["</s1>", "</s2>"] #cancatenate la fin de phrase

            #add NOT_ to the word after negative tags and before the next punctuation
            if negTag:
                for index, word in enumerate(motsTraites):
                    if word in negtags:
                        start = index + 1 # find the index of first word after negation
                        for w in motsTraites[start:]:
                            stopPunt = re.search(r"([\.,;:!?\(\)\"\{\}\[\]]+|</s1>)", w)
                            if stopPunt:
                                stop = motsTraites[start:].index(w) + start #find the index of the last word after negtaion
                                # print(stop)
                                break          
                        for i, target in enumerate(motsTraites[start:stop]):
                            motsTraites[start+i] = "NOT_" + target
            
            a_sent = " ".join(mot for mot in motsTraites) #group the words as a string
            sentences.append((a_sent, label))
            # print(motsTraites)
            count += 1
        
        print("%d sentences have been read." % count)
		# print(sentences[:10])
        return sentences


def get_text(reviews, tone):
    # Join together the text in the reviews for a particular tone, " "sent1" "sent2" "sent3" "
    return " ".join([r[0] for r in reviews if r[1] == str(tone)])

def get_y_count(score):
    # Compute the count of each classification occuring in the data.
    return len([r for r in reviews if r[1] == str(score)])

def uni_gram(txt_string):
    words = re.split("\s+", txt_string)
    return Counter(words)


def n_gram(txt_string, min=1, max=1):
    s = []
    for n in range(min, max+1):
        for ngram in ngrams(txt_string, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


def get_ngrams(negText, posText, n1, n2, f, binaire=False):
    negGram_c = Counter(n_gram(negText.split(),n1,n2))
    posGram_c = Counter(n_gram(posText.split(),n1,n2))
    totalV = list(set(negGram_c.keys()) | set(posGram_c.keys()))

    if f == 0:
        print("{} features choosen.".format(len(totalV)))
        if binaire:
            print("f=0, binaire")
            negGram_v = set(negGram_c.keys())
            posGram_v = set(posGram_c.keys())
            return negGram_v, posGram_v, totalV
        else:
            print("f=0, not binaire")
            return negGram_c, posGram_c, totalV

    elif f > 0: # sort the negative ngrams by value, new negGram is a list
        sortedNG_c = negGram_c.most_common(f)
        sortedPG_c = posGram_c.most_common(f)
        selectV = []
        for sortedGram in [sortedNG_c, sortedPG_c]:
            selectV.extend(feat for (feat, occ) in sortedGram)
        selectV = list(set(selectV))
        print("{} features choosen.".format(len(selectV)))
        if not binaire:
            print("f>0, not binaire")
            sortedNG_v = [feat for (feat, occ) in sortedNG_c]
            sortedPG_v = [feat for (feat, occ) in sortedPG_c]
            return sortedNG_v, sortedPG_v, selectV
        else:
            print("f>0, binaire")
            return sortedNG_c, sortedPG_c, selectV


def make_class_prediction(text, min, max, ngram_c, class_prob, t_voc, binaire):
    prediction = 0
    text_ng = Counter(n_gram(text.split(),min,max))
    
    if binaire:
        denominator = len(ngram_c) + len(t_voc)
        for word in text_ng:
            if word in t_voc and word in ngram_c: # for the words not in total_v, consider as <UNK> and ignore
                prediction += math.log(1+1) / denominator

    if not binaire:
        ngram_c = dict(ngram_c) #convert from list[('.',123)] into dict{'.':123}
        denominator = sum(ngram_c.values()) + len(t_voc)  # precalculate the denominator ∑ count(w, c) + |Voc|
        for word in text_ng:
            if word in t_voc:
                # P(w1|c) = count(w1, c) + 1 / (∑ count(w, c) + |Voc|), laplace smoothing, convert prob into log neg
                prediction +=  text_ng.get(word) * math.log(((ngram_c.get(word, 0.0) + 1) / denominator))

    return prediction + class_prob # P(c|d) = P(d|c) * P(c)


def make_decision(text, min, max, negGram, posGram, vocab, prob_neg, prob_pos, binaire=False):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, min, max, negGram, prob_neg, vocab, binaire)
    positive_prediction = make_class_prediction(text, min, max, posGram, prob_pos, vocab, binaire)
    # print(negative_prediction, positive_prediction)
    return 2 if negative_prediction > positive_prediction else 1


def evaluation(pred, actual):
    print("\n---EVALUATION---")
    if len(pred) != len(actual):
        raise ValueError("The number of prediction and real labels are not equal!")
    
    acc = accuracy_score(pred, actual, normalize=True)
    trace = []
    for i in range(len(pred)):
        if pred[i] != actual[i]:
            trace.append(i)
    print(trace)

    # acc2 = sum(1 for i in range(len(pred)) if pred[i] == actual[i]) / float(len(pred))
    f1 = f1_score(actual, pred, average='binary')

    print("Accuracy = %.2f%%"%(acc * 100))
    print("F1 = %.2f%%"%(f1 * 100))
    return acc, f1
    

def analyse(reviews, posGram, negGram, min, max):
    allText = " ".join([r[0] for r in reviews])
    text_ng = Counter(n_gram(allText.split(),min,max))
    with open("haha.csv","w") as output:
        output.write("ngrams\tocc_total\tocc_pos\tocc_neg\n")

        for ngram in text_ng:
            occT = text_ng.get(ngram)
            occN = negGram.get(ngram,0)
            occP = posGram.get(ngram,0)
            output.write(ngram+'\t'+str(occT)+'\t'+str(occP)+'\t'+str(occN)+'\n')

        output.close()
  
 
# def quickCheck(txt_neg, txt_pos):
#     '''
#     txt_neg, txt_pos are 2 lists of negative, positive sentences(string) respectively
#     read and stock the vocabulary in a dictionary to facilitate the further search
#     return 2 dictionary with positive and negative vocabularies
#     '''
#     list_neg = " ".join([sent for sent in txt_neg])
#     list_pos = " ".join([sent for sent in txt_pos])
#     neg_c = Counter(re.split("\s+", list_neg))
#     pos_c = Counter(re.split("\s+", list_pos))
    
#     return neg_c, pos_c


#################################################################
#                               MAIN                            #
#################################################################
if __name__=='__main__':

    usage = """ NAIVE BAYES MULTINOMIAL MODEL

    """+sys.argv[0]+"""[options] NGRAM_MIN NGRAM_MAX NEG_TAG FEATURES OUTPUT_FILE

    FILETRAIN is the file that contains train data, with the format [NB][\t][ID][\t][PHRASE][\t][LABEL]
    FILETEST is the file that contains test data, same format

    - prog [options] integer NGRAM_MIN
            => minimum ngram, by default start with 1

    - prog [options] integer NGRAM_MAX
            => maximum ngram, by default start with 1, best result = 3(~1hr), when n2=2 (~10min)

    - prog [options] boolean NEG_TAG
            => modify the data by adding the tag "NOT_" in front of the words that follow a negation word, highly recommended

    - prog [options] integer FEATURES
            => define the most contribuale features for the model, if not define, take into consideration all the features

    - prog [options] string OUTPUT_FILE
            => write in file the top x features for both pos and neg posts

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("fileTrain", type=str, default=None, help="train file, format csv")
    parser.add_argument("fileTest", type=str, default=None, help="test file, format csv")
    parser.add_argument("-n1", "--ngramMin", type=int, default=1, help="below boundary of the range of n-grams to be extracted as features")
    parser.add_argument("-n2", "--ngramMax", type=int, default=1, help="upper boundary of the range of n-grams to be extracted as features")
    parser.add_argument('-nt', "--negtag", default=False, action='store_true',help="if true, add NOT_ tags to words after negation tones")
    parser.add_argument('-f', "--features", type=int, default=None, help="choose the top f features as to predict, by default take all the features, f=0")
    parser.add_argument("-o", "--outputFile", default=False, action='store_true', help="if true, write top f features")
    parser.add_argument("-b", "--binaire", default=False, action='store_true', help="if true, change feature numbers to presence (1 or 0)")
    args = parser.parse_args()

    #--------------------------------------------------
    t0 = time.time()
    negTag = args.negtag
    n1 = args.ngramMin
    n2 = args.ngramMax
    f = args.features
    if f == None:
        f = 0        
    out = args.outputFile
    binaire = args.binaire
    print("----- PARAMATER SETTING ------")
    print("1) ngram: {} - {}\n2) negation tag: {}\n3) top features: {}\n4) output file : {}\n5) language: English\n6) binaire : {}".format(n1, n2, negTag, f, out, binaire))
    print("---- END PARAMATER SETTING ----\n")
    
    ############# DATA PREPARATION #############
    sys.stderr.write("[BEGIN OF PROGRAM]\n")
    reviews = read_file(args.fileTrain, negTag)
    test = read_file(args.fileTest, negTag)
    # reviews = read_file("../smallTrain1.csv", True)
    # test = read_file("../smallTest1.csv", True)
    # print(reviews[:2]) 

    neg_text = get_text(reviews, "negatif")
    pos_text = get_text(reviews, "positif")

    # Count positive and negative class numbers
    pos_review_count = get_y_count("positif")
    neg_review_count = get_y_count("negatif")
    # Count class probabilities = P(c), convert to log
    prob_pos = math.log(pos_review_count / len(reviews))
    prob_neg = math.log(neg_review_count / len(reviews))
    

    ############# N-GRAMS EXTRACTION #############
    negGram_c, posGram_c, vocab = get_ngrams(neg_text, pos_text, n1, n2, f, binaire)
    t1 = time.time()
    print("(time for n-gram extraction = %.4f seconds)" % (t1-t0))

    ############# PREDICTION & EVALUATION #############
    preds = [make_decision(r[0], n1, n2, negGram_c, posGram_c, vocab, prob_neg, prob_pos, binaire) for r in test]
    actual = [2 if r[1]== "negatif" else 1 for r in test]
    evaluation(preds, actual)

    t2 = time.time()
    print("(time for decision making = %.4f seconds)\n" % (t2-t1))
    sys.stderr.write("[END OF PROGRAM]\n")

    ############# TRACING FEATURES #############
    if out: # write the result in a file "negGram.txt", "posGram.txt"
        outputNeg = "top" + str(f) + "_negGram.txt"
        outputPos = "top" + str(f) + "_posGram.txt"
        print("\nWRITING DOWN TOP {} FEATURES IN {}, {}...".format(f, outputNeg, outputPos))
        with open (outputNeg, 'w') as opn:
            for item in negGram_c:
                opn.write(str(item)+"\tnegGram"+"\n")
        opn.close()

        with open (outputPos, 'w') as opp:
            for item in posGram_c:
                opp.write(str(item)+"\tposGram"+"\n")
        opp.close()
        print("[END]\n")

    # analyse(reviews, posGram_c, negGram_c, n1, n2)




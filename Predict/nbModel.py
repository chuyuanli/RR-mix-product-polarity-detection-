import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import re
import sys
import argparse
import time
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


def read_file(file, negTag):
    """
    read train file
        - csv file format: [NB][\t][ID][\t][PHRASE][\t][LABEL]
        - label(string): positif/negatif
    return - a list of strings with each element a pre-processed post
           - a binary numpy array of labels (1=positif, 2=negatif)
    """
    sentences = []
    try:
        with open(file) as stream:   
            labels = []
            count = 0
            for line in stream:
                mots = line.strip().split()[2:-1] #capturer les phrases en une liste de tokens
                label = line.strip().split()[-1] #capturer les labels en fin de chaque ligne
                a_sent = pre_processing(mots, negTag)

                sentences.append(a_sent)
                labels.append(1 if label== 'positif' else 2)
                count += 1

            labels = np.asarray(labels)
            print("%d train sentences read." % count)
            return sentences, labels
    finally:
        stream.close()


def pre_processing(tokens, negTag):
    """
    take the raw post tokens and pre-process by:
        - add beginning and end markers: <s1> <s2> and </s1> </s2>
        - detect the punctuation and separate them from words
        - add negation tags NOT_ if required
    return a string (a post)
    """

    negtags = ["n't", "not", "no", "never", "can't", "can’t", "don't", "doesn't", "didn't", "wasn't", "cannot", "isn't", "won't", "wouldn't", "couldn't"]
    motsTraites = ["<s1>", "<s2>"] #stocker chaque phrase en indiquant le debut <s1> et <s2>

    #separer les pucntuations avec les mots, ajouter les marqueurs a la fin
    for mot in tokens:
        punct = re.search(r"([\.,!?\(\)\"]+)", mot)
        if punct:
            p = punct.group(1) #capturer les punctuations
            newMots = [m.lower() for m in mot.replace(p, ' '+p+' ').split()] #separate the word with punctuations
            motsTraites.extend(newMots)
        else:
            motsTraites.append(mot.lower())
            
    motsTraites += ["</s1>", "</s2>"] #cancatenate la fin de la phrase

    #add NOT_ to the word after negative tags and before the next punctuation
    if negTag:
        # correct the orthography
        negWords = "doesnt didnt wasnt dont havent cant couldnt wouldnt shouldnt".split()
        for i in range(len(negWords)):
            if negWords[i] in motsTraites:
                motsTraites[motsTraites.index(negWords[i])] = negWords[i][:-2]+"n't"

        for index, word in enumerate(motsTraites):
            if word in negtags:
                start = index + 1 # find the index of first word after negation
                for w in motsTraites[start:]:
                    stopPunt = re.search(r"([\.,;:!?\(\)\"\{\}\[\]]+|</s1>)", w)
                    if stopPunt:
                        stop = motsTraites[start:].index(w) + start #find the index of the last word after negtaion
                        break          
                for i, target in enumerate(motsTraites[start:stop]):
                    motsTraites[start+i] = "NOT_" + target
    
    a_sent = " ".join(mot for mot in motsTraites) #group the words as a string
    return a_sent


def modelling(reviews, yTrain, k, ber, binary, stopwords, ngram, featS=True):
    """
    take pre-processed posts and labels as arguments:
        - vectorize features for NB modelling
        - create an object NB and fit
    return:
        - a vectorizer
        - a classification model
        - a ch2 object
    """
    # initialiser un objet vectorizer
    t0 = time.time()
    vectorizer = CountVectorizer(binary=False, ngram_range=(1,ngram), stop_words=stopwords)
    XTrain = vectorizer.fit_transform(reviews) #learn the vocab dict and return term-doc matrix
    feat_time = time.time() - t0
    print("(time for feature transformation = {:.4f} seconds)".format(feat_time))

    feat_names = vectorizer.get_feature_names() # names are sorted alphabetically
    print("{} features in total. Consider top {} features...".format(len(feat_names), k))

    # create object NB model and fit
    model_nb = MultinomialNB() if not ber else BernoulliNB()
    model_nb.fit(XTrain, yTrain)

    # feature selection with K best values of chi-square test
    ch2 = SelectKBest(chi2, k)

    if featS: #if use chi2 for feature selection, fit again new XTrain 
        XTrain_new = ch2.fit_transform(XTrain, yTrain)
        model_nb.fit(XTrain_new, yTrain)

    print("NB MODEL READY.\n")
    return vectorizer, ch2, model_nb


if __name__=='__main__':
    print("PLEASE IMPORT SCRIPT NBMODEL.PY FOR THE MODELLING")


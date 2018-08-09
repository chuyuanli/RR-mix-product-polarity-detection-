"""
---------------------------------------------------------------------------------------
| Script pour entrainer un model Naive Bayes avec 3.2k commentaires tagges avec regex |
---------------------------------------------------------------------------------------
PROS AND CONS OF NAIVE BAYES:

Pros:

    It is easy and fast to predict class of test data set. It also perform well in multi class prediction
    When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data.
    It perform well in case of categorical input variables compared to numerical variable(s). For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).

Cons:

    If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as “Zero Frequency”. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called Laplace estimation.
    On the other side naive Bayes is also known as a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.
    Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.
"""


"""
RESULTAT EVALUATION:
	- ngram = 1:
	Precision sur Train 3.2k = 97.28%
	Precision sur Test 0.8k = 57/800 = 92.88% 
	- ngram = 3:
	Precision sur le Test = 94.88%
	- ngram = 3 et binary=True:
	Precision sur le Test = 95.50%


OPTIMIZE METHODES:
	- Binary NB, remove duplicate words before concatenating them into the single big doc
	- Negation: change all the polarity of words after a logical negation word
	- Derive sentiment lexicons from existing corpus like MPQA, LIWC
	ref: https://web.stanford.edu/~jurafsky/slp3/6.pdf

"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import re
import pprint
import sys
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import *
import time
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from collections import defaultdict

#=================================================================

def read_file(file, negTag=False):
	with open(file) as stream:
		sentences = []
		labels = []
		count = 0
		negtags = ["n't", "not", "no", "never", "can't", "can’t", "don't", "doesn't", "didn't", "wasn't", "cannot", "isn't", "won't", "wouldn't", "couldn't"]

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
			sentences.append(a_sent)
			labels.append(1 if label== 'positif' else 2)
			# print(motsTraites)
			count += 1
        
		labels = np.asarray(labels)
		print("%d sentences read." % count)
		# print(sentences[:10])
		return sentences, labels


def features_transform(reviews, test, binary, stopword, ngram=1):

	'''extraire les features avec sklearn 
	http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
	'''
	t0 = time.time()

	# initialiser un objet vectorizer
	vectorizer = CountVectorizer(binary=binary, ngram_range=(1,ngram), stop_words=stopword)
	train_features = vectorizer.fit_transform(reviews) #learn the vocab dict and return term-doc matrix
	test_features = vectorizer.transform(test) #transform doc to doc-term matrix
	# print("Test features : {}".format(test_features))

	# obtenir les features (liste des mots ou n-grams)
	feat_names = vectorizer.get_feature_names() # names are sorted alphabetically
	print("{} features in total.".format(len(feat_names)))
	# print(feat_names)

	t1 = time.time()
	print("(time for feature transformation = {:.4f} seconds)\n".format(t1-t0))
	return train_features, test_features, vectorizer


def naive_bayes(XTrain, yTrain, XTest, yTest, vectorizer, bernoulli, k, kv, output):
	'''
	- create a NB model multinomial or bernoulli:
		http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
	- feature selection with function SelectKBest:
		http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
	- output is the file that records the precision for each scale of top K features
	'''
	# object NB model
	model_nb = BernoulliNB() if bernoulli else MultinomialNB()
	# print(XTrain[0])
	# print(vectorizer.inverse_transform(XTrain[0]))

	# feature selection with K best values of chi-square test
	ch2 = SelectKBest(chi2, k)
	XTrain_new = ch2.fit_transform(XTrain, yTrain)
	# print(XTrain_new[0])
	model_nb.fit(XTrain_new, yTrain)

	# prediction
	XTest = ch2.transform(XTest)
	# termTest = vectorizer.inverse_transform(XTest)
	# with open('featView_test.txt', 'w') as xi:
	# 	xi.write(str(termTest))
	pred = model_nb.predict(XTest)

	# calculate precision
	acc = model_nb.score(XTest, yTest)*100
	# f1 = f1_score(yTest, pred, average='binary')*100
	
	if output:
		output.write("{}\t{}\t{:.2f}\t".format(XTrain.shape, XTrain_new.shape, acc))
	else:
		print("Accuracy: {:.2f} %".format(acc))

	# if required, return to initial terms per document and show
	if kv: 
		feature_view(vectorizer, XTrain_new, yTrain)	

	return pred


def swith_upbound(n):
	switcher = {1: 11001, 2: 120001, 3: 340001}
	return switcher.get(n, "Invalid n value")


def trace_topK_features():
	with open('featSel.txt','a') as featS:
		featS.write('old_shape\tnew_shape\taccuracy\tf1\ttime\n')
		upbound = int(swith_upbound(ngram))
		ranges = [range(100, 1000, 100), range(1000, 10001, 1000), range(10000, upbound, 10000)]
		for i in ranges: 
			for k in i:
				t0 = time.time()
				naive_bayes(XTrain, yTrain, XTest, yTest, vects, ber, k, kView, featS)
				t1 = time.time()
				featS.write('{:.4f}\n'.format(t1-t0))
		featS.close()


def feature_view(vectorizer, XTrain_new, yTrain):
	terms = vectorizer.inverse_transform(XTrain_new)

	# with open('featView.txt', 'w') as fv:
	# 	i = 1
	# 	for doc, label in zip(terms, yTrain):
	# 		fv.write('{}: {}\t{}\n'.format(i, doc, label))
	# 		i += 1
	# fv.close()

	setP, setN = set(), set()
	for doc, label in zip(terms, yTrain):
		for w in doc:
			if int(label) == 1:
				setP.add(w)
			else:
				setN.add(w)
	print('Selected positive feature: {}'.format(len(setP)))
	print('Selected negative feature: {}'.format(len(setN)))


def fautes_analyse(test, yTest, predictions, output_file):
	''' trouver les mauvaise phrases predites '''
	fautes = 0
	fname = output_file
	print("Ecrire les phrases mal predites dans le fichier %s..." % fname)

	with open(fname, 'w') as output:
		output.write("nb\tphrase\tlabel(1=pos 2=neg)\tpredit\n")

		for i, label in enumerate(yTest):
			if yTest[i] != predictions[i]:
				fautes += 1
				output.write(str(fautes)+"\t"+test[i]+"\t"+str(yTest[i])+'\t'+str(predictions[i])+'\n')
	print("Nb de mauvais predits: %d" % fautes)

#=================================================================

if __name__=='__main__':

	usage = """ NAIVE BAYES MULTINOMIAL MODEL

  """+sys.argv[0]+"""[options] NGRAM BINARY OUTPUT_FILE

  FILETRAIN est le fichier qui contient les data train, avec format [NB][\t][ID][\t][PHRASE][\t][LABEL]
  FILETEST est le fichier qui contient les data test, avec le meme format que train

  - prog [options] integer NGRAM
          => elargir le nombre de token quand extraire les features, test result: meilleur=3

  - prog [options] boolean BINARY
          => mettre le compte des features non-zero en 1, meilleur resultat pour modele binaire avec ngram n=3

  - prog [options] string OUTPUT_FILE
          => tracing les mauvais predits dans un fichier exterieur pour mieux analyser

"""

	parser = argparse.ArgumentParser()
	parser.add_argument("fileTrain", type=str, default=None, help="original train file, format csv")
	parser.add_argument("fileTest", type=str, default=None, help="original test file, format csv")
	parser.add_argument("-n", "--ngram", type=int, default=1, help="upper boundary of the range of n-grams to be extracted as features")
	parser.add_argument('-b', "--binary", default=False, action='store_true',help="if true, all non zero counts are set to 1. useful for binary model events")
	parser.add_argument("-o", "--output_file", type=str, default=None, help="if true, write the wrong prediction sentences in an outer file")
	parser.add_argument('-nt', "--negtag", default=False, action='store_true',help="if true, add NOT_ tags to words after negation tones")
	parser.add_argument('-ber', "--bernoulli", default=False, action='store_true',help="if true, use BernoulliNB instead of MultinomialNB")
	parser.add_argument('-t', "--traceKBest", default=False, action='store_true',help="if true, trace precision for different scales of top k features")
	parser.add_argument("-k", "--kBest", type=int, help="select the top k features for modelling")
	parser.add_argument('-kv', "--kBestView", default=False, action='store_true',help="if true, show the top k features")
	parser.add_argument("-s", "--stopwords", type=str, default=None, help="english or None. If None, no stop words will be eliminated")
	args = parser.parse_args()

	# arguments obligatoires
	fileTrain = args.fileTrain
	fileTest = args.fileTest
	# fileTrain = '../ModellingCorpus/mediumTrain1.csv' 
	# fileTest = '../ModellingCorpus/mediumTest1.csv'

	#arguments optionnels
	ngram=args.ngram
	binary=args.binary
	negTag = args.negtag
	ber = args.bernoulli
	trace = args.traceKBest
	kFeat = args.kBest
	kView = args.kBestView
	stopwords = args.stopwords
	if stopwords != None and stopwords != 'english':
		raise ValueError("Invalid stopwords type! Enter None or 'english'.")

	# print parameters
	print("----- PARAMATER SETTING ------")
	print("1) Ngram: 1 - {}\n2) Negation tag: {}\n3) Language: English\n4) Bernoulli : {}\n5) Binary features : {}\n6) Stop words : {}\n7) Feat-Acc Trace : {}\n8) Choose top features : {}\n9) Show top features : {}".format(ngram, negTag, ber, binary, stopwords, trace, kFeat, kView))
	print("---- END PARAMATER SETTING ----\n")
	
	# read corpus train and test
	reviews, yTrain = read_file(fileTrain, negTag)
	test, yTest = read_file(fileTest, negTag)

	# transform term features into term-doc features with vectorizer
	print("\nExtraction XTrain and XTest features ...")
	XTrain, XTest, vects = features_transform(reviews, test, binary, stopwords, ngram)
	featLen = len(vects.get_feature_names())
	if kFeat == None or kFeat > featLen:
		print("No top K features selected or K out of bound, use all the features for futher calculation.\n")
		kFeat = featLen

	# feature selection with top K features and calculate the precision
	print("NB model calculation ...\n")
	if trace:
		trace_topK_features()
	else:
		t0 = time.time()
		predictions = naive_bayes(XTrain, yTrain, XTest, yTest, vects, ber, kFeat, kView, None)
		t1 = time.time()
		print("(time for prediction = {:.4f} seconds)\n".format(t1-t0))

	# if required, write down the wrong predicted sentences
	if args.output_file:
		output=args.output_file
		fautes_analyse(test, yTest, predictions, output)




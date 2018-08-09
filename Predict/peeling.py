import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import sys
import argparse
import time
import collections
from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pprint

def read_file(file, mode, negTag=True):
    sentences = []
    if mode == 'train':
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
            print("%d sentences train read." % count)
            return sentences, labels
        
    elif mode == 'test':
        for a_sent in file:
            # print("before stripping: " + a_sent)
            a_sent = stripping(a_sent)
            # print("after stripping : " + a_sent)
            a_sent = pre_processing(a_sent.split(), negTag)
            sentences.append(a_sent)
        return sentences


def stripping(a_sent, limit=100):
    key_word = "smell fragrance odor scent".split() # sentences without any one of the key words would be stripped out
    obj = []
    raw = a_sent.lower()

    inds = [] # find all the indexes that contains key words
    for w in key_word:
        inds.extend([m.start() for m in re.finditer(w, raw)])
    # print(inds)
    part = len(inds) # indicate the number of sub-parts contains key words
    # print(inds)
    position = defaultdict(int)

    # write the start and end position of sub-sentences for each key words in the form of a dictionary
    for ind in inds:
        stop1 = ind # default left border, beginning of key word
        stop2 = len(raw) # default right border, end of sentence
        for i in reversed(range(stop1)):
            punct = re.search(r"([,.?!;]+)", raw[i])
            if punct: # has punctuation before the sub-sent which contains the key-word, exmp: 'xxx. I like the smell'
                stop1 = i + 1
                break
            else:
                stop1 = 0 # if no punctuation separating the sub-sent that contains the key-word, start from the first word of the sentence. exmp: 'i like the smell'

        for j in range(ind, stop2):
            punct = re.search(r"([,.?!;]+)", raw[j])
            if punct: # has punctuation after the sub-sent, exmp: 'i like the smell. however xxx', strip right after the punctuation
                stop2 = j + 1
                break
            # if no punctuation, take stop2 default value till the end of the sentence
        position[ind] = (stop1, stop2)

    for ind, (a, b) in position.items():
        sub_sent = raw[a:b] # create sub sentences
        # double check, if the sub-sentence is too long, probably due to lack of punctuation, strip again based on the length of characters
        begin, end = a, b        
        if len(sub_sent) > limit: # limit is the longest length of characters, by default = 100
            # print(sub_sent)
            sub_sent = sub_sent.split()
            # find exactly the key word in the sub-sentence
            for i in range(ind, b):
                if raw[i] == " ":
                    end = i
                    break
            for j in reversed(range(ind)):
                if raw[j] in [" ", '.', '!', '\xa0', '?']:
                    begin = j + 1
                    break
            mot = raw[begin:end] # find the key word
            print(mot)
            mot_ind = sub_sent.index(mot) # find the position of key word
            a,b = 0, -1 # re-scale the beginning(a) and end(b) position of sub-sentence
            if mot_ind > 5: # if there is more than 5 words before key word, take the nearest 5
                a = mot_ind - 5
            if len(sub_sent) - mot_ind > 5: # if there is more than 5 words after key-word, take the nearest 5
                b = mot_ind + 5
            sub_sent = " ".join(sub_sent[a:b])
        obj.append(sub_sent)
    
    obj = " ".join(i for i in obj)
    # print(obj)
    return obj
    

def pre_processing(tokens, negTag):

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
            
    motsTraites += ["</s1>", "</s2>"] #cancatenate la fin de phrase

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
                        # print(stop)
                        break          
                for i, target in enumerate(motsTraites[start:stop]):
                    motsTraites[start+i] = "NOT_" + target
    
    a_sent = " ".join(mot for mot in motsTraites) #group the words as a string
    return a_sent


def modelling(reviews, yTrain, k, featS=True):
    """"vectorize features for NB modelling """
    t0 = time.time()
    # initialiser un objet vectorizer
    vectorizer = CountVectorizer(binary=False, ngram_range=(1,2), stop_words='english')
    XTrain = vectorizer.fit_transform(reviews) #learn the vocab dict and return term-doc matrix
    
    feat_time = time.time() - t0
    print("(time for feature transformation = {:.4f} seconds)".format(feat_time))
    feat_names = vectorizer.get_feature_names() # names are sorted alphabetically
    print("{} features in total. Consider top {} features".format(len(feat_names), k))

    # create object NB model and fit
    model_nb = MultinomialNB()
    model_nb.fit(XTrain, yTrain)

    # feature selection with K best values of chi-square test
    ch2 = SelectKBest(chi2, k)

    if featS: #if use chi2 for feature selection, fit again new XTrain 
        XTrain_new = ch2.fit_transform(XTrain, yTrain)
        model_nb.fit(XTrain_new, yTrain)

    return vectorizer, ch2, model_nb


def predict(vectorizer, ch2, model, one_test, featS=True, trace=True):
    """test with one group of XTest, nb of examples decided by interval"""
    # without feature selection
    if not featS:
        XTest = vectorizer.transform(one_test)
        termTest = vectorizer.inverse_transform(XTest)

    # with feature selection
    else:
        XTest = ch2.transform(vectorizer.transform(one_test))
        termTest = vectorizer.inverse_transform(ch2.inverse_transform(XTest)) # first trans-back to vectors without ch2, then trans-back to real tokens
    
    if trace:
        print("\n[FEATURE SELECTED]")
        for term in termTest:
            print(term) #show true terms

    pred = model.predict(XTest)
    pred_proba = model.predict_proba(XTest)

    print("\n[PREDICTION & SCORE]")
    for i, (label, proba) in enumerate(zip(pred, pred_proba)):
        print("p{}: {} {}".format(i+1, label, proba))
    
    return (['pos' if one_pred == 1 else 'neg' for one_pred in pred], pred_proba) #return a tuple with first element the class, and the proba for each of the two classes


def debug(lst_phrase):
    print("[BEFORE PRE-PROCESSIN]\n" + str(lst_phrase) +"\n")
    pre_pro = read_file(lst_phrase, 'test')
    print("[AFTER PRE-PROCESSING]\n"+ str(pre_pro))
    voc = set()
    for token in str(pre_pro).split():
        voc.add(token)

    # train model
    print("\n[MODELLING]")
    reviews, labels = read_file('../ModellingCorpus/mediumTrain1.csv', 'train')
    kFeat = 10000
    vectorizer, ch2, model_nb = modelling(reviews, labels, kFeat)

    predict(vectorizer, ch2, model_nb, pre_pro)



if __name__=='__main__':

    # test phrases
    p1 = "refreshing, clean, calming, smells good. I also really like that it's natural and it's made with a bunch of chemicals"
    p1_a = "refreshing, clean, calming, smells good. I also really like that it's natural and it's not made with a bunch of chemicals"
    p1_b = "refreshing,  clean, calming, smells good. I wasn't convinced of this when I first tried it because of the texture and application, but I really like it now. It seems to have helped my acne a lot and I feel very clean and smelled refreshed after washing my face. I also really like that it's natural and it's not made with a bunch of fragrance of chemicals"

    p2 = "It doesn't smell bad. I like it."
    
    p3 = "like"

    p4 = "This face wash takes off my makeup & the wash smells good.."
    p4_o = "This face wash takes off my makeup & the wash smells good.. I haven't noticed a difference in my pores & I still have the occasional break out. I'm not sure if I will buy it again."

    p5 = "The smell is great and fresh."

    p6 = "It didnt burn my eyes or smell bad like other eye creams I've tried."

    p7 = "This toner smells amazing!"
    p7_o = "I just started using this product so I can't really say its great or bad.. It's only been a couple days for me but I can defiantly tell that my skin is less oily. This toner smells amazing!"

    p8 = "The product itself is very light, faint nice smell,"
    p9 = "Nearly fragrance-free, though I've caught a whiff of a citrusy smell when I sniffed the cream up-close,"

    p10 = "Pros:-There is little to no scent to this."
    p10_o = "I usually use Benefit's eye cream, but I'm so in love with Tarte Maracuja oil that I thought I'd try a sample of this and maybe switch over. I don't think I'll be making a switch, but this product definitely has some benefits.Pros:-There is little to no scent to this. I know some people have major problems with scented products, so this would be good for them.-This is very moisturizing. My eyes are smooth and soft when I wake up in the morning.-A little goes a very long way.-Since my eyes look smoother, I'd say there is an anti-aging component to that.-It doesn't sting or make my eyes water if a little melts into my eyes, which is fantastic for me. I may not have fragrance sensitivity, but my eyes are SO easily irritated.Cons:-This cream does not settle in to the skin very well. It seems like it is impossible to wear it under makeup. I let it sit for about 45 minutes to an hour before I did my makeup, and my normally beloved concealer still wouldn't stay put.If you're looking for a soothing, moisturizing cream to dab on before bedtime, this is the one for sure, but something else would be the ticket for daytime use."

    p11 = "foamy, milky, fresh, ineffective. During the summer my skin gets very oily so I thought I could help combat this with Zero Oil. I was excited to try it at first but the novelty soon wore off, and grew into disdain for this product. I can't wait to finish using this so I can stop regretting this purchase.PROS- Smells refreshing- Doesn't irritate my sensitive skinCONS- Did not remove/reduce oil for me- Did not reduce acne- On the pricier side compared to what I usually buy"

    p12 = "refreshing clean calming smells good I wasn't convinced of this when I first tried it because of the texture and application but I really like it now It seems to have helped my acne a lot and I feel very clean and refreshed after washing my face I also really like that it's natural and it's made with a bunch of chemicals"

    p13 = "brightening, light. This has got to be one of my favorite finds from Sephora. I didn't think it would work for me, but it did. I applied it twice/day -day and night - for 2 weeks and I could tell the difference already. I will keep using this product, I am VERY happy with this. It's very light on the skin, you can't feel it after you put some on. Nearly fragrance-free, though I've caught a whiff of a citrusy smell when I sniffed the cream up-close, but I could just be imagining the scent too. You don't need too much, just a dab on the ring finger and apply around the eye area. The sooner you get into the habit of using eye treatment, the better!"

    p14 = "gentle, refreshing, non-drying, foamy, deep cleaning. I use this cleanser twice a day and it has never broken me out or irritated my skin, unlike other cleansers I've tried in the past. I've even accidentally rubbed it on my eyes, and after quickly rinsing it off I had no bad reaction. It makes my skin feel so clean, refreshed, and tingly—just as claims to. I love the consistency of the cleanser; it's very thick but when you massage it with your wet fingers it turns into thick foam and it feels (and smells) very refreshing on your face. A little goes a long way, too. My pores look much better and much cleaner after using it, and I've noticed they look much better than they used to before I started using this product. I've even had people notice and comment on how good my skin looks! It's great at controlling my oily T-zone without drying out my skin, either, which is fantastic because I have extremely combination skin!I'm so happy I purchased this cleanser, I love it so much, and I'll definitely buy it again! :)"

    p15 = "foamy, refreshing. I use this cleanser and also the toner and moisturizer in this line. I use it with my clarisonic. This cleanser feels so refreshing and doesn't leave my face feeling tight. I actually like the smell. You can tell it cleans deep, It does help with acne, but not a miracle product for acne. It doesn't control my oil like it claims."

    p16 = "This is a miracle in a jar! It's absolutely amazing! I've only used it once so I can't review it's long term benefits but I was so impressed with the immediate results that I had to post a review! I'm always on the hunt for products (makeup & skin care) that will give me that glowing, dewy look... I have found some that I really love & this is definitely at the top of my list. It brightened up my eyes instantly! My under eye area was glowing! I wish I could put it on my entire face! Tarte should really come out with an all over face cream that delivers the same effect! Totally obsessed! A little goes a long way so this will last quite awhile but I will definitely be repurchasing this when I run out. It didnt burn my eyes or smell bad like other eye creams I've tried. I highly recommend! Hope this review has helped. =)"

    p17 = "I usually use Benefit's eye cream, but I'm so in love with Tarte Maracuja oil that I thought I'd try a sample of this and maybe switch over. I don't think I'll be making a switch, but this product definitely has some benefits.Pros:-There is little to no scent to this. I know some people have major problems with scented products, so this would be good for them.-This is very moisturizing. My eyes are smooth and soft when I wake up in the morning.-A little goes a very long way.-Since my eyes look smoother, I'd say there is an anti-aging component to that.-It doesn't sting or make my eyes water if a little melts into my eyes, which is fantastic for me. I may not have fragrance sensitivity, but my eyes are SO easily irritated.Cons:-This cream does not settle in to the skin very well. It seems like it is impossible to wear it under makeup. I let it sit for about 45 minutes to an hour before I did my makeup, and my normally beloved concealer still wouldn't stay put.If you're looking for a soothing, moisturizing cream to dab on before bedtime, this is the one for sure, but something else would be the ticket for daytime use."

    p18 = "effective, harsh. Does it work? Yes, but not any better than the drugstore alternatives. I'm not sure what their definition of fresh-scented is but this smells no different than any other hair removal cream. I wouldn't say it necessarily burned me, but it was definitely an uncomfortable experience. Bottom line: It got the job done but there are better products out there at better prices."

    p19 = "hydrating. There were things I loved and hated about No-poo, but the bad far outweighed the good.Pros:-Smells fine-Hair feels slick and smooth in showerCons:-Hair felt & looked oily mid-day (normally feels oily after 3 days)-Hair drooped after a couple of hours-Looked worse after each use (didn't remove oils enough)I'm all for using a product with less bad stuff in it, but it has to perform. And I wouldn't have thought my 2A fine hair would be so difficult for it to manage. But this just wasn't right for me. Maybe I'll try the low-poo instead."

    p20 = "long wearing, fast-drying, chip resistent, gorgeous colours, excellent quality application brush. I'm an old fashionista who has purchased many polishes, all price ranges, and this, by far, is the best ever. The fact the polish itself doesn't have that awful stinky smell is wonderful, esp for asthma/allergy sufferers, but the polish lasts! And the best part is the brush - it holds the polish and applies it all to the nail at one time. Excellent product!!! Better than Chanel, Lippman, Opi, etc."

    # p = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
    # p = [p1, p2]
    # p = ["like", "love", "favorite", "don't like", "don't love", "not favorite"]
    # p = ["not good", "good", "not bad", "bad"]
    # p = ['I actually like the smell.']
    # p = [p10_o, p11, p12, p13, p14]
    # p = [p14, p15, p16, p17]
    p = [p20]

    debug(p)

    # stripping(p12)




    

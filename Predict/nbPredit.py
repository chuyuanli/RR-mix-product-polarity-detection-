import nbModel
import pymysql.cursors
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import sys
import argparse
import time
from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def read_post(cont, negTag):
    """
    take raw contents from mysql, format: ['content1', 'content2']
    return a stripped and pre-processed list of strings
    """
    sentences = []
    for a_sent in cont:
        a_sent = pre_processing(stripping(a_sent, limit).split(), negTag)
        sentences.append(a_sent)
    return sentences


def stripping(a_sent, limit):
    """
    2 steps for stripping:
        - take only the sub sentence with key words, separated by punctuations
        - restrict the length of the sub sentences by limit of the characters
    return a string
    """
    key_word = "smell fragrance odor scent".split()
    obj = []
    raw = a_sent.lower()

    inds = [] # find all the indexes that contains key words
    for w in key_word:
        inds.extend([m.start() for m in re.finditer(w, raw)])
    part = len(inds) # indicate the number of sub-parts contains key words
    position = defaultdict(int) # create a dictionnary to stock the beginning and end position of sub-sentence for each key word

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
            # print("mot cle = " + mot)
            mot_ind = sub_sent.index(mot) # find the position of key word
            a,b = 0, -1 # re-scale the beginning(a) and end(b) position of sub-sentence
            if mot_ind > 5: # if there is more than 5 words before key word, take the nearest 5
                a = mot_ind - 5
            if len(sub_sent) - mot_ind > 5: # if there is more than 5 words after key-word, take the nearest 5
                b = mot_ind + 5
            sub_sent = " ".join(sub_sent[a:b])
        obj.append(sub_sent)
    
    obj = " ".join(i for i in obj)
    return obj

    

def pre_processing(tokens, negTag):
    """
    take the stripped sub-sentence for pre-processing:
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
    return a_sent


def predict(vectorizer, ch2, model, one_test, idTest):
    """
    test with one group of XTest, nb of examples decided by interval
    return a tuple: (label, [score_pos score_neg])
    """
    XTest = ch2.transform(vectorizer.transform(one_test)) #reshape text into vectors and transform with ch2
    pred = model.predict(XTest)
    pred_proba = model.predict_proba(XTest)

    #return a tuple with first element the class, and the second element the probability for each of the two classes
    return (['pos' if one_pred == 1 else 'neg' for one_pred in pred], pred_proba) 


def fetch_sql(interval, update=False):
    """
    major parts: 
        - connect to mysql and get the content
        - pass the content to predict
        - if required, update the result to mysql
    """
    # Connect to the database
    connection = pymysql.connect(host=host,
                                user=user,
                                password=pw,
                                db=db,
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor)

    tableName = table
    preds = []
    ids = []
    score_pos = []
    score_neg = []
    
    try:
        with connection.cursor() as cursor:
            t1 =  time.time()
            for start in range(0, 20000, interval):
                # sql = "SELECT `id`, `content` FROM " + tableName + " WHERE smell_tag IS NULL ORDER BY `id` LIMIT "+ str(start)+ "," + str(interval)
                sql = "SELECT `id`, `content` FROM " + tableName + " ORDER BY `id` LIMIT "+ str(start)+ "," + str(interval)
                cursor.execute(sql)
                results = cursor.fetchall()
                cont = [exm['content'] for exm in results]
                idTest = [exm['id'] for exm in results]
                ids.extend(exm['id'] for exm in results)

                # predict process
                xTest = read_post(cont, negTag)
                res_class, res_proba = predict(vectorizer, ch2, model_nb, xTest, idTest)
                preds.extend(res_class)
                score_pos.extend(res_proba[i][0] for i in range(len(idTest)))
                score_neg.extend(res_proba[j][1] for j in range(len(idTest)))
                print("posts traites {}".format(start+interval))
        
        pred_t = time.time() - t1
        print("(time for prediction = {:.4f} seconds)\n".format(pred_t))

        if update: # insert the result into sql
            print("UPDATING TO DATABASE...")
            with connection.cursor() as cursor:
                t2 = time.time()
                for i in range(len(ids)):
                    sql_update = "UPDATE "+tableName+" SET `smell_tag` = '"+ preds[i] +"', `score_pos` = ROUND(" + str(score_pos[i]) +",4), `score_neg` = ROUND(" + str(score_neg[i]) +",4) WHERE id = "+ str(ids[i])
                    cursor.execute(sql_update)  
            connection.commit() # commit to save the changes, each time commit 1k posts
            update_t = time.time() - t2
            print("(time for updating = {:.4f} seconds)\n".format(update_t))

    except pymysql.InternalError as e:
        print("Error {!r}, errno is {}".format(e, e.args[0]))

    finally:
        connection.close()
        c_neg = preds.count('neg')
        c_pos = preds.count('pos')
        print("Fini\npositif posts vs. negatif posts = {} vs. {}".format(c_pos, c_neg))


def show_config():
    # show the configuration of modelling
    print("----- PARAMATER SETTING ------")
    print("1) Ngram: 1 - {}\n2) Negation tag: {}\n3) Language: English\n4) Bernoulli: {}\n5) Binary features: {}\n6) Stop words: {}\n7) Choose top features: {}\n8) Character length limit of treated sentences: {}\n9) Patch length for each extraction mySQL: {}\n10) Choose to update the result to mySQL: {}".format(ngram, negTag, ber, binary, stopwords, kFeat, limit, interval, update))
    print("---- END PARAMATER SETTING ----\n")



if __name__=='__main__':

    usage = """ NAIVE BAYES MULTINOMIAL MODEL
    """+sys.argv[0] + """[options] NGRAM BINARY OUTPUT_FILE

    FILETRAIN is a csv file that contains 20k train data, with the format [NB][\t][ID][\t][PHRASE][\t][LABEL]
    - prog [options] integer NGRAM
            => the upbound of ngram, choose among 1-3
    - prog [options] boolean BINARY
            => change the features to binary
    - prog [options] boolean NEGTAG
            => the use of negation tag NOT_ before the words that following a negation word such as "doesn't", "not"
    - prog [options] integer KBEST
            => take into account the top best k features for modelling, suggest 10000
    - prog [options] integer INTERVAL
            => equals to LIMIT 0, i in mySQL, take i posts each time from the cursor, suggest 1000
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("fileTrain", type=str, default=None, help="original train file, format csv")
    parser.add_argument("-n", "--ngram", type=int, default=1, help="upper boundary of the range of n-grams to be extracted as features")
    parser.add_argument('-b', "--binary", default=False, help="if true, all non zero counts are set to 1. useful for binary model events")
    parser.add_argument('-nt', "--negtag", default=False, help="if true, add NOT_ tags to words after negation tones")
    parser.add_argument('-ber', "--bernoulli", default=False, help="if true, use BernoulliNB instead of MultinomialNB")
    parser.add_argument("-k", "--kBest", type=int, help="select the top k features for modelling")
    parser.add_argument("-s", "--stopwords", type=str, default=None, help="english or None. If None, no stop words will be eliminated")
    parser.add_argument("-i", "--interval", type=int, help="treat i posts each time, i should > 0 and < 20000")
    parser.add_argument("-l", "--limit", type=int, default=100, help="the limit of stripped sentence, the second criterion to pass if the sub-sentence is too long, refer to function stripping")
    parser.add_argument('-up', "--update", default=False, help="if true, update the result into database. Pay attention, make sure the corresponding colume is empty for insertion.")
    parser.add_argument('-ho', "--host", type=str, default='index-fr.semantiweb.fr', help="host")
    parser.add_argument("-u", "--user",type=str, default='chuyuan', help="user name")
    parser.add_argument("-pw", "--password", type=str, default='chuyuan', help="user code")
    parser.add_argument("-db", "--database", type=str, default='ratings_and_reviews_ml', help="database name")
    parser.add_argument("-t", "--tableName", type=str, default="customers_avis_smell", help="tableName")
    args = parser.parse_args()

    # load arguments
    fileTrain = args.fileTrain
    ngram = args.ngram
    binary = args.binary
    negTag = args.negtag
    ber = args.bernoulli
    kFeat = args.kBest
    stopwords = args.stopwords
    interval = args.interval
    limit = args.limit
    update = args.update
    host = args.host
    user = args.user
    pw = args.password
    db = args.database
    table = args.tableName

    if stopwords != None and stopwords != 'english':
        raise ValueError("Invalid stopwords type! Enter None or 'english'.")
    if kFeat == None:
        print("No top K features selected, by default choose top 10k features.")
        kFeat = 10000
    if interval == None:
        print("No interval number entered, By default interval is 1000.")
        interval = 1000
    show_config()

    # load script nbModel and prepare for modelling
    print("LOADING MODELLING PROCESS...")
    reviews, labels = nbModel.read_file(fileTrain, negTag)
    vectorizer, ch2, model_nb = nbModel.modelling(reviews, labels, kFeat, ber, binary, stopwords, ngram)

    # predict and update in mySQL
    print("PREDICTING PROCESS...")
    fetch_sql(interval)

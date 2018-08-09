import json
import os

def pre_processing(rawString):
    """pre-processing: delect strings like '\x92', normalize words"""

    # 1.eliminate the space at the begining
    if rawString.startswith(" "):
        rawString = rawString[1:]

    # change to lower-case
    rawString = rawString.lower()

    # 2.unicodes to be changed
    changes = {(u"\x93", u"\x94", u"\x96", u"\r", u"<em>", u"</em>", "{", "}"):u"", 
                (u"\x92",u"''", u"\'", u"’"):u"'", 
                (u"\x97",u"\n"):u" "}

    # use replace to clean the raw sentences
    for oldS, replacemt in changes.items():
        for old in oldS:
            rawString = rawString.replace(old, replacemt)
            
    # 3.promotion sentences to be delected
    to_delect = "[this review was collected as part of a promotion.]".split()
    possible_parts = []
    for i in range(len(to_delect)):
        part = ' '.join(to_delect[:len(to_delect)-i])
        possible_parts.append(part)
    for part in possible_parts:
        if rawString.endswith(part):
            rawString = rawString.replace(part, "")

    # 4. negation spelling correction
    negWords = "doesnt didnt wasnt dont havent cant couldnt wouldnt shouldnt".split()
    for i in range(len(negWords)):
        if negWords[i] in rawString:
            rawString = rawString.replace(negWords[i], negWords[i][:-2]+"n't")

    #return pre-processed sentence
    # print(rawString)
    return rawString

            
# pre_processing("It doesn\x92t help with hydration. If that''s the smell you''re going for, [This review was collected as part of a promotion.]")


#======================================================================
def extract_from_json(file):
    """
    read a json file and transform into csv file
    format: nb \t id \t sent \t tone
    
    """
    # with urllib.request.urlopen(url) as url2:
    #     data = json.loads(url2.read().decode())
    try:
        f = open(file, 'r')
        data = json.load(f)
        with open ('fr_11k_neg.csv','w') as output:
            i = 1
            for nb, content in data["highlighting"].items():
                # print(content)
                for phrase in content.values():
                    # print(phrase)
                    phrase = pre_processing(str(phrase)[2:-2])
                    # print(phrase)
                    # input()
                    output.write(str(i)+'\t'+str(nb)+'\t'+str(phrase)+'\tnegatif')
                    i += 1
                output.write('\n')
                # input()
            output.close()
        f.close()
        
    except FileNotFoundError:
        print("ERROR: "+ str(file) + " doesn't exist.")

# file = 'fr_11k_neg.json'
# extract_from_json(file)


#======================================================================
def pure_text(file):
    with open(file, 'r') as f:
        data = f.readlines()
        with open('MarkovGenerator/example_neg.txt','w') as output:
            for line in data[101:200]:
                phrase = line.split('\t')[2]
                # print(phrase)
                # input()
                output.write(str(phrase)+'\n')
        output.close()
    f.close()

# file = '18k_neg_hl.csv'
# pure_text(file)

#======================================================================
def count_voc(csvFile):
    """ check the nb of voc, sent and token in a csv file"""

    voc = []
    sent = 0
    with open(csvFile) as f:
        lines = f.readlines()
        for line in lines:
            phrase = line.split('\t')[2]
            voc.extend(phrase.split())
            # print(voc)
            # input()
            sent += 1
    
    f.close()
    print(sent)
    print(len(voc))
    print(len(set(voc)))

# count_voc('fr_25k_pos.csv')



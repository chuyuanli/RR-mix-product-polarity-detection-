import re

def read_file(file):
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
                # if curly_q in mot:
                #     print(mot)
                #     input()
                #     mot.replace("’", "'")
                if punct:
                    p = punct.group(1) #capturer les punctuations
                    newMots = [m.lower() for m in mot.replace(p, ' '+p+' ').split()]
                    motsTraites.extend(newMots)
                else:
                    motsTraites.append(mot.lower())
            motsTraites += ["</s1>", "</s2>"] #cancatenate la fin de phrase


            #add NOT_ to the word after negative tags and before the next punctuation
            for index, word in enumerate(motsTraites):
                if word in negtags:
                    start = index + 1 # find the index of first word after negation
                    # print(word)
                    # print(start)
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



reviews = read_file("../ModellingCorpus/smallTrain1.csv")
# test = read_file("../smallTest1.csv")
print()

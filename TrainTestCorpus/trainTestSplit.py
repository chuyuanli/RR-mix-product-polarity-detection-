# import numpy as np
# from skleran.model_selection import train_test_split

# X = np.array(25000)
# X_train, X_test = train


negFile = "../25k_neg_highlight.csv"
medCorpus = "mediumTest1.csv"

with open(negFile, 'r') as inputF:
    with open(medCorpus, 'a') as output:
        lines = inputF.readlines()
        output.write('\n')
        for line in lines:
            nb = line.split()[0]
            if int(nb) > 23000 and int(nb) <= 25000 :
                output.write(line)
        output.close()
    inputF.close()


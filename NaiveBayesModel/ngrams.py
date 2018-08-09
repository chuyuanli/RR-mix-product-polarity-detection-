from itertools import tee, islice
from collections import Counter
from nltk.util import ngrams

def ngrams1(lst, n):
  tlst = lst
  while True:
    a, b = tee(tlst)
    l = tuple(islice(a, n))
    if len(l) == n:
      yield l
      next(b)
      tlst = b
    else:
      break


def ngrams2(txt_string, min=1, max=2):
    s = []
    for n in range(min, max+1):
        for ngram in ngrams(txt_string, n):
            s.append(' '.join(str(i) for i in ngram))
    # print(s)
    return s


words = "the quick person did not realize his speed and the quick person bumped . He likes my perfum !"

newng = Counter(ngrams2(words.split()))
print(newng)
print(newng['person'])
print(set(newng.keys()))

if 'person' in newng:
  print('yes')


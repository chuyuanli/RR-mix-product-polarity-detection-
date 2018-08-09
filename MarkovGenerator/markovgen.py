import random

class Markov(object):
	
	def __init__(self, open_file):
		self.cache = {}
		self.open_file = open_file
		self.words = self.file_to_words()
		self.word_size = len(self.words)
		self.database()
		
	
	def file_to_words(self):
		self.open_file.seek(0)
		data = self.open_file.read()
		words = data.split()
		voc = set(words)
		print("Voc={} Token={}".format(len(voc), len(words)))
		return words
		
	
	def triples(self):
		""" Generates triples from the given data string. So if our string were
				"What a lovely day", we'd generate (What, a, lovely) and then
				(a, lovely, day).
		"""
		if len(self.words) < 3:
			return
		
		for i in range(len(self.words) - 2):
			yield (self.words[i], self.words[i+1], self.words[i+2])
			

	def database(self):
		print(len(list(self.triples()))) #7394
		for w1, w2, w3 in self.triples():
			key = (w1, w2)
			if key in self.cache:
				self.cache[key].append(w3)
			else:
				self.cache[key] = [w3]        
		# print(len(self.cache)) #5147
		# print(self.cache)
				

	def generate_markov_text(self, file, size=15, sent=7000):
		"""
		generate 7k sentences with an averaged length of 15 words
		write down in the file
		"""
		seed = random.randint(0, self.word_size-3)
		end = "</f>"
		# print(seed)
		seed_word, next_word = self.words[seed], self.words[seed+1]
		w1, w2 = seed_word, next_word
		restart = 0

		with open (file, 'a') as output: # 'append' instead of 'w'
			index = 18001 # the previous 18k sentences are already written down in file 
			for i in range(sent):
				gen_words = [] # record one sentence	
				for j in range(1, size):
					gen_words.append(w1)
					# when comes to the end of words, restart with a new random seed number for w1 and w2
					if w2 == end:
						restart += 1 # record the restarting number
						seed = random.randint(0, self.word_size-3)
						w1, w2 = self.words[seed], self.words[seed+1]
					w1, w2 = w2, random.choice(self.cache[(w1, w2)])
				gen_words.append(w2)
				# print(str(i+1) + '. ' + ' '.join(gen_words))
				sentence = ' '.join(gen_words)
				output.write(str(index)+'\t0000000\t'+str(sentence)+'\tnegatif\n')
				index += 1
		output.close()
		# print(restart)

		
if __name__=='__main__':
	mk = Markov(open('seed_neg.txt','r'))
	test = mk.generate_markov_text('../RawData/25k_neg_highlight.csv')

 
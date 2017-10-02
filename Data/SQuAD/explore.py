import json
from tqdm import tqdm
import pandas as pd
import sys

f = open('PreProcessed_Data/augmented_train.txt')
datas = f.read()
f.close()


data = json.loads(datas)

print 'DATA LOADED\n'
for d in data['data']:
	# print '*******',d['title'].upper(),'********'
	for p in d['paragraphs']:
		for tok in p['tokens']:
			print tok['word'],'-',
		for q in p['qas']:
			for tok in q['tokens']:
				print tok['word'],'-',
			print '\n'
			for a in q['answers']:
				print '>',a['text']
		break
	break


print '#'*10, 'Count the max length' , '#'*10
mp,mq,ma = 0,0,0

for d in tqdm(data['data']):
	for p in d['paragraphs']:
		mp = max(mp,len(p['tokens']))
		for q in p['qas']:
			mq = max(mq,len(q['tokens']))
			for a in q['answers']:
				pass

print 'Paragraph : ',mp # 767
print 'Question : ',mq # 49


# check if on join by ' ' and split at ' ' makes any difference -> NO

for d in tqdm(data['data']):
	for p in d['paragraphs']:
		tok = [i['word'] for i in p['tokens']]
		a = len(tok)
		b = len(' '.join(tok).split(' '))
		if not (a == b):
			print "ERROR in",tok
		for q in p['qas']:
			tok = [i['word'] for i in q['tokens']]
			a = len(tok)
			b = len(' '.join(tok).split(' ')) # <-- remember ' '
			if not (a == b):
				print "ERROR in",tok
			for a in q['answers']:
				pass  

# Vocabulary is important if you want to make trainable embeddings. Then You also need to have a constant size lookp table.
# DATA UTILS : PREPARE VOCABULARY FILE
class Vocabulary:
	def __init__(self):
		self.word_to_index = dict()
		self.index_to_word = dict()
		self.counter = 0
		self.default_index = 0
		self.default_word = '<default_word>'

	def insert(self,word):
		word = word.lower()
		self.index_to_word[self.counter] = word 
		self.word_to_index[word] = self.counter 
		self.counter += 1

	def make_vocab(self, data):
		# self.insert(self.default_word)
		for d in tqdm(data['data']):
			for p in d['paragraphs']:
				for tok in p['tokens']:
					if not self.word_to_index.has_key(tok['word'].lower()): 
						self.insert(tok['word'])
				for q in p['qas']:
					for tok in q['tokens']:
						if not self.word_to_index.has_key(tok['word'].lower()): 
							self.insert(tok['word'])

	def get_word_index(self,word):
		word = word.lower()
		if word not in self.word_to_index.keys(): 
			return self.default_index
		return self.word_to_index[word]

	def get_sentence_index(self,sentence):
		sent = sentence.split(' ')
		return [ self.get_word_index(w) for w in sent]



	# def get_word(index):
	# 	if index >= counter: 
	# 		return self.default_word
	# 	return self.index_to_word[index]

vocab = Vocabulary()
vocab.make_vocab(data)
for i in vocab.index_to_word.keys():
	print i, vocab.index_to_word[i]
print vocab.get_sentence_index('The man went to the bathroom .') # So as to use lookuptable for retrieval of wordvectors.


# PREPARE DATA IN A PANDAS FILE
# def prepare_usable_data(data_obj):
# 	for d in data_obj['data']:
# 	for p in d['paragraphs']:
# 		tok = [i['word'] for i in p['tokens']]
# 		paragraph = get_sentence_index(' '.join(tok))
# 		for q in p['qas']:
# 			tok = [i['word'] for i in p['tokens']]
# 			question = get_sentence_index(' '.join(tok))
# 			for a in q['answers']:
# 				print '>',a['text']











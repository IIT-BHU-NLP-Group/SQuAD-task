import json
from tqdm import tqdm
import pandas as pd
import sys

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
		print 'Preparing Vocabulary ...'
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

	def get_index_word(self,idx):
		if (idx < 0) or (idx>=counter):
			return self.default_index
		return self.index_to_word[idx] 

	def get_sentence_index(self,sentence):
		sent = sentence.split(' ')
		return [ self.get_word_index(w) for w in sent]


# Class to handle data, make batches
class Data:
	def __init__(self, debug_mode = False, percent_debug_data = 10):
		print 'Loading data from file ...'
		with open('../data/SQuAD/PreProcessed_Data/augmented_train.txt') as f:
			self.datas = f.read()
			self.data = json.loads(self.datas)
		self.vocab = Vocabulary()
		self.vocab.make_vocab(self.data)
		self.data_size = 0 # No. of Examples
		for d in tqdm(self.data['data']):
			for p in d['paragraphs']:
				for q in p['qas']:
					self.data_size += 1
		if debug_mode:
			self.data_size = (0.01*percent_debug_data*self.data_size)
			self.data['data'] = self.data['data'][:self.data_size]

	def example_iter(self):
		for d in self.data['data']:
			for p in d['paragraphs']:
				p_tok = p['tokens']
				paragraph = ' '.join([i['word'] for i in p['tokens']])
				paragraph_idx = self.vocab.get_sentence_index(paragraph)
				for q in p['qas']:
					q_tok = q['tokens']
					question = ' '.join([i['word'] for i in q['tokens']])
					question_idx = self.vocab.get_sentence_index(question)
					for a in q['answers']:
						answers_span_char = (a['answer_start'],a['answer_end'])
						answers_span_id = (a['begin_id'],a['end_id'])
						yield {'para':paragraph, 'para_idx':paragraph_idx, 'question':question, 'question_idx':question_idx, 'ans_id_span':answers_span_id, 'ans_char_span':answers_span_char}

	def prepare_minibatch(self,batch):
		# returns ( N * M ) batch of indices in form of a 2D list
		zero_token = None # {'word':,'id':-1, } # Needs to be changed if you chane token in prerpocessing.py 
		max_question_len, max_para_len = 0, 0
		for e in batch:
			max_question_len = max(max_question_len, len(e['question_idx']))
			max_para_len = max(max_para_len, len(e['para_idx']))
		
		# Add padding 
		for e in batch:
			e['para_idx'] = [self.vocab.default_index]*(max_para_len-len(e['para_idx'])) + e['para_idx']
			e['question_idx'] = [self.vocab.default_index]*(max_para_len-len(e['question_idx'])) + e['question_idx']
		e['para_vectors'] = []
		e['question_vectors'] = []
		for e in batch:
			for i in e['para_idx']:
				e['para_vectors'].append(get_word_vector(self.vocab.get_index_word(i)))
			for i in e['question_idx']:
				e['question_vectors'].append(get_word_vector(self.vocab.get_index_word(i)))
		return batch

	def minibatch_iter(self, batch_size = 50):
		n = self.data_size // batch_size # No of elements in the Batch
		counter = 0
		minibatch = []
		for e in self.example_iter():
			if counter == n * batch_size :
				break
			counter += 1
			minibatch.append(e)
			if counter%batch_size == 0:
				yield self.prepare_minibatch(minibatch)
				minibatch = []

if __name__ == '__main__':
	d = Data()
	for j in d.minibatch_iter(2):
		for i in j:  
			print 'P>',i['para']
			print 'Q>',i['question']
			print '*'*20
		break



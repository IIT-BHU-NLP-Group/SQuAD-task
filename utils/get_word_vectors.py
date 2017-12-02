import numpy as np

VECTOR_DIM = 50
NLP = None

def load_vector_dict():
	global NLP
	if NLP is not None:
		return

	NLP = {}
	with open("data/GloVe/glove.6B.50d.txt", "r") as file:
		for line in file:
			l = line.strip().split()
			NLP[l[0]] = np.array([float(l[x]) for x in range(1,VECTOR_DIM+1)])

def get_sentence_vectors(sentence):
	"""
	Returns word vectors for complete sentence as a python list"""
	s = sentence.strip().split()
	vec = [ get_word_vector(word) for word in s ]
	return vec

def get_word_vector(word):
	"""
	Returns word vectors for a single word as a python list"""

	load_vector_dict()

	if NLP.has_key(word):
		return NLP[word]
	else:
		return np.zeros(VECTOR_DIM)
	
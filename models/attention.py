import sys
import tensorflow as tf
import numpy as np
import pandas as pd

sys.path.append('../utils')
import data_utils as du
from get_word_vectors import get_word_vector

class DCNModel():
	"""docstring for Model"""
	def __init__(self, arg):
		self.batch_size = 50
		self.embed_size = 300
		self.lstm_units = 100
		self.max_epochs = 100
		self.dropout = 0.0

		self.data = du.Data(debug_mode = True, percent_debug_data = 10)
		
		self._encoder()
		self._decoder()

	def _encoder(self, lstm_units):
		# ADD PLACEHOLDERS
		self.passage_input_placeholder = tf.placeholder(tf.float32, (None, None, None))
		self.question_input_placeholder = tf.placeholder(tf.float32, (None, None, None))

		self.start_label = tf.placeholder(tf.int32, None)
		self.end_label = tf.placeholder(tf.int32, None)

		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_units)
		lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=1-self.dropout)
		
		question_embb, _ = tf.nn.dynamic_rnn(lstm_cell, self.question_input_placeholder, dtype=tf.float32)
		passage_embb, _ = tf.nn.dynamic_rnn(lstm_cell, self.passage_input_placeholder, dtype=tf.float32)

		# D and Q according to the paper
		passage_embb = tf.transpose(passage_embb, [0, 2, 1])
		question_embb = tf.transpose(question_embb, [0, 2, 1])

		# Affinity matrix calculation
		passage_embb_trans = tf.transpose(passage_embb, [0, 2, 1])
		affinity_matrix = tf.matmul(passage_embb_trans, question_embb)

		# A_Q and A_D calculation
		attention_weights_Q = tf.nn.softmax(affinity_matrix)
		attention_weights_D = tf.nn.softmax(tf.transpose(affinity_matrix, [0, 2, 1]))

		# Computing summaries/attention contexts
		att_context_Q = tf.matmul(passage_embb, attention_weights_Q)
		att_context_D = tf.matmul(tf.concat([question_embb, att_context_Q], 1), attention_weights_D)

		# Final coattention context
		bi_lstm_cell = tf.contrib.rnn.BidirectionalGridLSTMCell(self.lstm_units)
		bi_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=bi_lstm_cell, output_keep_prob=1-self.dropout)
		self.coattention_context = tf.nn.dynamic_rnn(
								bi_lstm_cell,
								tf.transpose(tf.concat([passage_embb, att_context_D], 1), [0, 2, 1]),
								dtype=tf.float32)

	def _decoder():
		
		loss_start = tf.losses.sparse_softmax_cross_entropy(self.start_label, self.coattention_context)
		loss_end = tf.losses.sparse_softmax_cross_entropy(self.end_label, self.coattention_context)
		
		self.loss = tf.reduce_sum(tf.add(loss_start, loss_end))

		self.optimizer = self.get_optimizer().minimize(self.cost_)

	def get_optimizer(self):
		return tf.train.AdamOptimizer()


def epoch(data):
	pass
	# for batch in data.minibatch_iter(batch_size):
	# 	para, question = [], []
	# 	for e in batch:
	# 		para.append(e['para_idx'])
	# 		question.append(e['question_idx'])
	# 	para, question = np.array(para).T, np.array(question)    
		
	# 	feed_dict = {document_input_placeholder :  , question_input_placeholder :  }




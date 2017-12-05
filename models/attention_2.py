import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm

sys.path.append('utils')

from get_word_vectors import VECTOR_DIM

class DCNModel():
	"""docstring for Model"""
	def __init__(self, data_reader, *arg):
		self.batch_size = 40
		self.embed_size = 300
		self.lstm_units = 100
		self.dropout = 0.0

		# self.data = DU.Data(debug_mode=True, percent_debug_data=1)
		self.data = data_reader
		
		self._encoder()
		self._decoder()

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def _encoder(self):
		# ADD PLACEHOLDERS
		self.passage_input_placeholder = tf.placeholder(tf.float32, (None, None, VECTOR_DIM))
		self.question_input_placeholder = tf.placeholder(tf.float32, (None, None, VECTOR_DIM))

		self.start_label = tf.placeholder(tf.int32, [None, 1])
		self.end_label = tf.placeholder(tf.int32, [None, 1])

		with tf.variable_scope('encoder') as scope:
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_units)
			lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=1-self.dropout)
			
			question_embb, _ = tf.nn.dynamic_rnn(lstm_cell, self.question_input_placeholder, dtype=tf.float32)
			scope.reuse_variables()
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
		lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_units)
		lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_units)

		outputs, states  = tf.nn.bidirectional_dynamic_rnn(
			cell_fw=lstm_fw_cell,
			cell_bw=lstm_bw_cell,
			dtype=tf.float32,
			inputs=tf.transpose(tf.concat([passage_embb, att_context_D], 1), [0, 2, 1]))

		self.coattention_context = tf.concat(outputs, 2)
		# self.coattention_context = tf.transpose(self.coattention_context, [0, 2, 1]) # batch_size * 2L * m

	def _decoder(self):
		embed = tf.reshape(self.coattention_context, [-1, 2*self.lstm_units])
		
		softmax_weight_start = tf.Variable(tf.random_uniform([2*self.lstm_units, 1]), name='softmax_weight_start')
		softmax_bias_start = tf.Variable(tf.random_uniform([1, 1]), name='softmax_bias_start')

		self.scores_start = tf.matmul(embed, softmax_weight_start) + softmax_bias_start
		self.scores_start = tf.reshape(self.scores_start, [-1, tf.shape(self.coattention_context)[1]])

		loss_start = tf.losses.sparse_softmax_cross_entropy(self.start_label, self.scores_start)
		
		softmax_weight_end = tf.Variable(tf.random_uniform([2*self.lstm_units, 1]), name='softmax_weight_end')
		softmax_bias_end = tf.Variable(tf.random_uniform([1, 1]), name='softmax_bias_end')
		
		self.scores_end = tf.matmul(embed, softmax_weight_end) + softmax_bias_end
		self.scores_end = tf.reshape(self.scores_end, [-1, tf.shape(self.coattention_context)[1]])

		loss_end = tf.losses.sparse_softmax_cross_entropy(self.end_label, self.scores_end)
		# TODO: End positions must adapt from start positions.
		
		self.loss = tf.reduce_sum(tf.add(loss_start, loss_end))

		self.optimizer = self.get_optimizer().minimize(self.loss)

	def get_optimizer(self):
		return tf.train.AdamOptimizer()

	def prepare_batch(self, batch):
		max_question_len, max_para_len = 0, 0
		for e in batch:
			max_question_len = max(max_question_len, len(e['question']))
			max_para_len = max(max_para_len, len(e['context']))
		
		# Add padding 
		for e in batch:
			e['padding_length_para'] = (max_para_len-len(e['context']))
			e['context'] = [self.data.vocab.default_index]*(max_para_len-len(e['context'])) + e['context']
			e['question'] = [self.data.vocab.default_index]*(max_question_len-len(e['question'])) + e['question']

			e['answer_start'] = e['answer_start'] + e['padding_length_para']
			e['answer_end'] = e['answer_end'] + e['padding_length_para']

			# Temporary
			if e['answer_end'] == max_para_len:
				e['answer_end'] -= 1;

		return batch

	def train(self, epochs=10):
		for epo in tqdm(range(epochs)):
			total_loss = []
			for batch in self.data.get_minibatch(self.batch_size):
				# Add padding
				processed_batch = self.prepare_batch(batch)

				para, question, start_index, end_index = [], [], [], []
				for e in processed_batch:
					para.append(self.data.vocab.get_sentence_embeddings(e['context']))
					question.append(self.data.vocab.get_sentence_embeddings(e['question']))
					start_index.append(e['answer_start'])
					end_index.append(e['answer_end'])

				para, question = np.array(para), np.array(question)    

				start_index, end_index = np.array(start_index), np.array(end_index)
				start_index = start_index.reshape([start_index.shape[0], 1])
				end_index = end_index.reshape([end_index.shape[0], 1])
				
				feed_dict = {
					self.passage_input_placeholder: para,
					self.question_input_placeholder: question,
					self.start_label: start_index,
					self.end_label: end_index
				}

				loss,_ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
				total_loss.append(loss)

			print "Epoch %s: Training Loss = %s \n" % (epo, np.sum(total_loss) / len(total_loss))

	def predict(self, batch):
		batch = self.prepare_batch(batch)

		para, question, start_index, end_index = [], [], [], []
		for e in batch:
			para.append(self.data.vocab.get_sentence_embeddings(e['context']))
			question.append(self.data.vocab.get_sentence_embeddings(e['question']))	

		feed_dict = {
			self.passage_input_placeholder: para,
			self.question_input_placeholder: question,
		}

		start, end = self.sess.run([self.scores_start, self.scores_end], feed_dict=feed_dict)
		start_index = np.argmax(start, axis=1)
		end_index = np.argmax(end, axis=1)

		answers = []
		for i1, i2, v in zip(start_index, end_index, batch):
			answers.append([i1 - v['padding_length_para'], i2 - v['padding_length_para']]) 
		return answers

	def load(self, path):
		saver = tf.train.Saver()
		saver.restore(self.sess, path)

	def save(self, path):
		saver = tf.train.Saver()
		saver.save(self.sess, path)



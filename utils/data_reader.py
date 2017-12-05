import pickle
import pandas as pd

from random import shuffle
from data_utils import Vocabulary

class DataReader:
	def __init__(
			self, 
			file_path,
			vocab_path,
			train_split_ratio=0.80,
			dev_split_ratio=0.10,
			test_split_ratio=0.10,
			debug_mode=False, percent_debug_data=10):
		self.file_path = file_path

		self.read_data(
			train_split_ratio, dev_split_ratio, test_split_ratio, 
			debug_mode, percent_debug_data)

		self.vocab = pickle.load(open(vocab_path, 'rb'))

	def read_data(
			self, 
			train_split_ratio, dev_split_ratio, test_split_ratio, 
			debug_mode, percent_debug_data):

		assert (train_split_ratio + dev_split_ratio + test_split_ratio) == 1

		# Read CSV file
		df = pd.read_csv(self.file_path, delimiter=",")
		self.data = df.to_dict('records')

		# Temporary fix (Error in making pandas CSV)
		for x in self.data:
			x['question'] = [int(i) for i in x['question'].strip('[').strip(']').split(',')]
			x['context'] = [int(i) for i in x['context'].strip('[').strip(']').split(',')]

		# Now, `self.data` is a list of dict
		# shuffle(self.data)

		if debug_mode:
			self.data = self.data[:int(percent_debug_data * 0.01 * len(self.data))]

		data_size = len(self.data)
		self.train = self.data[:int(train_split_ratio*data_size)]
		self.dev = self.data[
						int(train_split_ratio * data_size):
						int((train_split_ratio + dev_split_ratio) * data_size)]
		self.test = self.data[int(- (test_split_ratio) * data_size):]

		self.data_dict = {
			'train':	self.train,
			'dev':		self.dev,
			'test':		self.test}

	def get_minibatch(self, batch_size):
		# Useful function to return minibatches of training data
		train_size = len(self.train)

		assert batch_size < train_size

		count = train_size // batch_size

		for i in range(count):
			yield self.train[i*batch_size : (i+1)*batch_size]

	def get_complete_batch(self, choice):
		assert (choice == 'train' or choice == 'dev' or choice == 'test')

		return self.data_dict[choice]










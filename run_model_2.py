import pickle

from models import attention_2

from utils import data_reader as DR
from utils.data_utils import Vocabulary

VOCAB = pickle.load(open("./data/SQuAD/Augmented/pickled-vocab", 'rb'))

main_data_reader = DR.DataReader(
	file_path="data/SQuAD/Augmented/Data.csv",
	vocab_path="./data/SQuAD/Augmented/pickled-vocab",
	debug_mode=False,
	percent_debug_data=1)

model = attention_2.DCNModel(main_data_reader)

# model.load("./model.ckpt")

model.train(epochs=10)

ans = model.predict(main_data_reader.get_complete_batch('train')[:5])

for x,a in zip(main_data_reader.get_complete_batch('dev')[:5], ans):
	print "PASSAGE"
	for i in x['context']:
		print VOCAB.get_index_word(i),
	print "\nQUESTION"
	for i in x['question']:
		print VOCAB.get_index_word(i),
	print "\nAnswer:"
	print a
	print "Actual: ", [x['answer_start'], x['answer_end']]

model.save("./model.ckpt")



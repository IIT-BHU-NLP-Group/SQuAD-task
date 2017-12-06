import pickle

from models import attention_2

from utils import data_reader as DR
from utils.data_utils import Vocabulary

VOCAB = pickle.load(open("./data/SQuAD/Augmented/pickled-vocab", 'rb'))

main_data_reader = DR.DataReader(
	file_path="data/SQuAD/Augmented/Data.csv",
	vocab_path="./data/SQuAD/Augmented/pickled-vocab",
	debug_mode=True,
	percent_debug_data=1)

model = attention_2.DCNModel(main_data_reader)

model.load("./model.ckpt")

# model.train(epochs=10)

batch = main_data_reader.get_complete_batch('dev')[:50]

ans = model.predict(batch)

for x,a in zip(batch[:5], ans[:5]):
	print "PASSAGE"
	for i in x['context']:
		print VOCAB.get_index_word(i),
	print "\nQUESTION"
	for i in x['question']:
		print VOCAB.get_index_word(i),
	print "\nAnswer:"
	print a

	actual = [
		x['answer_start'] - x['padding_length_para'],
		x['answer_end'] - x['padding_length_para']]
	print "Actual: ", actual

	print "Prediction:"
	for i in x['context'][x['padding_length_para'] + a[0]:x['padding_length_para'] + a[1] + 1]:
		print VOCAB.get_index_word(i),
	print '\n'

n =  len(batch)
cnt = 0

for x, a in zip(batch, ans):
	actual = [
		x['answer_start'] - x['padding_length_para'],
		x['answer_end'] - x['padding_length_para']]
	if a == actual:
		cnt += 1

print "Exact Match (percent):" +  str(float(cnt*100)/n)


# model.save("./model.ckpt")

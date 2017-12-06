import pickle

from models import attention_2

from utils import data_reader as DR
from utils.data_utils import Vocabulary

def f1_score(actual_start, actual_end, predicted_start, predicted_end):
	if predicted_start > predicted_end :
		return 0
	a = max(actual_start, predicted_start)
	b = min(actual_end, predicted_end)
	overlap_len = 0
	prediction_len   = prediction_end - predicted_start + 1
	ground_truth_len = actual_end - actual_start + 1
	if a<=b:
		overlap_len = (b-a+1)
	else:
		return 0
    precision = 1.0 * overlap_len / prediction_len
    recall = 1.0 * overlap_len / ground_truth_len
    f1 = (2 * precision * recall) / (precision + recall)

VOCAB = pickle.load(open("./data/SQuAD/Augmented/pickled-vocab", 'rb'))

main_data_reader = DR.DataReader(
	file_path="data/SQuAD/Augmented/Data.csv",
	vocab_path="./data/SQuAD/Augmented/pickled-vocab",
	debug_mode=True,
	percent_debug_data=1)

model = attention_2.DCNModel(main_data_reader)
model.load("./model.ckpt")

def evaluate_data(choice,model,main_data_reader):
	assert (choice == 'train' or choice == 'dev' or choice == 'test')

	print "\nEvaluating the {} data..".format(choice)
	batch = main_data_reader.get_complete_batch(choice)
	ans = model.predict(batch)

	n =  len(batch)
	cnt = 0
	f1 = 0

	for x, a in zip(batch, ans):
		actual = [
			x['answer_start'] - x['padding_length_para'],
			x['answer_end'] - x['padding_length_para']]
		if a == actual:
			cnt += 1
		f1 += f1_score(actual[0], actual[1] ,a[0], a[1])

	f1 = (f1*100.0)/n

	print "Exact Match (percent):" +  str(float(cnt*100)/n)
	print "F1-Score (percent):" + str(f1)


evaluate_data('train', model, main_data_reader)
evaluate_data('dev', model, main_data_reader)



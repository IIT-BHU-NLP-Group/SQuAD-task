from models import attention_2

from utils import data_reader as DR

main_data_reader = DR.DataReader(
	file_path="data/SQuAD/Augmented/Data.csv",
	vocab_path="./data/SQuAD/Augmented/pickled-vocab",
	debug_mode=True)

model = attention_2.DCNModel(main_data_reader)
model.train()

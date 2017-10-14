import json
import xml.etree.ElementTree as ET
import sys
import codecs
from tqdm import tqdm
import os

def read_data(path_to_original_data = './Original/Train.txt'):
	print 'Reading Data ...'
	with open(path_to_original_data,'r') as f:
		DataJ = f.read()
	data = json.loads(DataJ)
	for d in tqdm(data['data']):
		for p in d['paragraphs']:
			context = p['context']
			for q in p['qas']:
				for a in q['answers']:
					answer = a['text']
					answer_start, answer_end = int(a['answer_start']), int(a['answer_start'])+len(answer)
					assert answer == context[answer_start:answer_end]
					a['answer_end'] = answer_end
	return data

def coreNLP(text,is_context = True):
	print "\n****  Parsing >>> ",text,' ****\n'
	f = codecs.open('./input.txt','w','utf-8-sig')
	f.write(text)
	f.close()
	os.system('java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -file input.txt')
	tree = ET.parse('input.txt.xml')
	root = tree.getroot()
	sentences = root.find('document').find('sentences').findall('sentence')
	sentence_list = []
	w_index = 0
	for s in sentences:
		sent = []
		# Get the Token information from the pipeline
		for tok in s.find('tokens').findall('token'):
			# Prepare tokens for the sentence
			Token = dict()
			Token['word']  = tok.find('word').text
			# Token['lemma'] = tok.find('lemma').text
			# Token['ner'] = tok.find('NER').text
			# Token['pos'] = tok.find('POS').text
			if is_context:
				Token['id']         = w_index # to be indexed from the start of the para.
				Token['char_begin'] = int(tok.find('CharacterOffsetBegin').text) -1
				Token['char_end']   = int(tok.find('CharacterOffsetEnd').text) -1
			else:
				Token['id']         = int(tok.attrib['id']) - 1
			sent.append(Token)
			w_index += 1
		sentence_list.append(sent)
	return sentence_list

def augment_data(data):
	os.chdir('./StanfordCoreNLP/')
	for d in tqdm(data['data']):
		for p in d['paragraphs']:
			aug_context = coreNLP(p['context'], True)
			words = []
			sents = []
			for i,s in enumerate(aug_context):
				words = words + s
				sents.append({"begin_id":s[0]['id'], "end_id":s[-1]['id']})
			p['tokens'] = words
			p['sentences'] = sents

			# Now augment the question
			for q in p['qas']:
				aug_question = coreNLP(q['question'], False)
				words = []
				q['tokens'] = aug_question[0]
			
			# Now, time to augment the answers
			for q in p['qas']:
				print q['tokens']
				for a in q['answers']:
					answer_start, answer_end = a['answer_start'], a['answer_end']
					start, end = len(p['tokens']), 0
					# print 'START, END:', answer_start, answer_end
					for w in p['tokens']:
						# print w['char_begin'], w['char_end'], answer_start, answer_end 
						if (w['char_begin'] >= answer_start) and w['char_end'] <= answer_end:
							start, end = min(start, w['id']), max(end, w['id'])
					if start > end :
						print '******** ERROR in *********\n',d['title'],'\n'# ,p['context'],'\n',q['question'],'\n',a['text'], start, '>' ,end
						quit()
					a['begin_id'], a['end_id'] = start, end
					# a['aug-answer-text'] = coreNLP(a['text'])
				q.pop('question')
			p.pop('context')
	os.chdir('../')

# def augment_data(data):
# 	os.chdir('./StanfordCoreNLP/')
# 	for d in tqdm(data['data']):
# 		for p in d['paragraphs']:
# 			Quetion_str = ''
# 			Answer_str = ''
			# Now augment the question
			# Questions = '' 
			# for q in p['qas']:
			# 	Questions = Questions + q['question'] + ' '
			# aug_question = coreNLP(Questions, False)
			# words = []
			# sents = []
			# for i,s in enumerate(aug_question):
			# 	p['qas'][i]['tokens'] = s

# 			for q in p['qas']:
# 				Quetion_str = Quetion_str + q['question'] + ''
# 				for a in q['answers']:
# 					Answer_str = Answer_str + a['text']
# 					# a['aug-answer-text'] = coreNLP(a['text'])
# 			p['aug-context'] = coreNLP(p['context'])
# 			q_parsed_list = coreNLP(Quetion_str) # q['question']
# 			a_parsed_list = coreNLP(Answer_str)
# 			for i in range(len(q_parsed_list)):
# 				p['qas'][i]['aug-question'] = q_parsed_list[i]
# 				print q_parsed_list[i]
# 			for i in range(len(a_parsed_list)):
# 				p['qas'][i]['answers'][0]['aug-answer_text'] = a_parsed_list[i]
# 				print a_parsed_list[i]
# 	os.chdir('../')

def write_out(target_path = 'PreProcessed_Data/augmented_train.txt'):
	with open(target_path,'w') as f:
	 	json.dump(data,f)

if __name__ == '__main__':
 	data = None
 	# Read Data
 	if len(sys.argv)>1:
 		data = read_data(sys.argv[1]) 
 	else:
 		data = read_data()
 	# Augment Data
 	augment_data(data)
 	# Writeout Data
 	if len(sys.argv)>2:
 		write_out(sys.argv[2]) 
 	else:
 		write_out()
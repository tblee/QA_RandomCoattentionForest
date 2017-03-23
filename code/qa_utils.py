## ==== Utility function to help process data ====
import numpy as np
from os.path import join as pjoin
from qa_data import PAD_ID

def load_glove(data_dir, glove_dim = 100):
	glove_dir = pjoin(data_dir, "glove.trimmed.{}.npz".format(glove_dim))
	glove_file = np.load(glove_dir)
	glove = glove_file['glove']

	return glove.astype(np.float32)

def prepare_data(data_dir, c_max_length, q_max_length, train_val = "train", sample_size = None):
	## obtain directories
	context_file_dir = pjoin(data_dir, "{}.context".format(train_val))
	context_id_file_dir = pjoin(data_dir, "{}.ids.context".format(train_val))
	question_file_dir = pjoin(data_dir, "{}.question".format(train_val))
	question_ids_file_dir = pjoin(data_dir, "{}.ids.question".format(train_val))
	span_file_dir = pjoin(data_dir, "{}.span".format(train_val))

	dirs = [context_file_dir, context_id_file_dir, question_file_dir, question_ids_file_dir, span_file_dir]
	keys = ["contexts", "context_ids", "questions", "question_ids", "spans"]

	data = {}
	## first read file content to memory
	for file_dir, key in zip(dirs, keys):
		lines_count = 0
		cur_buf = []
		with open(file_dir) as file:
			for line in file:
				if (sample_size is None) or (lines_count < sample_size):
					cur_buf.append(line)
				lines_count += 1
		data[key] = cur_buf

	## process original context and questions into array of strings
	contexts, questions = [], []
	for context_line, question_line in zip(data['contexts'], data['questions']):
		c = context_line.strip('\n').split(' ')
		q = question_line.strip('\n').split(' ')
		contexts.append(c)
		questions.append(q)

	## process context and question ids
	context_ids, context_masks, question_ids = [], [], []
	for context_line, question_line in zip(data['context_ids'], data['question_ids']):
		c_ids = map(int, context_line.strip('\n').split(' '))
		q_ids = map(int, question_line.strip('\n').split(' '))
		c_ids, c_mask = pad_and_trim_sentence_with_mask(c_ids, c_max_length)
		q_ids, _ = pad_and_trim_sentence_with_mask(q_ids, q_max_length)
		context_ids.append(c_ids)
		context_masks.append(c_mask)
		question_ids.append(q_ids)

	## process answer span and produce answer sentence
	start_ids, end_ids, answers = [], [], []
	for idx, span in enumerate(data['spans']):
		s_id, e_id = map(int, span.strip('\n').split(' '))
		s_id = min(s_id, c_max_length - 1)
		e_id = min(e_id, c_max_length - 1)
		ans = contexts[idx][s_id: e_id + 1]
		start_ids.append(s_id)
		end_ids.append(e_id)
		answers.append(ans)

	return contexts, questions, context_ids, context_masks, question_ids, start_ids, end_ids, answers

def pad_and_trim_sentence_with_mask(sentence, max_length):
	## input sentence is an array of integers
	mask = [True] * max_length
	n_sentence = sentence[:]
	if len(sentence) < max_length:
		n_sentence.extend([PAD_ID] * (max_length - len(sentence)))
		mask[len(sentence) : max_length] = [False] * (max_length - len(sentence))
	elif len(sentence) > max_length:
		n_sentence = n_sentence[: max_length]

	return n_sentence, mask

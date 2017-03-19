## ==== Implementation of Encoders and Decoders ====

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

from qa_model import Config


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        return


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        return

class BasicAffinityEncoder(Encoder):
	def __init__(self, config):
		self.config = config

	def encode(self, dataset):
		## use a dictionary to represent all inpus
		## contexts: word embeddings of context paragraph, shape = [None, c_max_length, state_size]
		## questions: word embeddings of questions, shape = [None, q_max_length, state_size]
		
		hidden_size = self.config.state_size
		c_max_length = self.config.c_max_length
		q_max_length = self.config.q_max_length
		#hidden_size = 128
		#c_max_length = 200
		#q_max_length = 30

		with vs.variable_scope("BasicAffinityEncoder"):

			context_input = dataset['contexts']
			question_input = dataset['questions']
			dropout = dataset['dropout']

			## compute basic context and question representation
			with vs.variable_scope("contextLSTM"):
				c_lstm = tf.nn.rnn_cell.LSTMCell(hidden_size)
				context, _ = tf.nn.dynamic_rnn(c_lstm, context_input, dtype = tf.float32, time_major = False)

			with vs.variable_scope("questionLSTM"):
				q_lstm = tf.nn.rnn_cell.LSTMCell(hidden_size)
				question, _ = tf.nn.dynamic_rnn(q_lstm, question_input, dtype = tf.float32, time_major = False)

				## transform question representation to allow for non-linearity
				W_q = tf.get_variable("W_question_nonlinearity",
					dtype = tf.float32,
					shape = [hidden_size, hidden_size],
					initializer = tf.contrib.layers.xavier_initializer())
				b_q = tf.get_variable("b_question_nonlinearity",
					dtype = tf.float32,
					shape = hidden_size,
					initializer = tf.constant_initializer(0.0))

				question = tf.reshape(question, [-1, hidden_size])
				question = tf.tanh( tf.matmul(question, W_q) + b_q )
				question = tf.reshape(question, [-1, q_max_length, hidden_size])


			## dropout
			context = tf.nn.dropout(context, 1.0 - dropout)
			question = tf.nn.dropout(question, 1.0 - dropout)

			## capture context and question interaction
			Z = batch_matmul(context, tf.transpose(question, [0, 2, 1]))
			A_Q = tf.nn.softmax(Z, dim = -1)
			A_P = tf.nn.softmax(Z, dim = 1)


			C_Q = batch_matmul(tf.transpose(A_Q, [0, 2, 1]), context)
			C_P = batch_matmul(A_P, tf.concat(2, [question, C_Q]))

			context_question = tf.concat(2, [context, C_P])

			with vs.variable_scope("encoderLSTMoutput"):
				out_lstm = tf.nn.rnn_cell.LSTMCell(hidden_size)
				context_attn, _ = tf.nn.dynamic_rnn(out_lstm, context_question, dtype = tf.float32, time_major = False)

			"""
			context_attn_concat = tf.concat(1, [C_P, context])
			context_attn_concat = tf.reshape(context_attn_concat, [-1, hidden_size * 2])

			with vs.variable_scope("interactions"):
				W = tf.get_variable("W_encoder_interaction",
					dtype = tf.float32,
					shape = [hidden_size * 2, hidden_size],
					initializer = tf.contrib.layers.xavier_initializer())
				b = tf.get_variable("b_encoder_interaction",
					dtype = tf.float32,
					shape = hidden_size,
					initializer = tf.constant_initializer(0.0))

				context_attn = tf.matmul(context_attn_concat, W) + b
				context_attn = tf.reshape(context_attn, [-1, c_max_length, hidden_size])
			"""

			return tf.nn.dropout(context_attn, 1.0 - dropout)
			#return context_attn

class BasicLSTMClassifyDecoder(Decoder):
	def __init__(self, config):
		self.config = config

	def decode(self, encoded):
		## encoded: encoded information with shape = [None, c_max_length, state_size]

		hidden_size = self.config.state_size
		c_max_length = self.config.c_max_length
		#hidden_size = 128
		#c_max_length = 200

		encoded = tf.reshape(encoded, [-1, hidden_size])

		with vs.variable_scope("startDecoder"):
			W_start = tf.get_variable("W_start_decoder",
				dtype = tf.float32,
				shape = [hidden_size, 1],
				initializer = tf.contrib.layers.xavier_initializer())

			start_preds = tf.matmul(encoded, W_start)
			start_preds = tf.reshape(start_preds, [-1, c_max_length])

		encoded = tf.reshape(encoded, [-1, c_max_length, hidden_size])

		with vs.variable_scope("endDecoder"):
			with vs.variable_scope("endDecoderLSTM"):
				c_lstm = tf.nn.rnn_cell.LSTMCell(hidden_size)
				encoded_e, _ = tf.nn.dynamic_rnn(c_lstm, encoded, dtype = tf.float32, time_major = False)

			W_end = tf.get_variable("W_end_decoder",
				dtype = tf.float32,
				shape = [hidden_size, 1],
				initializer = tf.contrib.layers.xavier_initializer())

			encoded_e = tf.reshape(encoded_e, [-1, hidden_size])

			end_preds = tf.matmul(encoded_e, W_end)
			end_preds = tf.reshape(end_preds, [-1, c_max_length])

		return start_preds, end_preds






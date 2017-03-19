
import numpy as np
import tensorflow as tf
from qa_encoder_decoder import *

encoder = BasicAffinityEncoder(None)
decoder = BasicLSTMClassifyDecoder(None)


c_len = 200
q_len = 30
embed_size = 100
batch_size = 21
dropout = 0.15

cp = tf.placeholder(tf.float32, [None, c_len, embed_size])
qp = tf.placeholder(tf.float32, [None, q_len, embed_size])

dataset = {}
dataset['contexts'] = cp
dataset['questions'] = qp
dataset['dropout'] = dropout

encoded = encoder.encode(dataset)
start_pred, end_pred = decoder.decode(encoded)

contexts = np.random.rand(batch_size, c_len, embed_size)
questions = np.random.rand(batch_size, q_len, embed_size)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	feed = {cp: contexts, qp: questions}
	s_p, e_p = sess.run([start_pred, end_pred], feed)

print(s_p)
print(e_p)
print(s_p.shape)
print(e_p.shape)



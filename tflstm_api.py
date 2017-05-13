# 20170506 run alone.Except one exception
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

n_hidden = 200
batch_size = 2
max_document_length = 5
embedded_dim = 8

embedded_matrix = np.float32(np.random.rand(batch_size, max_document_length, embedded_dim))
words = tf.placeholder(tf.float32, shape=[batch_size, max_document_length, embedded_dim])

num_steps = 5
q_dim = 128

lstm = tf.contrib.rnn.BasicLSTMCell(q_dim, state_is_tuple=False)

initial_state = lstm.zero_state(batch_size, dtype=tf.float32)

outputs, final_state = tf.nn.dynamic_rnn(lstm, words, sequence_length=None, initial_state=initial_state, dtype=None,time_major=False)

q_features=outputs[:,max_document_length-1,:]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # state = initial_state.eval()
    # print type(state)
    # print state
    # print 'outputs shape:',outputs
    print(sess.run(q_features,
                   feed_dict={
                       words: embedded_matrix
                   }))
    print 'outputs shape'

# def RNNLSTM(x,max_document_length,lstm_size):
#     sess = tf.get_default_session()
#     batch_size=x.get_shape().as_list()[0]
#     lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=False)
#     sess.run(tf.global_variables_initializer())
#     initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
#     print initial_state
#
#     print 'initial:',sess.run(initial_state)
#     sess.run(tf.global_variables_initializer())
#     outputs, final_state = tf.nn.dynamic_rnn(lstm, x, sequence_length=None, initial_state=initial_state, dtype=None,time_major=False)
#     final_output=outputs[:,max_document_length-1,:]
#     return final_output
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     final_output=RNNLSTM(words,max_document_length,lstm_size)
#     print 'final_output:',final_output
#     print sess.run(final_output,feed_dict={words:embedded_matrix})
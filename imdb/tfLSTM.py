import tensorflow as tf
import tfembedding
#x.shape=[batch_size,1,embedding_dim]
def LSTMCell(x, h, C_tminus1, batch_size, hidden_size):
    # Step1 forget_gate
    print 'LSTMCell Creating...',x
    print h
    print C_tminus1
    embedded_dim=x.get_shape().as_list()[2]
    lenth= embedded_dim + hidden_size
    print 'lenth of LSTMCell',lenth,type(lenth)
    tmp = tf.reshape(tf.concat([x, h], 2), [batch_size, 1, lenth])

    # sess.run(tf.global_variables_initializer())
    W1 = tf.Variable(tf.truncated_normal(shape=[batch_size, lenth, hidden_size], mean=0.0, stddev=1.0, dtype=tf.float32))
    #not shape[batch_size,1,hidden_size] for b1
    b1 = tf.Variable(tf.constant(0.2, shape=[1, hidden_size]), name="b1")
    res1 = tf.matmul(tmp, W1) + b1
    forget_gate = tf.sigmoid(tf.matmul(tmp, W1) + b1)
    # Step2 Decide what information we're going to store in the cell state
    # input_gate,decides which value we'll update
    W2 = tf.Variable(tf.constant(2.0, shape=[batch_size, lenth, hidden_size]), name="W2")
    b2 = tf.Variable(tf.constant(0.5, shape=[1, hidden_size]), name="b2")
    res2 = tf.matmul(tmp, W2) + b2
    input_gate = tf.sigmoid(tf.matmul(tmp, W2) + b2)

    # a tanh layer creates a vector of new candidate values,Ct_,that would be added to the state
    Wc = tf.Variable(tf.constant(2.0, shape=[batch_size, lenth, hidden_size]), name="Wc")
    bc = tf.Variable(tf.constant(0.5, shape=[1, hidden_size]), name="bc")
    res3 = tf.matmul(tmp, Wc) + bc
    Ct_candidate = tf.tanh(res3)

    # Step3 Update the old cell state,C_old,into the new cell state C_new
    Ct = tf.multiply(forget_gate, C_tminus1) + tf.multiply(input_gate, Ct_candidate)
    # Step4 decide what to output.This output will be based on our cell state,but will be a filtered version
    W4 = tf.Variable(tf.constant(2.0, shape=[batch_size, lenth, hidden_size]), name="W4")
    b4 = tf.Variable(tf.constant(0.5, shape=[1, hidden_size]), name="b4")
    # a sigmoid gate which decides what part of the cell state we're going to output
    Ot = tf.sigmoid(tf.matmul(tmp, W4) + b4)
    # We put the cell state through tanh(to push the values to be between -1 and 1)and multiply it by the output of the
    # sigmoid gate
    ht = tf.multiply(Ot, tf.tanh(Ct))
    #Don't forget to initialize all variables at last!
    # sess=tf.Session()
    # sess.run(tf.local_variables_initializer())
    print 'LSTMCell Done!'
    return ht,Ct

def LSTMUnroll(x, batch_size, hidden_size, max_document_length=20):
    ht = tf.Variable(tf.constant(0.0, shape=[batch_size, 1, hidden_size]))
    Ct = tf.Variable(tf.constant(0.0, shape=[batch_size, 1, hidden_size]))
    for i in range(max_document_length):
        print x[:,i:i+1,:]
        print x[:,i:i+1,:].shape
        ht, Ct = LSTMCell(x[:, i:i + 1, :], ht, Ct, batch_size, hidden_size)
        # sess=tf.Session()
        # sess.run(tf.local_variables_initializer())
        # print 'i', i, 'ht', sess.run(ht), 'Ct', sess.run(Ct)
    print 'LSTMNet successfully done!'
    return ht

def LSTMNet(x, batch_size, hidden_size, max_document_length=20):
    x = tfembedding.embedding(x, max_document_length)
    ht = LSTMUnroll(x, batch_size, hidden_size, max_document_length)
    return ht





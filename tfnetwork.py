import tensorflow as tf
import numpy as np
import testMLB
def FullyConnected(input,hidden_size,n_classes):
    print 'input shape',input.shape
    batch_size=input.get_shape().as_list()[0]
    input_dim=input.get_shape().as_list()[1]
    assert(input_dim==hidden_size)
    input=tf.reshape(input,shape=[batch_size,hidden_size])

    W=tf.Variable(tf.random_normal([hidden_size, n_classes]))
    b=tf.Variable(tf.random_normal([n_classes]))
    logits=tf.matmul(input,W)+b
    return logits

def Combine(img_features, q_features,dimg,dq,keep_prob):#???what if dq,dimg not the same?
    if dimg!= 84:
        # output layer
        fc3_w = tf.Variable(tf.truncated_normal(shape=(84, dimg), mean=0, stddev=1))
        fc3_b = tf.Variable(tf.zeros(dimg))
        fc3 = tf.matmul(img_features, fc3_w) + fc3_b
        fc3 = tf.nn.relu(fc3)
        fc3_drop = tf.nn.dropout(fc3, keep_prob=keep_prob)
        img_features=fc3_drop
    assert (dimg==dq)
    mixed_features = tf.multiply(img_features, q_features)
    return mixed_features

def Routine(mixed_features,output_dim,hidden_size,keep_prob=0.8,hidden1_units=200):
    weights = tf.Variable(
        tf.truncated_normal([hidden_size, hidden1_units],stddev=1.0 / np.sqrt(float(hidden_size))),name='weights')

    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')

    fc1 = tf.nn.relu(tf.matmul(mixed_features, weights) + biases)

    weights2 = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden1_units],stddev=1.0 / np.sqrt(float(hidden_size))),name='weights')

    biases2 = tf.Variable(tf.zeros([hidden1_units]),name='biases')

    fc2 = tf.nn.relu(tf.matmul(fc1, weights2) + biases2)

    fc2_drop=tf.nn.dropout(fc2,keep_prob=keep_prob)

    weights_o = tf.Variable(
        tf.truncated_normal([hidden1_units, output_dim],
                            stddev=1.0 / np.sqrt(float(hidden_size))))
    biases_o = tf.Variable(tf.zeros([output_dim]))

    logits = tf.matmul(fc2_drop, weights_o) + biases_o
    return logits


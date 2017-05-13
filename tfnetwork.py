import tensorflow as tf
import numpy as np
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

def Combine(img_features,q_features,pool_method):
    if pool_method==0:
        mixed_features = tf.multiply(img_features, q_features)
    return mixed_features

def Routine(mixed_features,output_dim,hidden_size,hidden1_units=200):
    weights = tf.Variable(
        tf.truncated_normal([hidden_size, hidden1_units],stddev=1.0 / np.sqrt(float(hidden_size))),name='weights')

    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')

    hidden1_features = tf.nn.relu(tf.matmul(mixed_features, weights) + biases)
    print mixed_features,weights,biases,hidden1_features
    weights_o = tf.Variable(
        tf.truncated_normal([hidden1_units, output_dim],
                            stddev=1.0 / np.sqrt(float(hidden_size))))
    biases_o = tf.Variable(tf.zeros([output_dim]))

    logits = tf.matmul(hidden1_features, weights_o) + biases_o
    return logits


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

def Combine(img_features, q_features,dimg,dq,dcommon,pool_method,keep_prob):#???what if dq,dimg not the same?
    #project img_features from lenet to dimg
    # img_features_dimg=img_features
    # if dimg!= 84:
    #     with tf.name_scope('img2dimg'):
    #         print 'img2dimg'
    #         # output layer
    #         fc3_w = tf.Variable(tf.truncated_normal(shape=(84, dimg), mean=0, stddev=1))
    #         fc3_b = tf.Variable(tf.zeros(dimg))
    #         fc3 = tf.matmul(img_features, fc3_w) + fc3_b
    #         fc3 = tf.nn.relu(fc3)
    #         fc3_drop = tf.nn.dropout(fc3, keep_prob=keep_prob)
    #         img_features_dimg=fc3_drop

    #project img from dimg to dcommon
    img_features_dcommon = img_features
    if  dimg!=dcommon:
        with tf.name_scope('dimg2d'):
            print 'dimg2d'
            fc_w = tf.Variable(tf.truncated_normal(shape=(dimg, dcommon), mean=0, stddev=1))
            fc_b = tf.Variable(tf.zeros(dcommon))
            fc = tf.matmul(img_features, fc_w) + fc_b
            fc = tf.nn.relu(fc)
            fc_drop = tf.nn.dropout(fc, keep_prob=keep_prob)
            img_features_dcommon=fc_drop
    #project qfeatures from dq to dcommon
    q_features_dcommon=q_features
    if dq!=dcommon:
        with tf.name_scope('dq2d'):
            projectq_w = tf.Variable(tf.truncated_normal(shape=(dq, dcommon), mean=0, stddev=1))
            projectq_b = tf.Variable(tf.zeros(dcommon))
            q_fc = tf.matmul(q_features, projectq_w) + projectq_b
            q_relu = tf.nn.relu(q_fc)
            q_drop = tf.nn.dropout(q_relu, keep_prob=keep_prob)
            q_features_dcommon=q_drop
    if pool_method==0:
        print 'Combine concat'
        mixed_features = tf.concat([img_features_dcommon, q_features_dcommon], 1, 'concat')
    elif pool_method==1:
        print 'Combine multiply'
        mixed_features = tf.multiply(img_features_dcommon, q_features_dcommon)
    elif pool_method==2:
        print 'Combine add'
        mixed_features=tf.add(img_features_dcommon,q_features_dcommon)
    else:
        mixed_features = tf.multiply(img_features_dcommon, q_features_dcommon)
    return mixed_features

def Routine(mixed_features, output_dim, dcommon, pool_method,keep_prob=0.5, hidden1_units=200):
    if pool_method==0:
        combine_size=dcommon*2
    elif pool_method==1:
        combine_size=dcommon
    elif pool_method==2:
        combine_size=dcommon
    else:
        combine_size=dcommon
    print 'Combine size:',combine_size
    with tf.name_scope('Routine'):
        weights = tf.Variable(
            tf.truncated_normal([combine_size, hidden1_units],mean=0,stddev=1),name='weights')

        biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')

        fc1 = tf.nn.relu(tf.matmul(mixed_features, weights) + biases)

        weights2 = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden1_units], mean=0,stddev=1),name='weights')

        biases2 = tf.Variable(tf.zeros([hidden1_units]),name='biases')

        fc2 = tf.nn.relu(tf.matmul(fc1, weights2) + biases2)

        fc2_drop=tf.nn.dropout(fc2,keep_prob=keep_prob)

        weights_o = tf.Variable(
            tf.truncated_normal([hidden1_units, output_dim],mean=0, stddev=1))
        biases_o = tf.Variable(tf.zeros([output_dim]))

        logits = tf.matmul(fc2_drop, weights_o) + biases_o
        return logits


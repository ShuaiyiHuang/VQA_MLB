import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

def LeNet(x,dim=3,output=10):
    print 'Lenet test~~'
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1': 6,
        'layer_2': 16,
        'layer_3': 120,
        'layer_f1': 84
    }

    #Layer1 convolutional.Input=32x32x3.Output=28x28x6
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, dim, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    #Activation
    conv1 = tf.nn.relu(conv1)
    #Pooling.Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #layer2 convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    #Activation
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print 'pool_2:',pool_2
    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    print 'flattern:',flatten

    # layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b
    #Activation
    fc1 = tf.nn.relu(fc1)

    #layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    # Activation.
    fc2 = tf.nn.relu(fc2)

    # layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, output), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(output))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

def padding(data,input_size):
    nbefore=(32-input_size)/2
    nafter=(32-input_size)/2
    data_padded=np.pad(data, ((0, 0), (nbefore, nafter), (nbefore, nafter), (0, 0)), 'constant')
    return data_padded

def LeNet_4(x,use_mlb,dim=3,img_dim=84,keep_prob=0.8):
    # Hyperparameters
    with tf.name_scope('Lenet'):
        mu = 0
        sigma = 0.1

        layer_depth = {
            'layer_1': 6,
            'layer_2': 16,
            'layer_3': 120,
            'layer_f1': 84
        }

        #layer1 convolutional
        conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, dim, 6], mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        #Activation
        conv1 = tf.nn.relu(conv1)
        #Pooling. Input = 28x28x6. Output = 14x14x6.
        pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        #layer 2: Convolutional. Output = 10x10x16.
        conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        #Activation.
        conv2 = tf.nn.relu(conv2)
        #Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        if use_mlb==1:
            print 'use_mlb',use_mlb,'pool_2:', pool_2
            return pool_2
        else:
            #Flatten. Input = 5x5x16. Output = 400.
            fc1 = flatten(pool_2)
            print 'flattern:',flatten
            #Layer 3: Fully Connected. Input = 400. Output = 120.
            fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
            fc1_b = tf.Variable(tf.zeros(120))
            fc1 = tf.matmul(fc1, fc1_w) + fc1_b
            #Activation.
            fc1 = tf.nn.relu(fc1)

            #layer 4: Fully Connected. Input = 120. Output = 84.
            fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
            fc2_b = tf.Variable(tf.zeros(84))
            fc2 = tf.matmul(fc1, fc2_w) + fc2_b
            #Activation.
            fc2 = tf.nn.relu(fc2)

            fc2_drop = tf.nn.dropout(fc2, keep_prob=keep_prob)
            return fc2_drop


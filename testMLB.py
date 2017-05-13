import tensorflow as tf
import numpy as np

batch_size=2
N=6
s=5
d=3
M=16
G=2

qfeatures=tf.placeholder(tf.float32,(batch_size,N))
ifeatures=tf.placeholder(tf.float32, (batch_size, s, s, M))


def MLB_test(ifeatures,qfeatures,s,N,M,d,G):
    # feature map,Input=[batch_size,s,s,M],Output=[batch_size,s,s,d]
    Vf = tf.Variable(tf.truncated_normal(shape=[1, 1, M, d], mean=0, stddev=1), name='Vf')
    img_conv2d = tf.nn.conv2d(ifeatures, Vf, strides=[1, 1, 1, 1], padding='VALID', name='img_conv2d')
    out_img = tf.tanh(img_conv2d, 'img_tanh')
    print out_img
    # question,Input=[batch_size,N],output=[batch_size,s,s,d]
    Uq = tf.Variable(tf.truncated_normal(shape=[N, d], mean=0, stddev=1), name='Uq')
    qfeatures_projected = tf.matmul(qfeatures, Uq)
    qfeatures_tanh = tf.tanh(qfeatures_projected, 'q_tanh')
    q_tiled = tf.tile(qfeatures_tanh, [1, s * s])
    out_q = tf.reshape(q_tiled, [batch_size, s, s, d])
    print out_q
    # element-wise product
    mix_features = tf.multiply(out_img, out_q)
    # Last step,Input=[batch_size,s,s,d],output=[batch_size,s,s,G]
    P = tf.Variable(tf.truncated_normal(shape=[1, 1, d, G], mean=0, stddev=1), name='P')
    att_map = tf.nn.conv2d(mix_features, P, strides=[1, 1, 1, 1], padding='VALID', name='attention_map')
    print att_map
    return att_map

att_map=MLB_test(ifeatures,qfeatures,s,N,M,d,G)

with tf.Session() as sess:
    test_qfeatures = np.random.rand(batch_size, N)
    test_ifeatures=np.random.rand(batch_size,s,s,M)
    sess.run(tf.global_variables_initializer())
    print sess.run(att_map, feed_dict={qfeatures: test_qfeatures, ifeatures: test_ifeatures})

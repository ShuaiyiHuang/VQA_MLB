import tensorflow as tf
import numpy as np

# batch_size=2
# N=6
# s=5
# d=3
# M=16
# G=1
# dim_out=2
#
# qfeatures=tf.placeholder(tf.float32,(batch_size,N))
# ifeatures=tf.placeholder(tf.float32, (batch_size, s, s, M))


def MLB_pool(ifeatures, qfeatures, s, N, M, d, G, batch_size):
    # feature map,Input=[batch_size,s,s,M],Output=[batch_size,s,s,d]
    print 'ifeatures in MLB_pool:',ifeatures
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
    att_map_raw = tf.nn.conv2d(mix_features, P, strides=[1, 1, 1, 1], padding='VALID', name='attention_map')
    # att_map=[]
    # for i in range(G):
    #     map=att_map_raw[:,:,:,i]
    #     print 'type map:',type(map)
    #     map2=tf.reshape(map,shape=[batch_size,s*s])
    #     map_softmax=tf.nn.softmax(map2)
    #     map_softmax2=tf.reshape(map_softmax,shape=[batch_size,s,s,1])
    #     att_map[i]=map_softmax2
    trans=tf.transpose(att_map_raw,[0,3,2,1])
    trans_reshape=tf.reshape(trans,shape=[batch_size,G,s*s])
    trans_softmax=tf.nn.softmax(trans_reshape)
    trans_back=tf.reshape(trans_softmax,shape=[batch_size,G,s,s])
    att_map=tf.transpose(trans_back,[0,3,2,1])
    # print 'trans',trans
    # print 'trans_reshape',trans_reshape
    # print 'att_map',att_map
    return att_map


# att_map=MLB_pool(ifeatures, qfeatures, s, N, M, d, G, batch_size)

#Output=[batch_size,M]
def MLB_concat_img(ifeatures, qfeatures, s, N, M, d, G, batch_size):
    att_map=MLB_pool(ifeatures, qfeatures, s, N, M, d, G, batch_size)
    att_img=[]
    for i in range(G):
        map=att_map[:,:,:,i:i+1]
        # print 'map:',map
        map_tiled=tf.tile(map,[1,1,1,M])
        att_features=tf.multiply(ifeatures,map_tiled)
        att_features_reshape = tf.reshape(att_features, shape=[batch_size, s * s, M])
        f = np.zeros(shape=[batch_size, M], dtype=np.float32)
        for j in range(s * s):
            cur = att_features_reshape[:, j, :]
            # print 'cur', cur
            f = tf.add(f, cur)
        att_img.append(f)
    print 'att_img:',att_img

    if G==1:
        concat_img=att_img[G-1]
    if G==2:
        concat_img=tf.concat([att_img[0],att_img[1]],1)
    print 'concat_img:',concat_img
    return concat_img

# concat_img=MLB_concat_img(ifeatures, qfeatures, s, N, M, d, G, batch_size)

def MLB_predict(ifeatures,qfeatures,s,N,M,d,G,batch_size,dim_out):
    #image part Input=[batch_size,M] Output=[batch_size,d]
    concat_img=MLB_concat_img(ifeatures, qfeatures, s, N, M, d, G, batch_size)

    Vv = tf.Variable(tf.truncated_normal(shape=[G*M, d], mean=0, stddev=1), name='Vv')
    img_projected=tf.matmul(concat_img,Vv)
    img_out=tf.nn.relu(img_projected)
    #question part Input=[batch_size,N] Output=[batch_size,d]
    Wq = tf.Variable(tf.truncated_normal(shape=[N, d], mean=0, stddev=1), name='Wq')
    q_projected=tf.matmul(qfeatures,Wq)
    q_out=tf.nn.relu(q_projected)
    #element-wise product
    mix_features=tf.multiply(q_out,img_out)
    #output layer Input=[batch_size,d] output=[batch_size,dim_out]
    Po=tf.Variable(tf.truncated_normal(shape=[d,dim_out], mean=0, stddev=1), name='Po')
    logits=tf.matmul(mix_features,Po)

    return logits
#
# logits=MLB_predict(qfeatures,ifeatures,s,N,M,d,G,batch_size,dim_out)
#
# with tf.Session() as sess:
#     test_qfeatures = np.random.rand(batch_size, N)
#     test_ifeatures=np.random.rand(batch_size,s,s,M)
#     sess.run(tf.global_variables_initializer())
#     # print sess.run(att_map, feed_dict={qfeatures: test_qfeatures, ifeatures: test_ifeatures})
#     # print 'concat_img',sess.run(concat_img, feed_dict={qfeatures: test_qfeatures, ifeatures: test_ifeatures})
#     # print 'img_out', sess.run(img_out, feed_dict={qfeatures: test_qfeatures, ifeatures: test_ifeatures})
#     # print 'q_out', sess.run(q_out, feed_dict={qfeatures: test_qfeatures, ifeatures: test_ifeatures})
#     print 'logits', sess.run(logits, feed_dict={qfeatures: test_qfeatures, ifeatures: test_ifeatures})
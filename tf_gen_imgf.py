import tfloader2
import numpy as np
import tensorflow as tf
import tfcifar2
import scipy.misc
import os
#train_prefix='../../myMLB/data/shapes_control-2x/train.large'
#val_prefix='../../myMLB/data/shapes_control-2x/val'
#test_prefix='../../myMLB/data/shapes_control-2x/test'

#data_dir = '../../myMLB/data/shapes/cifar_features/'
#train_prefix='../../myMLB/data/shapes/train.large'
#val_prefix='../../myMLB/data/shapes/val'
#test_prefix='../../myMLB/data/shapes/test'

data_dir = '../../myMLB/data/shapes_control-3x/cifarfeatures/'
train_prefix='../../myMLB/data/shapes_control-3x/train.large'
val_prefix='../../myMLB/data/shapes_control-3x/val'
test_prefix='../../myMLB/data/shapes_control-3x/test'


max_doclength=7

shapes_data=tfloader2.get_dataset(train_prefix,val_prefix,test_prefix,max_document_length=max_doclength)

X_train,y_train,q_train,ques_train= shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries, shapes_data.train.ques
X_validation,y_validation,q_validation,ques_validation= shapes_data.val.images, shapes_data.val.labels, shapes_data.val.queries, shapes_data.val.ques
X_test,y_test,q_test,ques_test= shapes_data.test.images, shapes_data.test.labels, shapes_data.test.queries, shapes_data.test.ques

len_train=len(X_train)
len_valid=len(X_validation)
len_test=len(X_test)
new_size=24

# Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic'or 'cubic').
#X_train_new=scipy.misc.imresize(X_train,(new_size,new_size))
#X_vlid_new=scipy.misc.imresize(X_validation,(new_size,new_size))
#X_test_new=scipy.misc.imresize(X_test,(new_size,new_size))



#main
y=tf.placeholder(tf.int32,name='length')
#train = tf.placeholder(tf.float32, (len_train, 30, 30,3),name='x_img')
valid = tf.placeholder(tf.float32, (len_valid, 30, 30,3),name='x_img')
test = tf.placeholder(tf.float32, (len_test, 30, 30,3),name='x_img')

with tf.Session() as sess:
#    print cifar_features
    #when put outside,error
#    cifar_features1=tfcifar2.inference(train,len_train)
    for i in range
    cifar_features2=tfcifar2.inference(valid,len_valid)
    cifar_features3=tfcifar2.inference(test,len_test)
    
#    cf_vect_train=sess.run(cifar_features1,feed_dict={train:X_train})
    cf_vect_valid=sess.run(cifar_features2,feed_dict={valid:X_validation})
    cf_vect_test=sess.run(cifar_features3,feed_dict={test:X_test})
    
#    print cf_vect_test.shape,cf_vect_valid.shape,cf_vect_train.shape
#    print cf_vect_test

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
#np.save(data_dir+'train_cifarimg_lg', cf_vect_train)
print cf_vect_valid
np.save(data_dir+'valid_cifarimg', cf_vect_valid)
np.save(data_dir+'test_cifarimg', cf_vect_test)




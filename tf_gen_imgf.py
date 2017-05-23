import tfloader2
import numpy as np
import tensorflow as tf
import tfcifar2
import scipy.misc

#train_prefix='../../myMLB/data/shapes_control-2x/train.large'
#val_prefix='../../myMLB/data/shapes_control-2x/val'
#test_prefix='../../myMLB/data/shapes_control-2x/test'

data_dir = '../../myMLB/data/shapes/'
train_prefix='../../myMLB/data/shapes/train.large'
val_prefix='../../myMLB/data/shapes/val'
test_prefix='../../myMLB/data/shapes/test'


max_doclength=7

shapes_data=tfloader2.get_dataset(train_prefix,val_prefix,test_prefix,max_document_length=max_doclength)

X_train,y_train,q_train,ques_train= shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries, shapes_data.train.ques
X_validation,y_validation,q_validation,ques_validation= shapes_data.val.images, shapes_data.val.labels, shapes_data.val.queries, shapes_data.val.ques
X_test,y_test,q_test,ques_test= shapes_data.test.images, shapes_data.test.labels, shapes_data.test.queries, shapes_data.test.ques

length=len(X_train)
new_size=24

# Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic'or 'cubic').
#X_train_new=scipy.misc.imresize(X_train,(new_size,new_size))
#X_vlid_new=scipy.misc.imresize(X_validation,(new_size,new_size))
#X_test_new=scipy.misc.imresize(X_test,(new_size,new_size))

print 'length',length


#main
x = tf.placeholder(tf.float32, (length, 30, 30,3),name='x_img')

with tf.Session() as sess:
#    print cifar_features
    #when put outside,error
    cifar_features=tfcifar2.inference(x,length)
    cf_vect_train=sess.run(cifar_features,feed_dict={x:X_train})
    cf_vect_valid=sess.run(cifar_features,feed_dict={x:X_validation})
    cf_vect_test=sess.run(cifar_features,feed_dict={x:X_test})



np.save(data_dir+'train_cifarvec_lg', cf_vect_train)
np.save(data_dir+'valid_cifarvec', cf_vect_valid)
np.save(data_dir+'test_cifarvec', cf_vect_test)

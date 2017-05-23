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

data_dir = '../../myMLB/data/shapes_control-2x/cifarfeatures-delete/'
train_prefix='../../myMLB/data/shapes_control-2x/train.large'
val_prefix='../../myMLB/data/shapes_control-2x/val'
test_prefix='../../myMLB/data/shapes_control-2x/test'


max_doclength=7

shapes_data=tfloader2.get_dataset(train_prefix,val_prefix,test_prefix,max_document_length=max_doclength)

X_train,y_train,q_train,ques_train= shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries, shapes_data.train.ques
X_validation,y_validation,q_validation,ques_validation= shapes_data.val.images, shapes_data.val.labels, shapes_data.val.queries, shapes_data.val.ques
X_test,y_test,q_test,ques_test= shapes_data.test.images, shapes_data.test.labels, shapes_data.test.queries, shapes_data.test.ques

len_train=len(X_train)
len_valid=len(X_validation)
len_test=len(X_test)
batch_size=128

# Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic'or 'cubic').
#X_train_new=scipy.misc.imresize(X_train,(new_size,new_size))
#X_vlid_new=scipy.misc.imresize(X_validation,(new_size,new_size))
#X_test_new=scipy.misc.imresize(X_test,(new_size,new_size))



#main

img = tf.placeholder(tf.float32, (batch_size, 30, 30,3),name='x_img')


with tf.Session() as sess:
#    print cifar_features
    #when put outside,error
    cifar_features=tfcifar2.inference(img,batch_size)
    all_features_train=[]
    for offset in range(0,len_train,batch_size):
#        print 'offset in train:',offset
        batch_X=X_train[offset:offset+batch_size]
        batch_features=sess.run(cifar_features,feed_dict={img:batch_X})
        if offset==0:
            all_features_train=batch_features
        else:
            all_features_train=np.concatenate((all_features_train,batch_features),axis=0)
#        print 'all_features_train shape',all_features_train.shape
        
    all_features_valid=[]
    for offset in range(0,len_valid,batch_size):
        batch_X=X_validation[offset:offset+batch_size]
        batch_features=sess.run(cifar_features,feed_dict={img:batch_X})
        if offset==0:
            all_features_valid=batch_features
        else:
            all_features_valid=np.concatenate((all_features_valid,batch_features),axis=0)
        
    all_features_test=[]
    for offset in range(0,len_test,batch_size):
        batch_X=X_test[offset:offset+batch_size]
        batch_features=sess.run(cifar_features,feed_dict={img:batch_X})
        if offset==0:
            all_features_test=batch_features
        else:
            all_features_test=np.concatenate((all_features_test,batch_features),axis=0)

    print 'all_features shape:',all_features_train.shape,all_features_valid.shape,all_features_test.shape
    
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
np.save(data_dir+'train_cifarimg_lg', all_features_train)
np.save(data_dir+'valid_cifarimg',all_features_valid)
np.save(data_dir+'test_cifarimg',all_features_test)




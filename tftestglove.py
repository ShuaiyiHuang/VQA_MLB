import tensorflow as tf
import tfargs
import tfembedding
import tfnetwork
import numpy as np
import tfloader
import pickle
import tflenet

tfargs.definition()
tfargs.embedded_dim=50
tfargs.use_glove=True
tfargs.is_embd_matrix_trainable=True
tfargs.max_doc_length=10
tfargs.batch_size=16
tfargs.hidden_size=84
tfargs.epochs=100
tfargs.rate=0.001
tfargs.n_classess=2
tfargs.vocab_size=14

dim=3
img_dim=84
pool_method=0
input_size=30

x = tf.placeholder(tf.float32, (None, 32, 32, dim))
ques=tf.placeholder(tf.int32)
y=tf.placeholder(tf.int64,(None))

train_prefix='../data/shapes/train.tiny'
val_prefix='../data/shapes/val'
test_prefix='../data/shapes/test'
shapes_data=tfloader.get_dataset(train_prefix,val_prefix,test_prefix,max_document_length=tfargs.max_doc_length)

X_train,y_train,q_train,ques_train= shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries, shapes_data.train.ques

tfembedding.embedding_prepare2(tfargs.max_doc_length,tfargs.use_glove,tfargs.is_emdb_matrix_trainable)

print ques_train[0,:]
print q_train[0]
raw_text1=['is green left of red','red']
testarr=np.array(raw_text1)
print testarr.shape,testarr
res1=np.array(list(tfargs.Pretrain.transform(testarr)))
print res1.shape,res1

testarr2=q_train[0:2]
print testarr2.shape,testarr2
res2=np.array(list(tfargs.Pretrain.transform(testarr2)))
print res2.shape,res2

###error Cannot feed value of shape (128, 5, 32, 30) for Tensor u'Placeholder:0', which has shape '(?, 32, 32, 3)'

# image = inputs[iimg, :, :, :].copy()
# # make it directly processable by CNN
# tmp = image[:, :, 0].copy()
# image[:, :, 0] = image[:, :, 2]
# image[:, :, 2] = tmp
# image = np.transpose(image, (2, 0, 1)) / 255.
# image = np.float32(image)

#error ValueError: Cannot feed value of shape (128, 5, 32, 30) for Tensor u'Placeholder:0', which has shape '(?, 32, 32, 3)'
# image = inputs[iimg, :, :, :].copy()
# image = np.transpose(image, (2, 0, 1)) / 255.
# image = np.float32(image)

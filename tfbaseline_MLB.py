import tensorflow as tf
import tfargs
import tfembedding
import tfnetwork
import numpy as np
import tfloader
import pickle
import tflenet
import testMLB
import argparse
from sklearn.utils import shuffle


tfargs.definition()
tfargs.embedded_dim=128
tfargs.use_glove=False
tfargs.is_embd_matrix_trainable=True
tfargs.max_doc_length=7
tfargs.batch_size=128
tfargs.q_dim=64
tfargs.epochs=200
tfargs.rate=0.001
tfargs.n_classess=2
# tfargs.vocab_size=14


dim=3
img_dim=84
use_mlb=1
input_size=30
#para for MLB
#grid size s*s
s=5
#feature vector length
M=16
#number of glimpse
G=2
d=128

x = tf.placeholder(tf.float32, (None, 32, 32, dim))
ques=tf.placeholder(tf.int32)
y=tf.placeholder(tf.int64,(None))

#imdb_data=tfimdbloader.load_imdb(max_length=tfargs.max_doc_length)

# shapes_data =pickle.load(open(dataroot))
# train,val,test=tfloader.load_shapes(data_root)
train_prefix='../data/shapes/train.large'
val_prefix='../data/shapes/val'
test_prefix='../data/shapes/test'

# train_prefix='../data/shapes_control-2x/train.large'
# val_prefix='../data/shapes_control-2x/val'
# test_prefix='../data/shapes_control-2x/test'

# train_prefix='../data/shapes_control-3x/train.large'
# val_prefix='../data/shapes_control-3x/val'
# test_prefix='../data/shapes_control-3x/test'

tfembedding.embedding_prepare(tfargs.max_doc_length,tfargs.use_glove,tfargs.is_emdb_matrix_trainable)

shapes_data=tfloader.get_dataset(train_prefix,val_prefix,test_prefix,max_document_length=tfargs.max_doc_length,use_glove=tfargs.use_glove)

X_train,y_train,q_train,ques_train= shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries, shapes_data.train.ques
X_validation,y_validation,q_validation,ques_validation= shapes_data.val.images, shapes_data.val.labels, shapes_data.val.queries, shapes_data.val.ques
X_test,y_test,q_test,ques_test= shapes_data.test.images, shapes_data.test.labels, shapes_data.test.queries, shapes_data.test.ques

#shuffle
X_train,y_train,q_train,ques_train=shuffle(X_train,y_train,q_train,ques_train)
X_validation,y_validation,q_validation,ques_validation=shuffle(X_validation,y_validation,q_validation,ques_validation)
X_test,y_test,q_test,ques_test=shuffle(X_test,y_test,q_test,ques_test)


#Padding to fit Lenet
X_train=tflenet.padding(X_train,input_size)
X_validation=tflenet.padding(X_validation,input_size)
X_test=tflenet.padding(X_test,input_size)

#LSTM


embedded_chars=tfembedding.get_embedded_from_wordid(ques)
lstm = tf.contrib.rnn.BasicLSTMCell(tfargs.q_dim, state_is_tuple=False)
initial_state = lstm.zero_state(tfargs.batch_size, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(lstm, embedded_chars, sequence_length=None, initial_state=initial_state, dtype=None,time_major=False)
q_features=outputs[:,tfargs.max_doc_length-1,:]
#****CNN***
img_features=tflenet.LeNet_4(x,use_mlb=use_mlb,dim=dim,img_dim=img_dim,)
print 'img_features in main:',img_features
#Combine
if use_mlb==0:
    mixed_features=tfnetwork.Combine(img_features, q_features, use_mlb)
    logits=tfnetwork.Routine(mixed_features, tfargs.n_classes, tfargs.q_dim)
if use_mlb==1:
    logits = testMLB.MLB_predict(img_features, q_features, s, tfargs.q_dim, M, d, G, tfargs.batch_size, tfargs.n_classes)

# logits=tfnetwork.FullyConnected(q_features,tfargs.hidden_size,tfargs.n_classes)
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate =tfargs.rate)
training_operation = optimizer.minimize(loss_operation)
argmax_logits=tf.argmax(logits, 1)

correct_prediction = tf.equal(tf.argmax(logits, 1),y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
num_examples=len(ques_train)
saver = tf.train.Saver()

# def evaluate(X_data, y_data, batch_size):
#     num_examples = len(X_data)
#     total_accuracy = 0
#     total_loss=0
#     sess = tf.get_default_session()
#     for offset in range(0, num_examples, batch_size):
#         batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
#         loss=sess.run(loss_operation, feed_dict={ques:batch_x, y:batch_y})
#         accuracy = sess.run(accuracy_operation, feed_dict={ques:batch_x, y: batch_y})
#         total_accuracy += (accuracy *len(batch_x))
#         total_loss+=(loss*len(batch_x))
#     mean_accuracy=total_accuracy/num_examples
#     mean_loss=total_loss/num_examples
#     #print('Total accuracy{:.3},num examples{},mean_accuracy{:.3}'.format(total_accuracy,num_examples,mean_accuracy))
#     #print('Total loss{:.3},num examples{},mean_loss{:.3}'.format(total_loss, num_examples, mean_loss))
#     return mean_accuracy,mean_loss

def evaluate(X_data, y_data,ques_data,batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss=0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y,batch_ques= X_data[offset:offset + batch_size], y_data[offset:offset + batch_size],ques_data[offset:offset + batch_size]
        loss=sess.run(loss_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques})
        accuracy = sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques})
        total_accuracy += (accuracy *len(batch_x))
        total_loss+=(loss*len(batch_x))
    mean_accuracy=total_accuracy/num_examples
    mean_loss=total_loss/num_examples
    return mean_accuracy,mean_loss


sess=tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    for i in range(tfargs.epochs):
        print('Epoch{}...'.format(i))
        #before each epoch,shuffle the training set
        X_train, y_train, q_train, ques_train = shuffle(X_train, y_train, q_train, ques_train)
        total_train_accuracy=0
        total_train_loss=0
        for offset in range(0,num_examples,tfargs.batch_size):
            batch_x=X_train[offset:offset+tfargs.batch_size]
            batch_ques=ques_train[offset:offset+tfargs.batch_size]
            batch_y=y_train[offset:offset+tfargs.batch_size]
            # print batch_y.shape,batch_y
            # print type(batch_y[0])
            # batch_ques=np.ones(shape=[tfargs.batch_size,tfargs.max_doc_length],dtype=int)
            # print 'np.ones',batch_ques.shape,batch_ques
            train_loss=sess.run(loss_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques})
            train_accuracy=sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques})

            sess.run(training_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques})
            total_train_accuracy += (train_accuracy * len(batch_ques))
            total_train_loss+=(train_loss*len(batch_ques))

        train_accuracy=total_train_accuracy/num_examples
        train_loss=total_train_loss/num_examples
        print('Train Accuracy= {:.3f}, loss = {:.3f} '.format(train_accuracy,train_loss))

        val_accuracy,val_loss=evaluate(X_validation,y_validation,ques_validation,tfargs.batch_size)
        print("Validation Accuracy = {:.3f} , loss = {:.3f} ".format(val_accuracy,val_loss))

        test_accuracy, test_loss = evaluate(X_test, y_test,ques_test, tfargs.batch_size)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    saver.save(sess, '../data/Models/baseline')
    print("LSTM Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('../data/Models'))

    test_accuracy2,test_loss2 = evaluate(X_test, y_test,ques_test,tfargs.batch_size)
    print("Test Accuracy = {:.3f}".format(test_accuracy2))

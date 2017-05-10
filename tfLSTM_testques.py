import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import tfembedding
import tfLSTM
import tfloader
import tfimdbloader
import tfargs

train_prefix='shapes/train.large'
val_prefix='shapes/val'
test_prefix='shapes/test'
max_document_length=10
shapes_data=tfloader.get_dataset(train_prefix,val_prefix,test_prefix,max_document_length)
# imdb_data=tfimdbloader.load_imdb(max_length=tfargs.max_doc_length)

batch_size=64
hidden_size=128
n_classes=2
rate=0.001
Epochs=20
embedded_dim=50

X_train,y_train,q_train,ques_train=shapes_data.train.images,shapes_data.train.labels,shapes_data.train.queries,shapes_data.train.ques
X_validation,y_validation,q_validation,ques_validation=shapes_data.val.images,shapes_data.val.labels,shapes_data.val.queries,shapes_data.val.ques
X_test,y_test,q_test,ques_test=shapes_data.test.images,shapes_data.test.labels,shapes_data.test.queries,shapes_data.test.ques

# ques_train=imdb_data.train.ques

y=tf.placeholder(tf.int64,(None))
ques=tf.placeholder(tf.int32)
#########LSTM######################

raw_embedded_chars=tfembedding.get_embedded_from_wordid(ques)
embedded_chars=tf.reshape(raw_embedded_chars,shape=[batch_size,max_document_length,embedded_dim])
# embedded_chars = np.float32(np.random.rand(batch_size, max_document_length, embedded_dim))
lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=False)
initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(lstm, embedded_chars, sequence_length=None, initial_state=initial_state, dtype=None,time_major=False)
q_features=outputs[:,max_document_length-1,:]
# q_featrues = tfLSTM.LSTMNet(batch_q, batch_size, hidden_size,max_document_length)
########3#Output layer#############
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

logits=FullyConnected(q_features,hidden_size,n_classes)
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
argmax_logits=tf.argmax(logits, 1)
# correct_prediction = tf.equal(argmax_logits,y)
# cast_correctpre=tf.cast(correct_prediction, tf.float32)
# accuracy_operation = tf.reduce_mean(cast_correctpre)
correct_prediction = tf.equal(tf.argmax(logits, 1),y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
num_examples=len(X_train)
saver = tf.train.Saver()

def evaluate(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss=0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        loss=sess.run(loss_operation,feed_dict={ques:batch_x,y:batch_y})
        accuracy = sess.run(accuracy_operation, feed_dict={ques:batch_x,y: batch_y})
        total_accuracy += (accuracy *len(batch_x))
        total_loss+=(loss*len(batch_x))
    mean_accuracy=total_accuracy/num_examples
    mean_loss=total_loss/num_examples
    #print('Total accuracy{:.3},num examples{},mean_accuracy{:.3}'.format(total_accuracy,num_examples,mean_accuracy))
    #print('Total loss{:.3},num examples{},mean_loss{:.3}'.format(total_loss, num_examples, mean_loss))
    return mean_accuracy,mean_loss


sess=tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    for i in range(Epochs):
        print('Epoch{}...'.format(i))
        num_examples=len(X_train)
        total_train_accuracy=0
        total_train_loss=0
        for offset in range(0,num_examples,batch_size):
            batch_q=q_train[offset:offset+batch_size]
            batch_ques=ques_train[offset:offset+batch_size]
            batch_y=y_train[offset:offset+batch_size]
            #print('batch {} q_features:'.format(offset),q_featrues,sess.run(q_featrues))

            # print('batch{} logits:'.format(offset),logits,sess.run(logits))
            # print('batch{} cross entropy:'.format(offset),cross_entropy,sess.run(cross_entropy,feed_dict={y:batch_y}))
            # print('batch{} batch_q:'.format(offset),batch_q)
            # print('batch{} batch_ques:'.format(offset),batch_ques)
            train_loss=sess.run(loss_operation, feed_dict={ques:batch_ques,y: batch_y})
            train_accuracy=sess.run(accuracy_operation,feed_dict={ques:batch_ques,y:batch_y})

            # print('batch{} y:'.format(offset), batch_y.shape, batch_y)
            # print('batch{} argmax_logits:'.format(offset), argmax_logits, sess.run(argmax_logits))
            # print('batch{} correct_prediction:'.format(offset), correct_prediction, sess.run(correct_prediction, feed_dict={y: batch_y}))

            # print('batch{} cast_correctpre:'.format(offset), cast_correctpre,
            #       sess.run(cast_correctpre, feed_dict={y: batch_y}))
            sess.run(training_operation,feed_dict={ques:batch_ques,y:batch_y})
            total_train_accuracy += (train_accuracy * len(batch_q))
            total_train_loss+=(train_loss*len(batch_q))

        train_accuracy=total_train_accuracy/num_examples
        train_loss=total_train_loss/num_examples
        print('Train Accuracy= {:.3f}, loss = {:.3f} '.format(train_accuracy,train_loss))

        val_accuracy,val_loss=evaluate(ques_validation,y_validation,batch_size)
        print("Validation Accuracy = {:.3f} , loss = {:.3f} ".format(val_accuracy,val_loss))

        test_accuracy, test_loss = evaluate(ques_test, y_test, batch_size)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    saver.save(sess, 'Models/0508/LSTM')
    print("LSTM Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./Models/0508'))

    test_accuracy2,test_loss2 = evaluate(ques_test, y_test,batch_size)
    print("Test Accuracy = {:.3f}".format(test_accuracy2))


import tensorflow as tf
import tfargs
import tfembedding
import tfnetwork
import numpy as np
import tfloader
import pickle

tfargs.definition()
tfargs.embedded_dim=50
tfargs.use_glove=False
tfargs.is_embd_matrix_trainable=True
tfargs.max_doc_length=10
tfargs.batch_size=64
tfargs.hidden_size=128
tfargs.epochs=200
tfargs.rate=0.001
tfargs.n_classess=2
tfargs.vocab_size=14
dataroot='./noglovedata/shapes.large.pkl'

y=tf.placeholder(tf.int64,(None))
x=tf.placeholder(tf.int32)

#imdb_data=tfimdbloader.load_imdb(max_length=tfargs.max_doc_length)

# shapes_data =pickle.load(open(dataroot))
# train,val,test=tfloader.load_shapes(data_root)
shapes_data=tfloader.get_dataset(max_document_length=tfargs.max_doc_length)

X_train,y_train,q_train,ques_train= shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries, shapes_data.train.ques
X_validation,y_validation,q_validation,ques_validation= shapes_data.val.images, shapes_data.val.labels, shapes_data.val.queries, shapes_data.val.ques
X_test,y_test,q_test,ques_test= shapes_data.test.images, shapes_data.test.labels, shapes_data.test.queries, shapes_data.test.ques

tfembedding.embedding_prepare(tfargs.max_doc_length,tfargs.use_glove,tfargs.is_emdb_matrix_trainable)

embedded_chars=tfembedding.get_embedded_from_wordid(x)
lstm = tf.contrib.rnn.BasicLSTMCell(tfargs.hidden_size, state_is_tuple=False)
initial_state = lstm.zero_state(tfargs.batch_size, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(lstm, embedded_chars, sequence_length=None, initial_state=initial_state, dtype=None,time_major=False)
q_features=outputs[:,tfargs.max_doc_length-1,:]



logits=tfnetwork.FullyConnected(q_features,tfargs.hidden_size,tfargs.n_classes)
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate =tfargs.rate)
training_operation = optimizer.minimize(loss_operation)
argmax_logits=tf.argmax(logits, 1)

correct_prediction = tf.equal(tf.argmax(logits, 1),y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
num_examples=len(ques_train)
saver = tf.train.Saver()

def evaluate(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss=0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        loss=sess.run(loss_operation,feed_dict={x:batch_x,y:batch_y})
        accuracy = sess.run(accuracy_operation, feed_dict={x:batch_x,y: batch_y})
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
    for i in range(tfargs.epochs):
        print('Epoch{}...'.format(i))

        total_train_accuracy=0
        total_train_loss=0
        for offset in range(0,num_examples,tfargs.batch_size):

            batch_ques=ques_train[offset:offset+tfargs.batch_size]
            # print type(batch_ques[0]),batch_ques[0,:]
            # print batch_ques.shape,batch_ques
            batch_y=y_train[offset:offset+tfargs.batch_size]
            # print batch_y.shape,batch_y
            # print type(batch_y[0])
            # batch_ques=np.ones(shape=[tfargs.batch_size,tfargs.max_doc_length],dtype=int)
            # print 'np.ones',batch_ques.shape,batch_ques
            train_loss=sess.run(loss_operation, feed_dict={x:batch_ques,y: batch_y})
            train_accuracy=sess.run(accuracy_operation,feed_dict={x:batch_ques,y:batch_y})

            sess.run(training_operation,feed_dict={x:batch_ques,y:batch_y})
            total_train_accuracy += (train_accuracy * len(batch_ques))
            total_train_loss+=(train_loss*len(batch_ques))

        train_accuracy=total_train_accuracy/num_examples
        train_loss=total_train_loss/num_examples
        print('Train Accuracy= {:.3f}, loss = {:.3f} '.format(train_accuracy,train_loss))

        val_accuracy,val_loss=evaluate(ques_validation,y_validation,tfargs.batch_size)
        print("Validation Accuracy = {:.3f} , loss = {:.3f} ".format(val_accuracy,val_loss))

        test_accuracy, test_loss = evaluate(ques_test, y_test, tfargs.batch_size)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    saver.save(sess, 'Models/0509/LSTM')
    print("LSTM Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./Models/0509'))

    test_accuracy2,test_loss2 = evaluate(ques_test, y_test,tfargs.batch_size)
    print("Test Accuracy = {:.3f}".format(test_accuracy2))

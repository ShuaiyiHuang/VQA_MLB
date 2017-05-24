import tfloader
import tensorflow as tf
import numpy as np
import tflenet
from tensorflow.examples.tutorials.mnist import input_data
import tfLSTM
EPOCHS = 1
BATCH_SIZE = 2
dim=3
rate = 0.001
input_size=30
output=10
q_dim=84
max_document_length=20
#After q*i
hidden1_units=200
output_dim=2

train_prefix='shapes/train.tiny'
val_prefix='shapes/val'
test_prefix='shapes/test'
shapes_data=tfloader.get_dataset(train_prefix,val_prefix,test_prefix)

X_train,y_train,q_train=shapes_data.train.images,shapes_data.train.labels,shapes_data.train.queries
X_validation,y_validation,q_validation=shapes_data.val.images,shapes_data.val.labels,shapes_data.val.queries
X_test,y_test,q_test=shapes_data.test.images,shapes_data.test.labels,shapes_data.test.queries

X_train=tflenet.padding(X_train,input_size)
X_validation=tflenet.padding(X_validation,input_size)
X_test=tflenet.padding(X_test,input_size)
# startid=0
batch_size=2
# batch_q=[]
# batch_x=X_train[startid:batch_size]
# batch_y=y_train[startid:batch_size]
batch_q=q_train[0:batch_size]

x = tf.placeholder(tf.float32, (None, 32, 32, dim))
y = tf.placeholder(tf.int64, (None))

# print batch_q.shape
cnn_features=tflenet.LeNet_4(x,dim,output)
lstm_features=tfLSTM.LSTMNet(batch_q, batch_size, q_dim, max_document_length)
cnn_feature_reshape=tf.reshape(cnn_features, shape=[batch_size, 1, q_dim])
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
# print cnn_features,sess.run(cnn_features, feed_dict={x:batch_x, y: batch_y})
# print 'cnn_feature_reshape',cnn_feature_reshape,sess.run(cnn_feature_reshape, feed_dict={x:batch_x, y: batch_y})
#
# print 'lstm_features',lstm_features,sess.run(lstm_features)

#
img_feature_updated=tf.multiply(cnn_feature_reshape,lstm_features)

img_feature_updated2=tf.reshape(img_feature_updated, shape=[batch_size, q_dim])
print 'img_feature_updated2 shape:',img_feature_updated2.shape
weights = tf.Variable(
    tf.truncated_normal([q_dim, hidden1_units],
                        stddev=1.0 / np.sqrt(float(q_dim))),
    name='weights')
print 'weights shape:',weights
biases = tf.Variable(tf.zeros([hidden1_units]),
                     name='biases')
print 'biases shape:',biases
res1=tf.matmul(img_feature_updated2,weights)
print 'res1 shape:',res1
hidden1 = tf.nn.relu(tf.matmul(img_feature_updated2, weights) + biases)
print 'hidden1 shape:',hidden1
#output
weights_o = tf.Variable(
    tf.truncated_normal([hidden1_units, output_dim],
                        stddev=1.0 / np.sqrt(float(q_dim))))
biases_o = tf.Variable(tf.zeros([output_dim]))

logits = tf.matmul(hidden1, weights_o) + biases_o

#calculate loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
correct_prediction = tf.equal(tf.argmax(logits, 1),y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        print ('val batch{} accuracy'.format(offset),sess.run(accuracy))
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print logits,'logits',sess.run(logits,feed_dict={x:batch_x, y: batch_y})
# print cross_entropy ,'cross_entropy ',sess.run(cross_entropy ,feed_dict={x:batch_x, y: batch_y})
# print loss_operation ,'loss_operation',sess.run(loss_operation ,feed_dict={x:batch_x, y: batch_y})

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num_examples = len(X_train)
#
#     print("Training...")
#     print()
#     for i in range(EPOCHS):
#         # X_train, y_train = shuffle(X_train, y_train)
#         for offset in range(0, 64, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#             batch_q=q_train[offset:end]
#             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
#             # print fc2
#             # print sess.run(fc2,feed_dict={x: batch_x, y: batch_y})
#             print sess.run(logits,feed_dict={x: batch_x, y: batch_y})
#             print sess.run(loss_operation,feed_dict={x: batch_x, y: batch_y})
#             print("Loss in batch {}:{}".format(offset,loss_operation))
#             train_accuracy = evaluate(batch_x, batch_y)
#             print("Train Accuracy = {:.3f}".format(train_accuracy))
#
#         print("EPOCH {} ...".format(i + 1))



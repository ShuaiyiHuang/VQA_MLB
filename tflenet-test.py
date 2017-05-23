import tfloader
import tensorflow as tf
import numpy as np
import numpy as np
import tflenet as tflenet
from tensorflow.examples.tutorials.mnist import input_data

EPOCHS = 10
BATCH_SIZE = 128
dim=1
rate = 0.001
input_size=28
output=10
#
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

# train_prefix='shapes/train.tiny'
# val_prefix='shapes/val'
# test_prefix='shapes/test'
# shapes_data=tfloader.get_dataset(train_prefix,val_prefix,test_prefix)
#
# X_train,y_train=shapes_data.train.images,shapes_data.train.labels
# X_validation,y_validation=shapes_data.val.images,shapes_data.val.labels
# X_test,y_test=shapes_data.test.images,shapes_data.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


# # Pad images with 0s
# X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# X_train = np.pad(X_train, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant')
# X_validation = np.pad(X_validation, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant')
# X_test = np.pad(X_test, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant')


X_train=tflenet.padding(X_train,input_size)
X_validation=tflenet.padding(X_validation,input_size)
X_test=tflenet.padding(X_test,input_size)

print("Updated Image Shape: {},label :{}".format(X_train[0].shape, type(y_train[0])))



from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

x = tf.placeholder(tf.float32, (None, 32, 32, dim))
y = tf.placeholder(tf.int64, (None))
# one_hot_y = tf.one_hot(y, 10)

# fc2=tflenet.LeNet_4(x,dim,output)
# print fc2

logits = tflenet.LeNet(x,dim,output)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
correct_prediction = tf.equal(tf.argmax(logits, 1),y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            # print 'offset',offset
            # print fc2
            # print sess.run(fc2,feed_dict={x: batch_x, y: batch_y})
            # print sess.run(logits,feed_dict={x: batch_x, y: batch_y})

        train_accuracy=evaluate(X_train,y_train)
        print("EPOCH {} ...".format(i + 1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        validation_accuracy = evaluate(X_validation, y_validation)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
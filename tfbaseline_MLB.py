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
import scipy.io as sio


tfargs.definition()
# tfargs.embedded_dim=50
# tfargs.use_glove=True
# tfargs.is_embd_matrix_trainable=False
# tfargs.max_doc_length=7
# tfargs.batch_size=16
# tfargs.q_dim=64
# tfargs.epochs=10
# tfargs.rate=0.001
# tfargs.n_classess=2
# tfargs.vocab_size=14

parser = argparse.ArgumentParser(description='tune VQA MLB baseline')



parser.add_argument('--gdp', type=float, default=0.25,
                    help='general dropout')
parser.add_argument('--dembd', type=int, default=50,
                    help='dimension for word embedding')
parser.add_argument('--dcommon', type=int, default=128,
                    help='q and img projected to the same dimension')
parser.add_argument('--dq', type=int, default=84,
                    help='dimension for question feature')
parser.add_argument('--dimg', type=int, default=84,
                    help='dimension for images CNN feature')
parser.add_argument('--use-mlb', type=int, default=0,
                    help='0 do not use mlb,1 use mlb')
parser.add_argument('--max-doclength', type=int, default=7,
                    help='max length for each question')
parser.add_argument('--epochs', type=int, default=12,
                    help='training epochs')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size for training epoch')
parser.add_argument('--use-glove', type=bool, default=True,
                    help='whether use glove')
parser.add_argument('--is-emtrainable', type=bool, default=False,
                    help='whether embedding matrix trainable')
parser.add_argument('--vocabs', type=int, default=400001,
                    help='if use-glove is true,vocabs=400001')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--n-class', type=int, default=2,
                    help='# output class')
parser.add_argument('--img-size', type=int, default=30,
                    help='size of input images')
parser.add_argument('--channel', type=int, default=3,
                    help='channel for input images')
parser.add_argument('--log-dir', type=str, default='../data/tensorboard',
                    help='directory for tensorboard')


#para for MLB
#grid size s*s
s=5
#feature vector length
M=16
#number of glimpse
G=2

args=parser.parse_args()

x = tf.placeholder(tf.float32, (None, 32, 32, args.channel))
ques=tf.placeholder(tf.int32)
y=tf.placeholder(tf.int64,(None))
keep_prob=tf.placeholder(dtype=tf.float32,name='gdp')

#imdb_data=tfimdbloader.load_imdb(max_length=tfargs.max_doc_length)

# shapes_data =pickle.load(open(dataroot))
# train,val,test=tfloader.load_shapes(data_root)

train_prefix='../data/shapes/train.small'
val_prefix='../data/shapes/val'
test_prefix='../data/shapes/test'

# train_prefix='../data/shapes_control-2x/train.large'
# val_prefix='../data/shapes_control-2x/val'
# test_prefix='../data/shapes_control-2x/test'

# train_prefix='../data/shapes_control-3x/train.large'
# val_prefix='../data/shapes_control-3x/val'
# test_prefix='../data/shapes_control-3x/test'

tfembedding.embedding_prepare(args.max_doclength,args.vocabs,args.use_glove,args.is_emtrainable,args.dembd)

shapes_data=tfloader.get_dataset(train_prefix,val_prefix,test_prefix,max_document_length=args.max_doclength,use_glove=args.use_glove)

X_train,y_train,q_train,ques_train= shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries, shapes_data.train.ques
X_validation,y_validation,q_validation,ques_validation= shapes_data.val.images, shapes_data.val.labels, shapes_data.val.queries, shapes_data.val.ques
X_test,y_test,q_test,ques_test= shapes_data.test.images, shapes_data.test.labels, shapes_data.test.queries, shapes_data.test.ques

#shuffle
X_train,y_train,q_train,ques_train=shuffle(X_train,y_train,q_train,ques_train)
X_validation,y_validation,q_validation,ques_validation=shuffle(X_validation,y_validation,q_validation,ques_validation)
X_test,y_test,q_test,ques_test=shuffle(X_test,y_test,q_test,ques_test)


#Padding to fit Lenet
X_train=tflenet.padding(X_train,args.img_size)
X_validation=tflenet.padding(X_validation,args.img_size)
X_test=tflenet.padding(X_test,args.img_size)

#LSTM
embedded_chars=tfembedding.get_embedded_from_wordid(ques,args.batch_size,args.max_doclength,args.dembd)
lstm = tf.contrib.rnn.BasicLSTMCell(args.dq, state_is_tuple=False)
initial_state = lstm.zero_state(args.batch_size, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(lstm, embedded_chars, sequence_length=None, initial_state=initial_state, dtype=None,time_major=False)
q_features=outputs[:,-1,:]
#****CNN***
img_features=tflenet.LeNet_4(x,use_mlb=args.use_mlb,dim=args.channel,img_dim=args.dimg,keep_prob=keep_prob)
# print 'img_features in main:',img_features
#Combine
if args.use_mlb==0:
    with tf.name_scope('Combine-0'):
        mixed_features=tfnetwork.Combine(img_features, q_features,args.dimg,args.dq,keep_prob)
        logits=tfnetwork.Routine(mixed_features, args.n_class, args.dq,keep_prob)
if args.use_mlb==1:
    with tf.name_scope('Combine-1'):
        logits = testMLB.MLB_predict(img_features, q_features, s, args.dq, M, args.dcommon, G, args.batch_size, args.n_class)

# logits=tfnetwork.FullyConnected(q_features,tfargs.hidden_size,tfargs.n_classes)
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate =args.lr)
training_operation = optimizer.minimize(loss_operation)
argmax_logits=tf.argmax(logits, 1)

correct_prediction = tf.equal(tf.argmax(logits, 1),y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
num_examples=len(ques_train)
saver = tf.train.Saver()

#for tensorboard
# tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.histogram('cross_entropy_histogram', cross_entropy)
tf.summary.scalar('loss operation', loss_operation)
merged=tf.summary.merge_all()


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

matpath='./matsmall/'
def evaluate(X_data, y_data,ques_data,batch_size, writer, merged, iternum):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss=0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y,batch_ques= X_data[offset:offset + batch_size], y_data[offset:offset + batch_size],ques_data[offset:offset + batch_size]
        loss=sess.run(loss_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques,keep_prob:1.0})
        accuracy = sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques,keep_prob:1.0})
        summary=sess.run(merged,feed_dict={x:batch_x, y:batch_y,ques:batch_ques,keep_prob:1.0})
        mixf=sess.run(mixed_features,feed_dict={x:batch_x, y:batch_y,ques:batch_ques,keep_prob:1.0})
        mixf_matrix=np.asmatrix(mixf)
        mat_str=matpath+'iternum'+str(iternum)
        sio.savemat(mat_str,{'feature':mixf_matrix})
        writer.add_summary(summary, iternum)
        total_accuracy += (accuracy *len(batch_x))
        total_loss+=(loss*len(batch_x))
    mean_accuracy=total_accuracy/num_examples
    mean_loss=total_loss/num_examples
    return mean_accuracy,mean_loss

sess=tf.Session()
train_writer=tf.summary.FileWriter(args.log_dir+'/train',sess.graph)
valid_writer=tf.summary.FileWriter(args.log_dir+'/valid')
test_writer=tf.summary.FileWriter(args.log_dir+'/test')

with sess.as_default():
    sess.run(tf.global_variables_initializer())
    iternum=0
    for i in range(args.epochs):
        print('Epoch{}...'.format(i))
        #before each epoch,shuffle the training set
        X_train, y_train, q_train, ques_train = shuffle(X_train, y_train, q_train, ques_train)
        total_train_accuracy=0
        total_train_loss=0
        for offset in range(0,num_examples,args.batch_size):

            batch_x=X_train[offset:offset+args.batch_size]
            batch_ques=ques_train[offset:offset+args.batch_size]
            batch_y=y_train[offset:offset+args.batch_size]
            # print batch_y.shape,batch_y
            # print type(batch_y[0])
            # batch_ques=np.ones(shape=[tfargs.batch_size,tfargs.max_doc_length],dtype=int)
            # print 'np.ones',batch_ques.shape,batch_ques
            train_loss=sess.run(loss_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques,keep_prob:args.gdp})
            train_accuracy=sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques,keep_prob:args.gdp})

            sess.run(training_operation, feed_dict={x:batch_x, y:batch_y,ques:batch_ques,keep_prob:args.gdp})
            total_train_accuracy += (train_accuracy * len(batch_ques))
            total_train_loss+=(train_loss*len(batch_ques))
            summary=sess.run(merged,feed_dict={x:batch_x, y:batch_y,ques:batch_ques,keep_prob:args.gdp})
            train_writer.add_summary(summary,iternum)
            iternum=iternum+1
        train_accuracy=total_train_accuracy/num_examples
        train_loss=total_train_loss/num_examples
        print('Train Accuracy= {:.3f}, loss = {:.3f} '.format(train_accuracy,train_loss))

        val_accuracy,val_loss=evaluate(X_validation,y_validation,ques_validation,args.batch_size,valid_writer,merged,iternum)
        print("Validation Accuracy = {:.3f} , loss = {:.3f} ".format(val_accuracy,val_loss))

        test_accuracy, test_loss = evaluate(X_test, y_test,ques_test, args.batch_size,test_writer, merged, iternum)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    train_writer.close()
    valid_writer.close()
    test_writer.close()
    saver.save(sess, '../data/Models/baseline')
    print("LSTM Model saved")



#with tf.Session() as sess:
    #saver.restore(sess, tf.train.latest_checkpoint('../data/Models'))

    #test_accuracy2,test_loss2 = evaluate(X_test, y_test,ques_test,args.batch_size)
    #print("Test Accuracy = {:.3f}".format(test_accuracy2))

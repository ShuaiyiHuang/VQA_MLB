import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import tfloader
import tfargs


def loadGloVe(filename='glove.6B/glove.6B.50d.txt'):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd

def embedding(x_text,max_document_length=20):
    vocab,embd=loadGloVe()
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding_matrix_str = np.asarray(embd)
    embedding_matrix=tf.string_to_number(embedding_matrix_str,name='ToFloat')
    # print 'str_matrix 12',embedding_matrix[12]
    print 'Embedding matrix loaded successful!'
    # embedding_tensor = tf.Variable(embedding, trainable=False, name="W")
    embedding_tensor = tf.Variable(tf.constant(1.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")
    embedding_tensor=embedding_tensor.assign(embedding_matrix)

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    pretrain = vocab_processor.fit(vocab)
    print 'getting word_ids matrix...:'
    word_ids = np.array(list(pretrain.transform(x_text)))
    print word_ids
    embedded_chars = tf.nn.embedding_lookup(embedding_tensor, word_ids)
    print 'embedded vectors successfully !:',embedded_chars
    # print 'embedded chars',sess.run(embedded_chars)
    # print sess.run(embedded_chars.dtype)
    return embedded_chars

def embedding_prepare(max_document_length=10,use_glove=True,trainable=False):
    print 'use_glove:',use_glove
    if use_glove:
        vocab,embd=loadGloVe()
        vocab_size = len(vocab)
        embedding_dim = len(embd[0])
        tfargs.embedded_dim=embedding_dim
        tfargs.vocab_size=vocab_size

        embedding_matrix = np.asarray(embd)
        tfargs.Embedding_tensor = tf.Variable(tf.constant(0.1, shape=[vocab_size, embedding_dim]),
                                          trainable=trainable, name="W")
        tfargs.Embedding_tensor=tfargs.Embedding_tensor.assign(embedding_matrix)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        tfargs.Pretrain= vocab_processor.fit(vocab)
    else:
        tfargs.Embedding_tensor = tf.Variable(tf.random_normal(shape=[tfargs.vocab_size, tfargs.embedded_dim]),trainable=True, name="W")
        print 'Not use glove,embdtensor is :',tfargs.Embedding_tensor

    return

def get_wordids(x_text):
    word_ids=np.array(list(tfargs.Pretrain.transform(x_text)))
    return word_ids

def get_embedded_from_wordid(word_ids):
    print 'word_ids:',word_ids
    embedded_chars_unshape=tf.nn.embedding_lookup(tfargs.Embedding_tensor, word_ids)
    print 'embedded_unshape',embedded_chars_unshape
    embedded_chars = tf.reshape(embedded_chars_unshape, shape=[tfargs.batch_size,tfargs.max_doc_length, tfargs.embedded_dim])
    return embedded_chars

def get_embedded_chars(x_text):
    print x_text
    word_ids=np.array(list(tfargs.Pretrain.transform(x_text)))
    embedded_chars=tf.nn.embedding_lookup(tfargs.Embedding_tensor, word_ids)
    return embedded_chars

#not use now
def LSTM(x,lstm_size=84,num_steps=5,batch_size=2):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=False)
    initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm, x, sequence_length=None, initial_state=initial_state, dtype=None,
                                             time_major=False)
    return outputs

#not use now
def LSTM_net(x,lstm_size=84,num_steps=5,batch_size=2):
    embedded=embedding(x,num_steps)
    outputs=LSTM(embedded,lstm_size,num_steps,batch_size)
    return outputs

if __name__=="__main__":
    train_prefix = 'shapes/train.tiny'
    val_prefix = 'shapes/val'
    test_prefix = 'shapes/test'
    shapes_data = tfloader.get_dataset(train_prefix, val_prefix, test_prefix)
    X_train, y_train, q_train = shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries
    # text = [['is green right of red'], ['is red left of green']]
    #
    #
    # with tf.Session() as sess:
    #     print 'get', sess.run(get_embedded_chars(q_train[0:3]))

    vocab,embd=loadGloVe()
    word_ids=[]
    print type(vocab)
    print type(vocab[0])
    index=0
    for i in vocab:
        if i=='is':
            word_ids.append(index)
        index+=1
    print vocab[14]
    print embd[14]
    print word_ids
    test=[['is green']]
    print test
    test2=q_train[:2]
    print test2
    with tf.Session() as sess:
        embedded_chars=get_embedded_chars(test2)
        print 'get', embedded_chars,sess.run(embedded_chars)




